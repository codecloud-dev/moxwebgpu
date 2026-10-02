# moxwebgpu 架构深潜

> 本文档面向想读懂源码、扩展算子或复现踩坑的读者。用户视角的介绍见 [README.md](../README.md)。
>
> 阅读顺序建议:**§1 模块地图 → §2 一次执行的旅程 → §3 内存与 uniform → §4 调度器 → §5 OpDef 约定 → §6 踩坑实录**。

---

## 目录

- [1. 模块地图](#1-模块地图)
- [2. 一次执行的完整旅程](#2-一次执行的完整旅程)
- [3. 内存与 uniform 字节布局](#3-内存与-uniform-字节布局)
- [4. 调度器:拓扑执行与引用计数回收](#4-调度器拓扑执行与引用计数回收)
- [5. OpDef 约定:如何添加一个新算子](#5-opdef-约定如何添加一个新算子)
- [6. BufferPool 与 PipelineCache 设计](#6-bufferpool-与-pipelinecache-设计)
- [7. 无 GPU 环境的 WebGPU:SwiftShader 配方](#7-无-gpu-环境的-webgpuswiftshader-配方)
- [8. 踩坑实录:六个真实 bug 及其修复](#8-踩坑实录六个真实-bug-及其修复)

---

## 1. 模块地图

依赖关系自上而下,**没有循环依赖**:

```text
                    ┌─────────────────────┐
                    │      index.ts       │  统一出口:installTensorOps + 版本号
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌──────────────┐
     │ tensor/    │   │  graph/    │   │   core/      │
     │ tensor.ts  │   │  lazy.ts   │   │  context.ts  │◄─── mox.init()
     │ ops/*      │   │ scheduler  │   │  buffer.ts   │
     │ codegen.ts │   │ pipeline   │   │  dtype.ts    │
     └─────┬──────┘   │  Cache.ts  │   │  kernel.ts   │
           │          └─────┬──────┘   └──────┬───────┘
           │                │                 │
           └──── 依赖 ──────┴────── 依赖 ─────┘
                            ▼
                     @webgpu/types(仅类型,运行时零依赖)
```

| 文件 | 职责 | 关键导出 |
| ---- | ---- | -------- |
| `core/dtype.ts` | dtype 注册表与 TypedArray 工厂 | `dtypeInfo` `DTYPES` `numElements` |
| `core/buffer.ts` | GPUBuffer 包装 + 幂次桶内存池 | `GpuDataBuffer` `BufferPool` |
| `core/context.ts` | 组装 device/queue/pool/pipelines/scheduler | `MoxContext` |
| `core/kernel.ts` | 原始 WGSL 逃生舱 | `Kernel` |
| `graph/lazy.ts` | 惰性图数据结构 + 拓扑排序 | `LazyNode` `OpDef` `KernelStep` `BindingSpec` `topoSort` |
| `graph/pipelineCache.ts` | WGSL 哈希 → pipeline 缓存 + 编译错误防御 | `PipelineCache` |
| `graph/scheduler.ts` | 拓扑执行、绑定解析、引用计数回收、读回 | `Scheduler` |
| `tensor/tensor.ts` | 用户手中的 Tensor 类 | `Tensor` |
| `tensor/codegen.ts` | 全部 WGSL 模板 + uniform 编码器 | `*Wgsl` `encode*Uniform` `fmtF32` |
| `tensor/ops/index.ts` | 把全部算子挂到 `Tensor.prototype` | `installTensorOps` |
| `tensor/ops/*.ts` | 各算子家族的 OpDef 定义 | `reduceDefs` 等 |

设计铁律:

1. **算子是纯元数据**。`OpDef` 只做形状推导和 codegen,不持有任何 GPU 状态 → 相同 `(op, attrs)` 的节点天然共享管线。
2. **GPU 侧只有三种资源被创建**:storage buffer(池)、uniform buffer(池)、pipeline(缓存)。staging buffer 唯一例外,读回即毁。
3. **运行时零 npm 依赖**,浏览器构建 IIFE 约 30 KB。

---

## 2. 一次执行的完整旅程

以 `gpu.tensor([1,2,3,4]).add(1).sum().item()` 为例。

### 2.1 建图阶段(同步、零 GPU 开销)

```text
gpu.tensor([1,2,3,4])
  └─ Tensor { data: GpuDataBuffer(上传), shape:[4], dtype:'f32' }
.add(1)
  └─ LazyNode { op: binary-scalar-add, attrs:{scalar:1}, inputs:[leaf],
                shape:[4], dtype:'f32', buffer:null, consumers:0 }
.sum()
  └─ LazyNode { op: global-reduce-sum, inputs:[add 节点],
                shape:[1], dtype:'f32', buffer:null, consumers:0 }
```

- `createNode()` 立刻做 `op.outShape()` 形状推导,形状错误**在建图期**抛出,而不是等到 GPU 执行。
- 叶子张量被包成 `{ kind:'leaf', tensor }`,避免 Tensor 直接引用图结构。

### 2.2 触发阶段(`.item()` → `Scheduler.materialize()`)

```text
topoSort(sum 节点)
  → [add 节点, sum 节点](后序 DFS,顺便累加 consumers)

runNode(add 节点):
  plan = op.build([[4]], ['f32'], {scalar:1})
       → 1 个 step:binaryScalarWgsl('f32', a+b)
  output = pool.acquire(4, 'f32')            // 16B → 16B 桶
  encoder.beginComputePass()
    uniform: encodeNUniform(4, 1, 0) → pool.acquireUniform(16)
    binding0=UBO binding1=input binding2=output
    dispatchWorkgroups(1)                    // ceil(4/64)
  queue.submit([...])
  → releaseUniform(ubo)
  → add 节点.buffer = output

runNode(sum 节点):
  plan → 2 个 step(两阶段树形归约)
  step0:phase1 写 partial(nWg 个部分和,来自池)
  step1:phase2 把 partial 归约到 out[0]
  queue.submit([...])
  → releaseUniform ×2
  → add 节点.consumers 1→0 → pool.release(add 的输出)  ★中间结果回收
  → sum 节点.buffer = output(1 元素)

readback(sum 节点.buffer)
  → 创建 staging → copyBufferToBuffer → mapAsync → Float32Array → 摧毁 staging
```

### 2.3 幂等性

`materialize()` 是幂等的:节点已有 `buffer` 就直接返回。所以:

```ts
const t = gpu.tensor(big).sum();   // 建图
await t.toArray();                  // 第一次执行
await t.item();                     // 直接读缓存,不重算
```

---

## 3. 内存与 uniform 字节布局

### 3.1 三种 buffer 的 usage

| 类型 | usage | 生命周期 |
| ---- | ----- | -------- |
| storage | `STORAGE \| COPY_SRC \| COPY_DST` | 池化,跨 dispatch 复用 |
| uniform | `UNIFORM \| COPY_DST` | 池化,**同队列时间线**内复用安全 |
| staging | `MAP_READ \| COPY_DST` | 一次读回后立即 destroy |

### 3.2 uniform 块字节表(codegen.ts 与编码器必须逐字节一致)

**① uniN / uni2D —— 16 B**(1D 逐元素、全局归约、2D 广播、按轴归约、softmax、transpose、matmul 维度变体)

```text
偏移   字段         WGSL 类型   写入 API
0      n / rows     u32         dv.setUint32(0, n, true)
4      _p0 / cols   u32         dv.setUint32(4, …, true)
8      scalar       f32         dv.setFloat32(8, scalar, true)
12     identity     f32         dv.setFloat32(12, identity, true)
```

- `scalar`:算子参数(加几、除以几)。mean 按轴归约把 `cols` 填进来。
- `identity`:归约初值。**为什么运行时传?见 §8.2。**

**② uniCopy —— 64 B**(slice / concat / 通用 ND gather)

```text
偏移    字段          WGSL 类型     内容
 0..15  outShape      vec4<u32>     输出形状(不足补 0)
16..31  inStrides     vec4<u32>     输入行主 strides
32..47  outStrides    vec4<u32>     输出行主 strides
48      inOffset      u32           输入起始偏移(元素)
52      outOffset     u32           输出起始偏移(元素)
56      total         u32           输出元素总数
60      rank          u32           实际维度数(≤4)
```

- 同一个 shader 覆盖两种场景:slice = 输入 strided / 输出 dense;concat = 输入 dense+偏移 / 输出 strided+偏移。
- **为什么是 vec4 而不是数组?见 §8.3。**

**③ Dims(matmul)—— 16 B**:`m:u32@0, n:u32@4, k:u32@8, _p:u32@12`。

### 3.3 WGSL 模板约定

所有模板遵守同一套约定,`Scheduler` 据此机械解析绑定:

- `@binding(0)` 恒为 uniform 块;
- 其余按顺序:input0、input1?(二元)、output(rw);
- `@workgroup_size(64)`(1D)/ `(16,16)`(2D tile);
- 全部行主序;
- 越界保护:`if (i >= uniforms.n) { return; }` —— dispatch 网格永远向上取整。

### 3.4 归约算法细节

**两阶段树形归约**(全局 sum/mean/max/min,`WORKGROUP_SIZE=64`,`REDUCE_CHUNK=8`):

```text
n 个元素
  ├─ phase1:⌈n/(64·8)⌉ 个 workgroup
  │    每个 lane strided 读 8 个元素 → 组内树形归约(32→16→…→1)
  │    → partial[wg]
  └─ phase2:1 个 workgroup 把 partial 折叠到 out[0](mean 在此除以 n)
```

- 每个 lane 一次读 **8 个连续跨步元素**(`idx = start + l.x + c*64`),内存合并友好;
- 树形归约用 `workgroupBarrier()` 而不是 `workgroupBarrier` 之外的任何同步——WGSL 中 barrier 只能出现在 uniform control flow,所以用 `loop + if` 而非递归分治。

**按最后一轴归约 / softmax**:1 个 workgroup 负责一整行(行长 ≤ 64 直接覆盖,>64 时 lane 循环 strided 累加),`workgroup_id` 即行号。

**argmax/argmin**:phase1 写**候选下标**(u32)而非值;phase2 用下标回读原值再比较——因为"首见优先"要求比较发生在原数组上,`partial` 里的下标无法表达平局时的先后。

### 3.5 matmul

经典 16×16 分块:`tileA/tileB` 两块 workgroup 内存,`tiles = ⌈k/16⌉` 轮循环,每轮协作加载、barrier、16 次 FMA。越界 tile 补零,因此非 16 倍数维度天然正确。

---

## 4. 调度器:拓扑执行与引用计数回收

### 4.1 BindingSpec 协议

每个 `KernelStep` 声明 `bindings: BindingSpec[]`,与 WGSL 声明顺序 1:1:

| BindingSpec | 解析为 |
| ----------- | ------ |
| `{ kind:'uniform' }` | 新 UBO,`step.uniforms()` 编码 → 池 → `queue.writeBuffer` |
| `{ kind:'read', input: i }` | 第 i 个算子输入的 buffer |
| `{ kind:'read', temp: true }` | 上一个 step 写的临时 buffer |
| `{ kind:'rw', output: true }` | 本算子的最终输出 |
| `{ kind:'rw', temp: true }` | 新临时 buffer,成为下一 step 的 `prevTemp` |

特例:`concat` 的每个 step 都显式声明 `output: true`——多 chunk 各自直写输出的不同偏移,`scheduler.ts` 用

```ts
const writesOutput =
  isLast || step.bindings.some((b) => b.kind === 'rw' && b.output === true);
```

识别这种情况,避免每个 chunk 各自 acquire 一块完整输出。

### 4.2 执行与回收的时序

```text
for node in topoOrder:
    runNode(node):
        1. materialize 叶子输入(上传数据)
        2. plan = op.build(shapes, dtypes, attrs)
        3. output = pool.acquire(outElements)
        4. 单个 CommandEncoder 内:每个 step 一个 compute pass
        5. queue.submit 一次
        6. 回收(同一函数内,submit 之后立刻执行):
           - 全部 scratch UBO → releaseUniform
           - 每个图输入 consumers--;归零且仍有 buffer → release
           - prevTemp(最后一个中间临时)→ release
    node.buffer = output
```

**为什么 UBO 可以 submit 后立即归还?** WebGPU 的 `queue.writeBuffer` 与 command buffer 消费在同一队列时间线上是按序的——归还进池后,下一个 `writeBuffer` 要么发生在更晚的提交里,要么 GPU 端早已完成读取。严格说,跨 `queue.submit` 的即时复用依赖实现按序执行;若未来出现乱序后端,只需给 release 加一个 fence 回调即可,池接口无需变化。

**引用计数何时建立?** `topoSort` 的 DFS 里 `input.consumers++`。同一张量被读两次(如 `t.add(t)`),计数为 2,只有两处都执行完才回收。

---

## 5. OpDef 约定:如何添加一个新算子

一个算子 = 一个 `OpDef` + 若干 `KernelStep`。完整五步示例(以全局 mean 为例):

```ts
// ① 定义 OpDef —— 纯元数据,无 GPU 状态
const meanDef: OpDef = {
  name: 'mean',
  outShape: () => [1],                    // 全局归约 → [1]
  // outDtype 省略 = 继承第一个输入的 dtype
  build: (shapes, dtypes) => ({
    steps: [
      { // ② phase1:树形部分和
        key: `reduce-p1-${dtypes[0]}`,
        wgsl: reducePhase1Wgsl(dtypes[0], (a, x) => `${a} + ${x}`),
        bindings: [
          { kind: 'uniform' },
          { kind: 'read', input: 0 },
          { kind: 'rw', temp: true },      // 写临时 partial
        ],
        uniforms: () => encodeNUniform(nElements(shapes[0]), 0, 0),
        workgroups: (s) => [wgCount(s[0]), 1, 1],
        tempOutputElements: (s) => wgCount(s[0]),
      },
      { // ③ phase2:折叠 + epilogue 除以 n
        key: `reduce-p2-mean-${dtypes[0]}`,
        wgsl: reducePhase2Wgsl(dtypes[0], (a, x) => `${a} + ${x}`,
                               (v) => `${v} / uniforms.scalar`),
        bindings: [
          { kind: 'uniform' },
          { kind: 'read', temp: true },    // 读 phase1 的 partial
          { kind: 'rw', output: true },    // 写最终输出
        ],
        uniforms: () => encodeNUniform(wgCount(shapes[0]), shapes[0][0], 0),
        workgroups: () => [1, 1, 1],
      },
    ],
  }),
};

// ④ 注册到 Tensor.prototype
p.mean = reduceMethod(meanDef, meanAxisDef);
```

约定清单:

| 项 | 约定 |
| -- | ---- |
| `outShape` | 建图期调用,必须纯函数;形状错误在此抛 |
| `outDtype` | 省略 = 继承输入;`argmax/argmin` 返回 `'u32'` |
| `build` | **执行期**调用(此时已知全部输入 buffer);可以依赖具体 shape |
| `steps[].key` | 稳定字符串:相同 key 共享 pipeline(`dtype` 必须编进 key) |
| `uniforms` | 返回 `ArrayBuffer`,字节布局必须与 WGSL struct 完全一致 |
| 临时 buffer | 中间 step 用 `rw: temp` 写、下一 step 用 `read: temp` 读;scheduler 自动 acquire/release |
| 数值特殊值 | ±Infinity / NaN 一律走 uniform 槽(§8.2),**绝不内联进 WGSL** |
| 新 uniform 布局 | 同步新增 `encode*Uniform` 并在 `tests/unit/core.test.ts` 写字节级断言 |

---

## 6. BufferPool 与 PipelineCache 设计

### 6.1 BufferPool:幂次桶 + 严格分池

```text
请求 4 元素 f32 = 16 B → bucket = 16
请求 1000 元素 f32 = 4000 B → bucket = 4096
请求 UBO 16 B → bucket = max(256, 16) = 256(UBO 最小保证对齐)
```

- **storage 与 uniform 是两棵完全独立的桶树**(`freeStorage` / `freeUniform` 两个 Map)。混池曾是真实 bug(§8.5):UNIFORM-only 的 buffer 被当 storage 绑定 = validation error = 静默 no-op dispatch。
- 桶大小从 16 B 起步,`nextPow2` 向上取整。最坏内部碎片 50%,换来 O(1) 命中与稳定的复用率。
- `live`(借出数)/ `pooled`(空闲数)两个计数器用于测试断言「执行后无泄漏」。
- `releaseRaw(buffer, usage)` 面向 Kernel 等绕过 GpuDataBuffer 包装的场景。

### 6.2 PipelineCache:把静默失败变成可读报错

```ts
const module = device.createShaderModule({ code: wgsl });
const info = await module.getCompilationInfo();
for (const msg of info.messages) {
  if (msg.type === 'error') throw new Error(`WGSL: ${msg.lineNum}:${msg.linePos} ${msg.message}`);
}
```

WebGPU 的 validation 错误**不抛异常**:对象变 invalid、后续 dispatch 变 no-op。这是「GPU 输出全 0 却无任何报错」的直接根源。PipelineCache 在创建每个 ShaderModule 后显式读 `getCompilationInfo()`,把错误变成带行列号的一行日志。代价是管线创建异步化(首次 `await`),之后全部命中缓存。

缓存的 key 是 `hash(wgsl + entryPoint)`(`util/hash.ts` 的 FNV-1a 变体)——WGSL 是模板产物,同一算子在不同 shape 间 WGSL 完全相同(维度走 uniform),所以一个 session 里 phase1 的 WGSL 通常只编译一次。

---

## 7. 无 GPU 环境的 WebGPU:SwiftShader 配方

`tests/gpu/harness.ts` + `scripts/run-gpu-tests.sh` 的完整配方,已进 CI:

| 要素 | 值 | 为什么 |
| ---- | -- | ------ |
| 虚拟显示 | `xvfb-run` | headless chrome 的 SwiftShader/SwANGLE 在无 DISPLAY 时崩溃 |
| Vulkan ICD | `VK_ICD_FILENAMES=<chrome>/vk_swiftshader_icd.json` | Chrome 自带的 CPU 模拟 GPU |
| 浏览器 | 完整版 Chromium / google-chrome-stable | headless-shell 裁掉了 WebGPU |
| 开关 | `--enable-unsafe-webgpu --enable-features=Vulkan --no-sandbox` | 打开 WebGPU + 强制 Vulkan 后端 |
| 安全上下文 | https(自签)或 localhost | `navigator.gpu` 只在 secure context 暴露 |
| 执行方式 | 页面内注入 `dist/moxwebgpu.browser.js`,vitest 与页面 postMessage 通信 | Node 侧没有 WebGPU 实现 |

`MOXWEBGPU_VITEST_CONFIG` 环境变量可替换 vitest 配置(`vitest.gpu.config.ts` / `vitest.bench.config.ts`)。

---

## 8. 踩坑实录:六个真实 bug 及其修复

全部在本仓库开发过程中真实发生,每条都值得单独一篇博文。这里记录现象 → 根因 → 修复,供遇到同样问题的后来者检索。

### 8.1 `shared` 是 WGSL 保留字

- **现象**:所有带 workgroup 内存的 shader 创建失败,报 `expected identifier, got 'shared'` 类语法错误。
- **根因**:从 GLSL/ Metal 迁移过来的肌肉记忆。WGSL 里 workgroup 地址空间的正确写法是 `var<workgroup>`,而 `shared` 是保留字,作为标识符都不可用。
- **修复**:`var<workgroup> smem: array<f32, 64>;`(25 处统一改名为 `smem`)。

### 8.2 ±Infinity 不能出现在 WGSL const-expression

- **现象**:max/min 归约、softmax 全部输出错误;编译信息报 `value -inf cannot be represented as 'f32'`。
- **根因链**:
  1. 归约需要 identity(sum=0、max=−inf、min=+inf);
  2. 最初把 identity 作为字面量拼进 WGSL;
  3. WGSL 规范禁止 const-expression 产生 ±Infinity/NaN,而 `var acc = -Infinity;` 顶部的初始化器恰是 const-expression;
  4. 中途还踩了 `f32(bitcast<u32>(0xff800000u))` 的坑:`bitcast<T>` 的 T 是**目标**类型,`bitcast<u32>` 是把 f32 位型转成 u32;而 `f32(bitcast<u32>(…))` 在标量上下文是**数值转换**(`u32(4286578688)` → `4286578688.0`),不是位重解释。想真的重解释得写 `bitcast<f32>(0xff800000u)`——但即便写对了,放在初始化器里依然是 const-expression,照样被拒。
- **修复(设计固化)**:每个 uniform 块增加 `identity: f32` 槽位,运行时 `DataView.setFloat32(12, -Infinity, true)` 传入。codegen 层的 `fmtF32()` 遇到 NaN/±Infinity 直接抛错,把这条规矩变成硬约束。

### 8.3 uniform 里数组的 stride 跨实现不一致

- **现象**:slice/concat 输出全 0。没有编译错误、没有 validation error,纯静默。
- **根因**:排查时用探针 shader 把 uniform 逐字节写回输出,发现 shader 只读到了 offset 0 的内容。uniCopy 原来用 `array<u32,4>` 传形状数组——规范说数组元素 stride 是 4 字节,但 SwiftShader 前端按 `array<u32,4>` 的布局规则(适配 stride 要求)当成 16 字节 stride 处理,于是 `outShape[1]` 读到的是 offset 16 而不是 4,全部维度错乱。
- **修复(设计固化)**:copy 系 struct 改用 `vec4<u32>` 成员——向量成员的布局规范定义得铁板一块,任何实现都不会错。副作用是块从 208 B 缩到 64 B(去掉对齐 padding)。

### 8.4 argmin 拿到越界下标 32

- **现象**:`[3,1,2]` 的 `argmin()` 返回 32,数组总共 3 个元素。
- **根因**:phase1 的初始候选下标是 `start + l.x`,超过 n 的 lane 会用这个下标发起 OOB 读——WebGPU 对 OOB 读返回 0,而 0 恰好是 `[3,1,2]` 的最小值,越界下标就这样"合法"地赢了。
- **修复**:初始候选 clamp 到合法区间:`var acc: u32 = min(start + l.x, uniforms.n - 1u);`。这个 clamp 同时保证了比较总发生在有效内存上。

### 8.5 BufferPool 混池:静默 no-op 的完美风暴

- **现象**:小张量的部分算子偶发输出全 0,大张量正常;无任何报错。
- **根因**:最初 storage 与 uniform 共用一个 `Map<size, GPUBuffer[]>`。16 B 的 storage 请求从"16 B 桶"里拿到的,可能是某次归约用过的 **UNIFORM-only** buffer。拿去绑 `var<storage>` = validation error = dispatch 静默跳过。大张量正常是因为 256 B+ 的桶恰好没有被 uniform 用过。
- **修复(设计固化)**:`freeStorage` / `freeUniform` 两个 Map 严格分池;`releaseUniform()` 单独归还;`clear()` 双清。

### 8.6 UBO 泄漏:每 dispatch 丢一块

- **现象**:`pool.live` 随每次执行单调上升,`pooled` 恒为 0。
- **根因**:scheduler 为每个 step 的 uniform 新建 UBO,submit 后从不归还——池借出的 UBO 全部"悬空"。
- **修复**:scratchUbos 数组收集,`queue.submit` 后逐个 `releaseUniform`。

### 附:两条测试基建的血泪

- **`-0` 与 `+0`**:`expect([-0]).toEqual([0])` 在 vitest 里是 `Object.is` 语义,失败。WGSL 与 JS 的 `-x` 对 `0` 都翻转符号位(IEEE 754 正确行为)。测试里用 `.map(v => v + 0)` 归一化。
- **withGpu 闭包序列化**:测试回调被序列化后在浏览器页面里执行,**模块级变量全部丢失**(`scoped is not defined`)。harness 回调必须完全自包含,任何常量都要内联。

---

## 9. 已知的边界与取舍

| 取舍 | 现状 | 影响 |
| ---- | ---- | ---- |
| dtype 只有 32 位 | f32/i32/u32 | f16/bf16 需要扩展支持(路线图) |
| 按轴归约只支持最后一轴 + 2D 的 0 轴 | 0 轴借道 transpose | >2D 任意轴待做 |
| 两阶段归约 | phase2 是**单 workgroup**,但用 64 线程 strided 折叠任意多个 partial,**无硬性上限**;整条链的实际上限来自 phase1 的 workgroup 数(≤ maxComputeWorkgroupsPerDimension = 65535)→ 约 33M 元素以内单 GPU pass 可承受 | 更大规模需 2D 网格(在路线图) |
| 调度器 temp 回收为单槽 `prevTemp` | 适配当前全部算子(最多 2 个 step、1 个中间 temp),正确无泄漏 | **加算子须知**:若未来出现 ≥3 个 step 且含多个中间 temp 的算子,中间 temp 不会被回收,会泄漏——需扩展为多槽回收或在该算子内自行串联 |
| `Kernel.run()` 读回长度 | 优先取传入 `GpuDataBuffer.elements`(池化 buffer 按 2 的幂字节桶向上取整,直接读原始尺寸会多读 padding);传入裸 `GPUBuffer` 则回退到其原始尺寸 | 逃生舱读回要拿到精确长度,请传 `GpuDataBuffer` 而非 `.buffer` |
| matmul 未用 tensor core / 双缓冲 | 经典 16×16 tile | 正确性优先,性能版本在路线图 |
| 同队列时间线 UBO 复用 | 见 §4.2 说明 | 未来可加 fence,接口不变 |
| `getCompilationInfo` 异步化管线创建 | 首次 await | 之后全部命中缓存,稳态零开销 |

---

*本文档随代码更新。发现与代码不符之处,欢迎提 issue / PR。*
