<p align="center">  
  <img src="assets/logo.svg" width="128" alt="moxwebgpu 液态玻璃标志">  
</p>

<h1 align="center">moxwebgpu</h1>

<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/codecloud-dev/moxwebgpu/ci.yml?branch=main&label=CI&color=8a7bff" alt="CI">
  <img src="https://img.shields.io/github/stars/codecloud-dev/moxwebgpu?style=social" alt="GitHub Stars">
  <img src="https://img.shields.io/github/discussions/codecloud-dev/moxwebgpu?label=Discussions&color=ff7ac3" alt="社区讨论">
  <img src="https://img.shields.io/badge/version-1.0.1-8a7bff" alt="version">
  <img src="https://img.shields.io/badge/npm-publish%20pending-ffb000" alt="npm">
  <img src="https://img.shields.io/badge/license-MIT-37d5d3" alt="license">
  <img src="https://img.shields.io/badge/WebGPU-GPGPU-8a7bff?logo=webgpu&logoColor=white" alt="WebGPU">
  <img src="https://img.shields.io/badge/deps-0-37d5d3" alt="zero deps">
  <img src="https://img.shields.io/badge/tests-48%20passing-3DDC84" alt="tests">
  <img src="https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/Ops-30%2B-ff7ac3" alt="ops">
</p>


<p align="center">
  <b>中文</b> · <a href="README.en.md">English</a> · <a href="https://codecloud-dev.github.io/moxwebgpu/docs/">📖 文档站(中/EN 一键切换)</a> · <a href="https://codecloud-dev.github.io/moxwebgpu/demo/">🚀 在线演示</a>
</p>

> **同一份源码,浏览器运行、Node 构建与测试。** 没有独立显卡的机器上,用 Chrome 自带的 SwiftShader 软件渲染也能把 27 个真实 WebGPU 端到端测试全部跑绿——这是 moxwebgpu 与大多数「纸面 WebGPU 项目」最大的不同:**它的每一行 GPU 代码都被真实验证过**。

<details>

<summary><b>目录</b>(点击展开)</summary>

- [项目概览](#项目概览)
- [关于本项目](#关于本项目)
- [为什么需要 moxwebgpu](#为什么需要-moxwebgpu)
- [三层架构](#三层架构)
- [安装](#安装)
- [快速上手](#快速上手)
- [统一 API 参考](#统一-api-参考)
- [浏览器在线训练实战](#浏览器在线训练实战)
- [算子清单](#算子清单)
- [数据类型与内存布局](#数据类型与内存布局)
- [错误处理与调试](#错误处理与调试)
- [性能与基准](#性能与基准)
- [测试与质量保证](#测试与质量保证)
- [浏览器演示](#浏览器演示)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [路线图](#路线图)
- [FAQ](#faq)
- [参与进来](#参与进来)
- [支持我们](#支持我们)
- [许可证](#许可证)

</details>

---

## 📦 项目概览

**moxwebgpu** = **MoX** + **WebGPU**:一个面向浏览器的 **WebGPU 通用计算(GPGPU)框架**。它把 WebGPU 底层繁琐的适配器管理、管线构建、缓冲区生命周期、数据读回全部封装起来,对外暴露三层递进式 API:

| 你想要   | moxwebgpu 给你的                   | 一句话示例                                                  |
| ----- | ------------------------------- | ------------------------------------------------------ |
| 快速做计算 | **Tensor 层** —— 类型化张量 + 链式算子    | `gpu.tensor([1,2,3]).add(1).sum().item()`              |
| 省心做优化 | **Lazy 计算图** —— 整链一次下发、中间缓冲自动回收 | 六个算子串起来也只 dispatch 一次                                  |
| 完全控制  | **Kernel 层** —— 直接跑原始 WGSL      | `gpu.kernel(wgsl).run([bufA, bufB], { elements: 64 })` |
| 关键指标  | 数值                                               |
| ----- | ------------------------------------------------ |
| 运行时依赖 | **0**(纯 TypeScript)                              |
| 产物    | ESM + CJS + IIFE(浏览器全局 `MoxWebGPU`)+ 完整 `.d.ts`  |
| 数据类型  | `f32` / `i32` / `u32`                            |
| 内置算子  | 30+(逐元素 / 归约 / 矩阵 / 形状 / NN / 类型转换)              |
| 测试    | 21 单元测试 + **27 个真实 GPU 端到端测试** + 微基准             |
| 最低环境  | 支持 WebGPU 的浏览器(Chrome / Edge 113+);Node ≥ 18(构建) |

> **moxwebgpu 适合谁**:需要在浏览器里做矩阵运算、图像处理、信号处理、ML 前向推理、并行数值计算,又不想手写一屏 WebGPU 样板代码的你。  
> **moxwebgpu 不做什么**:不做 WebGL 回退(WebGPU 是底线)、暂不做训练侧自动微分(在路线图上)、不绑定任何 UI 框架。

---

## 💡 关于本项目

moxwebgpu 是一个独立的开源库:把浏览器里「本该简单」的 WebGPU 通用计算能力,认真重造一遍。

| 项目 | 状态 | 一句话定位 |
| :--: | :--: | --- |
| **moxwebgpu** | 已上线(本仓库) | **WebGPU 通用计算框架 —— 浏览器里的张量与计算图** |



---

## 🤔 为什么需要 moxwebgpu

WebGPU 的 compute pipeline 能力极强,但裸用它做一次向量加法,你要亲手闯过五关:

| #  | 你要亲手做的事                                        | 容易踩的坑                                       |
| :- | ---------------------------------------------- | ------------------------------------------- |
| 1  | `requestAdapter` → `requestDevice` → 队列管理      | 上下文样板代码一写一屏,拿错 adapter 直接崩                  |
| 2  | 写 WGSL、建 `ShaderModule`、拼 bind group layout    | uniform 对齐规则隐蔽;**布局错了不报错,dispatch 变 no-op** |
| 3  | 手动分配 / 复用 / 销毁 storage buffer                  | 忘销毁就泄漏;复用错就是脏数据;池子写不好性能反而降                  |
| 4  | staging buffer + `copyBufferToBuffer` + map 读回 | 读回流程繁琐,同步语义搞错就读到半截数据                        |
| 5  | 每个形状重新建 pipeline                               | 反复 dispatch 时编译开销白白流失                       |

其中第 2 关的「静默失败」最阴险:**WebGPU 的 validation 错误不会抛异常**,只会让对象悄悄变 invalid、dispatch 变成 no-op,你盯着满屏的 0 毫无头绪。moxwebgpu 在开发期就栽过这些坑(WGSL `shared` 保留字、uniform 数组 stride 两种后端不一致、const-expression 禁止 ±Infinity),**坑全部沉淀成了框架的内置防御**(见[错误处理与调试](#错误处理与调试))。

**moxwebgpu 把五关全部关进框架里,你只需要关心「算什么」。**

---

## 🏗️ 三层架构

```text
┌───────────────────────────────────────────────────────────────────┐
│  Tensor 层 —— 统一链式 API(你所写的)                              │
│                                                                   │
│  gpu.tensor([1,2,3]).add(1).relu().mul(10).sum().item()           │
└──────────────────────────────┬────────────────────────────────────┘
                               │  每个算子只「建节点」,不执行
┌──────────────────────────────▼────────────────────────────────────┐
│  Lazy 计算图 —— 拓扑排序 · 一次下发 · 引用计数回收                  │
│                                                                   │
│  • 全链构建成 DAG,读回时一次性 topo 排序逐节点执行                 │
│  • 中间 buffer 在消费者计数归零的瞬间归还内存池                     │
│  • 已执行的节点缓存结果,重复读回不重算                              │
└──────────────────────────────┬────────────────────────────────────┘
                               │  按 BindingSpec 解析绑定、UBO 走池
┌──────────────────────────────▼────────────────────────────────────┐
│  Core —— BufferPool · PipelineCache · Kernel(逃生舱)              │
│                                                                   │
│  • BufferPool:2 的幂字节桶,storage / uniform 严格分池             │
│  • PipelineCache:按 WGSL 哈希缓存 compute pipeline                 │
│  • getCompilationInfo() 显式捕获 shader 编译错误,拒绝静默 no-op    │
└───────────────────────────────────────────────────────────────────┘
```

### 🔹 一次 `.add(1).sum().item()` 的完整生命周期

1. `gpu.tensor([...])` 上传数据到池内 storage buffer(Tensor 持有,惰性)。
2. `.add(1)` → 在计算图上挂一个 `add-scalar` 节点,**未执行**。
3. `.sum()` → 再挂一个两阶段归约节点(两个 kernel step)。
4. `.item()` → 触发执行:
   - 从根节点做**拓扑排序**;
   - 每个节点:从池里拿输出 buffer → 编码 uniform → 组 bind group → **一次** `dispatchWorkgroups`;
   - 中间 buffer 的消费者计数减到 0 的那一刻,立刻归还内存池;
   - 归约结果经 staging buffer 读回 CPU,返回标量。
5. 整条链共 **3 次 kernel dispatch**(add、归约 phase1、phase2),中间大 buffer 0 泄漏。

更深的实现细节(WGSL 模板、uniform 字节表、调度算法)见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**。

---

## 📦 安装

> 🟡 **待发布到 npm（配置已就绪）**：`package.json` 已就位（`version: 1.0.1`，`prepublishOnly` 会自动构建），OIDC 免 token 发布流程已配好（见 `.github/workflows/publish.yml`）。等你绑定 npm 账号后执行一次 `npm publish` 即上线；此前可用下方「方式三：从源码构建」或在线 Demo。

### 🌐 方式一:浏览器 `<script>`(零构建)

```html
<script src="https://unpkg.com/moxwebgpu/dist/moxwebgpu.browser.js"></script>
<script>
  const gpu = await MoxWebGPU.mox.init();
  console.log(await gpu.tensor([1, 2, 3]).sum().item()); // 6
</script>
```

IIFE 产物暴露全局 `MoxWebGPU`,内含 `mox`、`MoxContext`、`Tensor`、`Kernel` 等全部导出。

### 🔹 方式二:npm(Node / 打包器)

```bash
npm install moxwebgpu
# pnpm add moxwebgpu  /  yarn add moxwebgpu

```

```ts
import { mox } from 'moxwebgpu';          // ESM
// const { mox } = require('moxwebgpu');  // CJS
```

### 🔹 方式三:从源码构建

```bash
git clone https://github.com/codecloud-dev/moxwebgpu.git
cd moxwebgpu
pnpm install          # 安装依赖(TypeScript / tsup / vitest / playwright-core)
pnpm build            # 产物输出到 dist/
```

---

## 🚀 快速上手

以下示例默认 `const gpu = await mox.init();` 已执行。

### 🔹 1. 第一个张量

```ts
const a = gpu.tensor([1, 2, 3, 4]);                       // 一维,自动推断 shape=[4]
const m = gpu.tensor([[1, 2, 3], [4, 5, 6]]);             // 二维,shape=[2,3]
const t = gpu.tensor(flatData, { shape: [128, 128] });    // TypedArray + 显式形状
```

### 🔹 2. 链式调用与惰性求值

```ts
// 写法像 NumPy,执行像 CUDA 图
const r = await gpu.tensor([1, 2, 3, 4])
  .add(1)      // [2,3,4,5]
  .relu()
  .mul(10)     // [20,30,40,50]
  .sum()       // 归约
  .item();     // 140 —— 读回时整条链才执行,且只执行一次
```

### 🔹 3. 广播

```ts
const A = gpu.tensor([[1, 2, 3], [10, 20, 30]]);   // [2,3]

await A.add(gpu.tensor([1, 2, 3])).toArray();
// 行广播 → [2,4,6, 11,22,33]

await A.mul(gpu.tensor([2, 3])).toArray();
// 列广播 → [2,4,6, 30,60,90]

await A.add(gpu.tensor([1, 2])).toArray();
// ❌ 抛错:形状不合法绝不静默广播
```

### 🔹 4. 矩阵乘法

```ts
const a = gpu.tensor([[1, 2], [3, 4]]);
const b = gpu.tensor([[5, 6], [7, 8]]);
await a.matmul(b).toArray();   // [19, 22, 43, 50]
// 16×16 workgroup 分块;[512,512] 级别同样一条命令
```

### 🔹 5. 归约与 softmax

```ts
const x = gpu.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);

await x.sum().item();      // 36    全局归约 → [1]
await x.sum(-1).toArray(); // [10, 26]  按最后一轴 → [2]
await x.sum(0).toArray();  // [6, 8, 10, 12]  按第 0 轴 → [4]
const t3 = gpu.tensor(Array.from({ length: 24 }, (_, i) => i), { shape: [2, 3, 4] });
await t3.sum(1).toArray(); // [12,15,18,21,48,51,54,57] —— 3D 中间轴也行,负轴从末尾数
await x.max().toArray();   // [8]
await x.argmax().item();   // 7(首见优先)

await x.softmax().toArray();
// [0.032, 0.087, 0.237, 0.644, ...] —— 减最大值的数值稳定实现,逐行和恒为 1
```

### 🧮 6. 形状操作

```ts
const v = gpu.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
const m = gpu.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);  // 3×3

await v.slice(2, 4).toArray();            // [2,3,4,5](start=2, size=4)
await m.slice([1, 1], [2, 2]).toArray();  // 2D 块切片 → [[5,6],[8,9]]

await gpu.tensor([[1, 2], [3, 4]]).concat(gpu.tensor([[5, 6]]), 0).toArray();
// [1,2,3,4,5,6] —— axis 0 拼接;多 chunk 一次直写输出

const w = v.reshape([2, 5]);   // 零拷贝视图,不产生 GPU 计算
```

### 🔤 7. 类型转换

```ts
await gpu.tensor([1.7, 2.2, -3.9]).cast('i32').toArray();  // [1, 2, -3](向零截断)
gpu.tensor([1, 2]).toFloat().dtype;                        // 'f32'
```

### ⚙️ 8. 原始 Kernel(逃生舱)

```ts
const k = gpu.kernel(`
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read_write> b: array<f32>;
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) gid: vec3u) {
    b[gid.x] = a[gid.x] * 2.0 + 1.0;
  }`);

const a = gpu.tensor([1, 2, 3, 4, 5]);
const out = gpu.pool.acquire(5, 'f32');           // GpuDataBuffer,elements = 5
// 传入 GpuDataBuffer(而非 .buffer):readback 长度按 buf.elements 取,返回精确的 5 个元素
const res = await k.run([a.data, out], { elements: 5 });
// Float32Array(5) [3, 5, 7, 9, 11]
```

`k.run()` = dispatch + 读回一步到位;`k.dispatch(buffers, [wx, wy, wz])` 只下发不读回。WGSL 照常走管线缓存。

> 读回长度说明:`run()` 优先使用传入的 `GpuDataBuffer.elements` 作为读回元素数(池化 buffer 会按 2 的幂字节桶向上取整,直接读原始尺寸会多读 padding)。若传入裸 `GPUBuffer`,则回退到其原始尺寸。

### 📦 9. 资源管理

```ts
gpu.destroy();   // 归还并销毁全部池化 buffer + device(页面卸载前调用)
t.destroy();      // 单个张量释放(通常不需要,池会自动回收中间结果)
```

---

### 🧠 10. 自动微分（反向模式 / 训练）

moxwebgpu 1.0 起内置**反向模式自动微分**,直接叠加在现有惰性计算图上——`OpDef` 只是纯元数据 + codegen,每个算子附带一个 `backward`,反向图天然复用同一套算子,只有当你读取梯度时才上 GPU。

```ts
import { mox } from 'moxwebgpu';

const gpu = await mox.init();
const x = gpu.tensor([1, 2, 3, 4]).withGrad();   // 标记可微叶子
const loss = x.mul(x).sum();                       // L = Σ x²

loss.backward();                                   // 反向传播
console.log(await x.grad.toArray());               // [2, 4, 6, 8] = dL/dx
```

三个要点:

- **`.withGrad()`**:标记该叶子张量需要梯度(其余叶子默认不计)。
- **`.backward()`**:从标量 loss 出发,逆拓扑序传播,把梯度填进每个 `requiresGrad` 叶子的 **`.grad`**。整个过程只建惰性图,**不碰 GPU**,直到你 `await x.grad.toArray()`。
- **广播正确**:二元算子的梯度会用 `sumTo` 自动归约回较小操作数的形状;`sum/mean/softmax` 的梯度用 `expand` 复制回去。

```ts
// 一个最小线性层的前向 + 反向
const W = gpu.tensor([[0.1, 0.2], [0.3, 0.4]]).withGrad();
const inp = gpu.tensor([1, 1]);
const pred = W.matmul(inp);          // [2]
const loss = pred.mul(pred).sum();   // Σ pred²
loss.backward();
console.log(await W.grad.toArray());  // dL/dW,形状与 W 一致
```

**已支持反向的算子**:加/减/乘/除(含标量、含广播)、neg/abs/exp/log/sqrt/square/relu/sigmoid/tanh、matmul、sum/mean(全局与任意轴)、softmax(沿末轴,数值稳定)。

> 路线图中 layernorm / embedding / conv1d、pass 融合、Web Worker 运行属于 **1.0.x 后续规划**,本版未做——给后续更新留空间。

---

### 🚀 浏览器在线训练实战（无需安装，浏览器里当场训一个模型）

> **这是 moxwebgpu 最被低估的能力**：它不止能做单次 GPGPU 计算，还能在**浏览器里跑完整的自动微分 + 梯度下降训练循环**——数据不用出浏览器、不依赖任何后端，笔记本 / 手机 / 树莓派只要有 WebGPU 就能训。配合前面说的「前端直接 `<script>` 引入 SDK」，任意网站都能让访客的 GPU 当场帮你训练一个小模型。

下面用核心逻辑跑通一个 **线性回归（最小二乘 / MSE）**，梯度真的在 GPU 上算：

```ts
import { mox } from 'moxwebgpu';
const gpu = await mox.init();

// 造一点训练数据：y ≈ 2x + 1
const X = gpu.tensor([[0], [1], [2], [3]]);
const Y = gpu.tensor([[1], [3], [5], [7]]);

const lr = 0.05;
let W = gpu.tensor([[0]]).withGrad();   // 可微参数
let b = gpu.tensor([[0]]).withGrad();

for (let step = 0; step < 200; step++) {
  const pred = X.matmul(W).add(b);                       // 前向：ŷ = X·W + b
  const loss = pred.sub(Y).mul(pred.sub(Y)).mean();      // MSE
  loss.backward();                                       // 反向：梯度填进 W.grad / b.grad（惰性，未上 GPU）

  const wn = await W.sub(W.grad.mul(lr)).toArray();      // 手写 SGD 一步
  const bn = await b.sub(b.grad.mul(lr)).toArray();
  W = gpu.tensor(wn).withGrad();                         // 重建张量并重新标记可微
  b = gpu.tensor(bn).withGrad();
}
console.log('训完的 W =', await W.toArray(), 'b =', await b.toArray());  // ≈ [[2]], [[1]]
```

要点：

- **前向 / 反向全在惰性计算图上**：`backward()` 只建图、填 `.grad`，直到你 `await ...toArray()` 才真正下发 GPU——省得每一步都来回拷数据。
- **目前没有内置优化器**：训练需手写 `W = W - lr * W.grad`（如上）。`SGD / Adam` 已在路线图上。
- **想看真·在线演示？** 打开 [`examples/browser`](examples/browser/index.html) 或文档站的「在线演示」，你的 GPU 会当场把模型训给你看；也欢迎把它塞进你自己的网页（见 [浏览器演示](#浏览器演示)）。

> 📌 训练刚需「可微」：`withGrad()` 标记的叶子才会累积 `.grad`，其余张量默认不计梯度，省显存。

---

## 📚 统一 API 参考

### `mox.init(options?)` → `Promise<MoxContext>`

| 选项                      | 类型                                  | 说明                            |
| ----------------------- | ----------------------------------- | ----------------------------- |
| `adapter`               | `GPUAdapter`                        | 自带 adapter(跳过 requestAdapter) |
| `powerPreference`       | `'low-power' \| 'high-performance'` | 电耗偏好                          |
| `requestAdapterOptions` | `GPURequestAdapterOptions`          | 其余透传                          |

### 🔹 MoxContext

| 成员                    | 说明                                                             |
| --------------------- | -------------------------------------------------------------- |
| `tensor(data, opts?)` | 创建张量;`data` 支持嵌套数组 / TypedArray;`opts.shape` 可选                |
| `kernel(code, opts?)` | 包装原始 WGSL(`workgroupSize` / `entryPoint` / `output` / `dtype`) |
| `readback(buf)`       | GPU buffer → CPU TypedArray                                    |
| `info()`              | adapter 概要 `{ vendor, architecture, device, description }`     |
| `pool`                | BufferPool(高级用法)                                               |
| `pipelines`           | PipelineCache(高级用法)                                            |
| `scheduler`           | 计算图调度器(高级用法)                                                   |
| `destroy()`           | 释放一切                                                           |

### 🔹 Tensor

| 类别   | 成员                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------- |
| 属性   | `shape: number[]`、`ndim`、`dtype`、`size`(元素数)                                                      |
| 变换   | `reshape(...dims \| number[])`(**零拷贝**,支持 `-1`)、`transpose()`(2D)                                 |
| 二元   | `add` `sub` `mul` `div` `pow`(Tensor 或标量)                                                         |
| 标量反转 | `rsub(x)` ≙ `x - t`、`rdiv(x)` ≙ `x / t`                                                           |
| 一元   | `neg` `abs` `exp` `log` `sqrt` `sin` `cos` `tanh` `floor` `ceil` `relu` `sigmoid` `square` `sign` |
| 归约   | `sum` `mean` `max` `min`(无参=全局;**任意轴**=按轴,负轴从末尾数;`max(t)`/`min(t)`=元素级)、`argmax` `argmin`(→ `u32`) |
| 范围   | `clamp(lo, hi)`、`slice(start, size)`(≤4D)、`concat(other, axis?)`                                  |
| NN   | `softmax()`(数值稳定,逐最后一轴)                                                                           |
| 类型   | `cast('f32' \| 'i32' \| 'u32')`、`toFloat()`                                                       |
| 读回   | `await toArray(): TypedArray`、`await toBuffer(): GpuDataBuffer`、`await item(): number`            |
| 释放   | `destroy()`                                                                                       |

> **惰性语义**:`add` / `matmul` / `softmax` 等只是往计算图添节点;`toArray()` / `item()` / `toBuffer()` 才触发执行。已执行节点会缓存 GPU buffer,重复读回**不会重算**。

### ⚙️ Kernel

| 方法                                             | 说明                                     |
| ---------------------------------------------- | -------------------------------------- |
| `run(buffers, { elements } \| { workgroups })` | dispatch + 读回最后一个 binding 为 TypedArray |
| `dispatch(buffers, [wx, wy, wz])`              | 只 dispatch 不读回                         |

约定:WGSL 里 `@group(0) @binding(i)` 的声明顺序 = `buffers[i]` 的传入顺序。

### 🔹 BufferPool(高级)

| 方法                                     | 说明                                  |
| -------------------------------------- | ----------------------------------- |
| `acquire(elements, dtype)`             | 从 2 的幂字节桶取 storage buffer           |
| `acquireUniform(bytes)`                | 取 uniform buffer(独立桶族,与 storage 隔离) |
| `release(buf)` / `releaseUniform(ubo)` | 归还                                  |
| `live` / `pooled` / `clear()`          | 统计与清空                               |

---

## 🧮 算子清单

| 家族   | 算子                                                                                                | GPU 实现                                 |
| ---- | ------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 逐元素  | `add` `sub` `mul` `div` `pow` `min` `max` `clamp`                                                 | 1D / 2D(行/列广播)uniform 参数化              |
| 反转标量 | `rsub` `rdiv`                                                                                     | 同上                                     |
| 一元   | `neg` `abs` `exp` `log` `sqrt` `sin` `cos` `tanh` `floor` `ceil` `relu` `sigmoid` `square` `sign` | 单 pass 逐元素                             |
| 全局归约 | `sum` `mean` `max` `min` `argmax` `argmin`                                                        | 两阶段树形归约(workgroup=64,chunk=8)          |
| 按轴归约 | `sum(-1)` `mean(-1)` `max(-1)` `min(-1)`、`sum(0)`(2D,内部先 transpose)                               | 单 workgroup 负责一行的树形归约                  |
| 矩阵   | `matmul`                                                                                          | 16×16 tile,workgroup 内存分块累加            |
| 形状   | `transpose`(2D)、`slice`(≤4D)、`concat`(任意轴)、`reshape`(零拷贝)                                         | 通用 ND gather/copy shader(vec4 uniform) |
| NN   | `softmax`                                                                                         | 单 workgroup 融合三遍:max → exp 求和 → 归一     |
| 类型   | `cast`(f32 ↔ i32 ↔ u32)                                                                           | 单 pass 转换(向零截断)                        |

---

## 🔢 数据类型与内存布局

### 🔢 dtype

| dtype | WGSL  | 字节 | 读回 TypedArray  |
| ----- | ----- | -- | -------------- |
| `f32` | `f32` | 4  | `Float32Array` |
| `i32` | `i32` | 4  | `Int32Array`   |
| `u32` | `u32` | 4  | `Uint32Array`  |

### 🔹 uniform 块(全部 16 字节对齐)

| kernel 家族              | 布局                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------- |
| 1D 逐元素 / 全局归约          | `OpUniforms { n: u32, _pad: u32, scalar: f32, identity: f32 }`(16 B)                    |
| 2D 广播 / 按轴归约 / softmax | `OpUniforms { rows: u32, cols: u32, scalar: f32, identity: f32 }`(16 B)                 |
| copy(slice/concat)     | `outShape/inStrides/outStrides: vec4<u32>` + `inOffset/outOffset/total/rank: u32`(64 B) |
| matmul                 | `Dims { m, n, k, _pad }`(16 B)                                                          |

两条**血泪教训**已固化为设计:

1. **±Infinity 不能出现在 WGSL const-expression 里**(shader-creation error)。max/min 归约的 −inf/+inf identity 通过 uniform 的 `identity` 槽位在**运行时**传入。
2. **uniform 地址空间的数组 stride 在不同后端实现不一致**,而 `vec4<u32>` 成员布局处处一致——copy 系 shader 因此全部使用 vec4 维度(还更省:208 B → 64 B)。

---

## 🐛 错误处理与调试

### 🔹 WebGPU 的静默失败与 moxwebgpu 的防御

WebGPU validation 错误**不抛异常**:对象变 invalid、dispatch 变 no-op。moxwebgpu 的 `PipelineCache` 在创建 `ShaderModule` 后调用 `getCompilationInfo()` **显式检查编译错误并打印**,把「全 0 之谜」变成一行可读报错。

### 🐛 常见错误速查

| 错误消息(节选)                                       | 原因与解法                                            |
| ---------------------------------------------- | ------------------------------------------------ |
| `moxwebgpu: WebGPU is not available here...`   | 浏览器不支持 / 非 https 安全上下文;换 Chrome 113+ 或 localhost |
| `moxwebgpu: no suitable GPU adapter`           | 适配器被屏蔽;更新显卡驱动 / 浏览器                              |
| `moxwebgpu: broadcast failed ...`              | 二元运算形状既不相同也不可广播                                  |
| `moxwebgpu: slice range [...] out of bounds`   | 切片越界                                             |
| `moxwebgpu: axis ... reduce is not supported`  | 该算子不支持按轴归约(如 `argmax`);全局归约请不传参             |
| `moxwebgpu: mean requires f32 input`           | `mean` 只支持 f32;整型请先 `cast('f32')`                      |
| `[page:warning] Error while parsing WGSL: ...` | 测试 harness 转发的 shader 编译错误——按行列号修 WGSL           |

调试技巧:`gpu.kernel()` 的自定义 WGSL 同样走缓存与编译检查,报错会带行号列号打印到控制台。

---

## ⚡ 性能与基准

`pnpm bench` 在 SwiftShader(纯 CPU 软渲染,与 CI 同环境)上的中位数,**含 dispatch + 读回全链路**:

| 负载                         | 耗时      | 吞吐           |
| -------------------------- | ------- | ------------ |
| add 1M f32(标量)             | ~数十 ms  | —            |
| matmul [512×512]·[512×512] | ~890 ms | ~0.3 GFLOP/s |
| sum 1M f32(两阶段树形)          | ~69 ms  | —            |
| add→relu→mul→sum 1M(4 算子链) | ~285 ms | —            |

> SwiftShader 是**纯 CPU 模拟 GPU**,数字仅用于跨版本回归对比。同一份代码在真实硬件(集显/独显)上通常快 **10–100×**;matmul 受益于 16×16 分块,大矩阵吞吐会显著上升。

**内置的性能设计**:

- 惰性计算图:多算子链不产生中间 CPU 往返,一次 `item()` 一次读回;
- 中间 buffer 引用计数回收 + 幂次桶复用:稳态 dispatch 几乎零分配;
- pipeline 按 WGSL 哈希缓存:循环里反复执行同一计算零重建开销。

---

## 🧪 测试与质量保证

### 🧪 测试矩阵

| 套件                 | 数量        | 覆盖                                                                                                             |
| ------------------ | --------- | -------------------------------------------------------------------------------------------------------------- |
| `tests/unit/`      | 21 用例     | uniform 编码字节级校验、WGSL 生成、拓扑排序、OpDef 形状推导                                                                        |
| `tests/gpu/`       | **27 用例** | 逐元素 / 广播 / 一元链 / 惰性链 / matmul 3 种尺寸 / 全部归约 / argmax / softmax / slice / concat / reshape 视图 / 深流水线 / 低层 Kernel |
| `tests/gpu/bench/` | 微基准       | elementwise / matmul / 归约 / 链式(中位数统计)                                                                          |

GPU 用例全部与 **CPU 参考实现**逐值比对(matmul / softmax / argmax / argmin 均有 CPU 版),不是「不崩就算过」。

### 🔧 无 GPU 机器怎么跑真 WebGPU?(SwiftShader 配方)

这是本项目沉淀的独门配方,`pnpm test:gpu` 一条命令自动完成:

```bash
# 1. xvfb 提供虚拟显示(headless chrome 的 SwANGLE 在无 DISPLAY 时会挂)
# 2. Chrome 自带的 SwiftShader Vulkan ICD 作为 GPU 后端
export VK_ICD_FILENAMES=/opt/google/chrome/vk_swiftshader_icd.json

# 3. 完整版 Chromium(headless-shell 裁剪掉了 WebGPU)+ 关键开关
chromium \
  --enable-unsafe-webgpu \
  --enable-features=Vulkan \
  --no-sandbox

# 4. 必须加载 https 页面取得 secure context,才有 navigator.gpu
```

`tests/gpu/harness.ts` 把上面四步全部自动化:探测 Chrome 与 ICD → 拉起浏览器 → 打开 https 页面 → 注入 `dist/moxwebgpu.browser.js` → 每个用例在页面里执行并与 CPU 参考值比对。**GitHub Actions 用同一配方跑 CI**(见 `.github/workflows/ci.yml`)。

### 💻 本地命令

| 命令                       | 作用                              |
| ------------------------ | ------------------------------- |
| `pnpm build`             | tsup 构建 ESM / CJS / IIFE + d.ts |
| `pnpm test`              | 21 单元测试                         |
| `pnpm test:gpu`          | 27 GPU 端到端(自动 SwiftShader/xvfb) |
| `pnpm test:all`          | 两者都跑                            |
| `pnpm bench`             | 微基准                             |
| `pnpm demo`              | 构建并起本地演示页(localhost:5173)       |
| `pnpm exec tsc --noEmit` | 类型检查                            |

---

## 🌐 浏览器演示

🌐 **在线直接体验(无需安装)**:[https://codecloud-dev.github.io/moxwebgpu/demo/](https://codecloud-dev.github.io/moxwebgpu/demo/) —— 用你自己的 GPU 当场算给你看。

[`examples/browser/index.html`](examples/browser/index.html) 是一个自包含的液态玻璃演示页:近黑底、青→靛强调色、顶部高光反射、折射边、指针跟随光斑、缓慢漂移的环境光。

四张玻璃卡片,各自真刀真枪跑在 WebGPU 上:

| 卡片      | 内容                                |
| ------- | --------------------------------- |
| 张量链式调用  | `add → relu → mul → sum`,展示惰性一次下发 |
| 矩阵乘法    | [128×128]·[128×128],显示耗时          |
| Softmax | 逐行 softmax + 行和校验 = 1             |
| 归约      | 1M 元素 sum / max 与预期值对照            |

```bash
pnpm demo        # 构建 + 起服务,浏览器打开 http://localhost:5173
```

> 演示页需要 WebGPU 支持:Chrome/Edge 113+,且通过 `localhost` 或 https 访问。

---

## 📦 项目结构

```text
moxwebgpu/
├── src/
│   ├── index.ts                 # 统一导出 + 自动安装张量算子 + 版本号
│   ├── core/
│   │   ├── context.ts           # MoxContext:mox.init() 入口,组装以下全部组件
│   │   ├── dtype.ts             # f32/i32/u32 类型表、TypedArray 工厂
│   │   ├── buffer.ts            # GpuDataBuffer + BufferPool(幂次桶,storage/uniform 分池)
│   │   └── kernel.ts            # Kernel:原始 WGSL 逃生舱(dispatch/run/readback)
│   ├── graph/
│   │   ├── lazy.ts              # LazyNode / BindingSpec / OpDef / topoSort
│   │   ├── pipelineCache.ts     # WGSL 哈希 → pipeline 缓存 + getCompilationInfo 防御
│   │   └── scheduler.ts         # materialize/runNode:拓扑执行、绑定解析、引用计数回收
│   └── tensor/
│       ├── tensor.ts            # Tensor 类:fromData / apply / reshape / 读回
│       ├── codegen.ts           # WGSL 模板库 + uniform 编码器(字节级布局)
│       └── ops/
│           ├── index.ts         # installTensorOps:全部链式方法注册到 Tensor.prototype
│           ├── elementwise.ts   # 逐元素 / 广播 / 标量 / 一元算子定义
│           ├── reduce.ts        # 全局与按轴归约(identity 走 uniform)
│           ├── matmul.ts        # 16×16 分块矩阵乘
│           ├── shape.ts         # transpose / slice / concat
│           └── nn.ts            # softmax
├── tests/
│   ├── unit/core.test.ts        # 21 个单元测试(编码 / 图 / OpDef)
│   └── gpu/
│       ├── harness.ts           # Chrome + SwiftShader + xvfb 自动化套壳
│       ├── ref.ts               # CPU 参考实现(matmul/softmax/argmax/argmin)
│       ├── basics.test.ts       # 逐元素与低层 Kernel(8)
│       ├── matmul.test.ts       # 矩阵乘(3)
│       ├── reduce.test.ts       # 归约/softmax/形状/深流水线(16)
│       └── bench/bench.test.ts  # 微基准(独立 config,不进默认套件)
├── scripts/
│   └── run-gpu-tests.sh         # VK_ICD 探测 + xvfb-run 包装(支持配置覆盖)
├── examples/browser/
│   └── index.html               # 液态玻璃演示页(自包含)
├── docs/
│   └── ARCHITECTURE.md          # 架构深潜:uniform 字节表、调度算法、踩坑实录
├── assets/
│   └── logo.svg                 # 液态玻璃 logo
├── .github/workflows/ci.yml     # CI:类型检查 + 构建 + 单测 + SwiftShader GPU e2e
├── package.json / tsconfig.json / tsup.config.ts
├── vitest.config.ts / vitest.gpu.config.ts / vitest.bench.config.ts
├── CHANGELOG.md / LICENSE(MIT) / FUNDING.yml / README.md / README.en.md
```

---

## 🛠️ 开发指南

### 🔹 环境要求

- Node ≥ 18、pnpm ≥ 9
- 本地跑 GPU 测试需要 Chrome/Chromium(自动探测,也可 `MOXWEBGPU_CHROME=/path/to/chrome` 指定);Linux 无显示时自动包 `xvfb-run`
- 类型检查:`pnpm exec tsc --noEmit`

### 🔧 如何添加一个新算子(五步)

以 `rsqrt`(平方根倒数)为例:

1. **codegen.ts**:若现有模板不够,加一个 WGSL 模板(记住:binding 0 是 uniform;别用 `shared` 当变量名;±inf 别写进 const-expression);
2. **ops/elementwise.ts**:一行 unary 定义 —— `export const rsqrtDef = unaryOpDef('rsqrt', (a) => \`inverseSqrt(${a})\`);\`
3. **ops/index.ts**:`p.rsqrt = unaryMethod(rsqrtDef);` 注册到原型;
4. **测试**:单元测试(编码/形状)+ GPU 测试(与 CPU 值比对);
5. `pnpm test && pnpm test:gpu` 全绿,提 PR。

`OpDef` 的高层约定(形状推导、dtype 传递、多 step 输出)见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 🗺️ 路线图

- [x] **v0.1** —— 三层架构、30+ 算子、27 个真实 GPU e2e、液态玻璃演示
- [x] **v0.2** —— 任意轴归约(任意 rank/负轴)、整型归约修复、调度器 temp 回收加固
- [x] **v1.0** —— 反向模式自动微分(叠加在现有 lazy graph 上)、matmul/softmax/sum/mean 反向、OIDC 免 token 发布到 npm
- [ ] 更多 NN 算子:layernorm / embedding / conv1d(规划中,1.0.x)
- [ ] 相邻逐元素算子的 pass 融合(进一步压 dispatch 次数,规划中)
- [ ] Web Worker / OffscreenCanvas 内运行(规划中)
- [x] npm 首次发布
- [x] 文档站上线

> 有想要的功能或方向,欢迎提 Issue 一起讨论。

---

## ❓ FAQ

**Q:浏览器控制台报 `navigator.gpu is undefined`?**  
A:WebGPU 需要 Chrome/Edge 113+,且页面处于安全上下文(https 或 localhost)。`file://` 直接打开不行,用 `pnpm demo` 起本地服务。

**Q:计算结果是全 0 / 乱码?**  
A:99% 是 WGSL 编译失败(静默 no-op)。moxwebgpu 已把编译错误打印到控制台(带行列号),按提示修 shader 即可。直接用 moxwebgpu 内置算子则不会遇到。

**Q:和 TensorFlow.js / transformers.js 什么关系?**  
A:它们是「模型中心」:面向推理预置模型。moxwebgpu 是「算子中心」:给你 NumPy 式的原始计算能力 + 逃生舱,恰好可以作为它们没有的那层「通用 GPGPU 地基」。

**Q:支持训练(反向传播)吗?**  
A:支持。1.0.0 起内置反向模式自动微分:对叶子张量调 `.withGrad()`,正向建图后调 `.backward()`,梯度即填到 `.grad`(惰性,需要时才上 GPU)。详见[自动微分](#10-自动微分反向模式--训练)。

**Q:为什么我的机器跑 `pnpm test:gpu` 也能过?我没有 N 卡。**  
A:因为 SwiftShader——Chrome 自带的纯软件 Vulkan 实现。moxwebgpu 的 GPU 测试配方**不要求真显卡**,CI 上也一样。

**Q:i32/u32 乘法会溢出吗?**  
A:遵循 WGSL 语义(按位回绕)。归约 identity、编码器、读回视图都已按 dtype 处理。

---

## 🤝 参与进来

发现 bug 或想要新算子,欢迎提 [Issue](https://github.com/codecloud-dev/moxwebgpu/issues);想贡献代码直接提 Pull Request——WGSL 算子照着[五步指南](#如何添加一个新算子五步)加一个 `OpDef` 即可,测试会告诉你对不对。

---

## 💖 支持我们

moxwebgpu 是一个独立开发的免费开源项目,会持续维护和更新。**如果它帮你省下了写 WebGPU 样板代码的时间,欢迎给一个 ⭐ Star** —— 对一个独立小项目来说,这是最大的鼓励,也是让更多需要它的人能找到它的方式。

如果它对你的工作有实际帮助,也可以通过下面的通道支持开发:

- 爱发电:https://afdian.com/a/cloudharbor

在此之前,**Star、转发、把它用起来并告诉我用在哪里**,就是最好的支持。

也欢迎任何形式的共建:报 bug、提建议、交代码、写文档、做翻译。开发者主导设计 + AI 协作实现,每个方向都由人拍板。

---

## 📜 许可证

[MIT](LICENSE) © Codecloud —— 可自由商用、修改、分发,保留版权声明即可。

## 🤖 AI 辅助声明

本项目(含全部代码、文档与演示页)由开发者 **Codecloud** 主导设计,**AI 辅助生成代码**:架构决策、需求定义与验收由人完成,代码实现与文档撰写由 AI 协作完成并经人工审核修订。
