# Changelog

本项目所有显著变更都记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

计划中(见 README 路线图):f16/bf16、tensor-core matmul、多 GPU pass 融合、moxsh 生态集成。

## [0.2.0] - 2026-10-02

### 新增

- **任意轴归约**:`sum(axis)` / `mean(axis)` / `max(axis)` / `min(axis)` 现支持
  **任意 rank、任意轴**(负轴从末尾数,`-1` 即最后一轴)。新通用内核
  `axisReduceWgsl`(每输出元素一线程,沿轴串行折叠,维度走 uniform 因此全形状共享
  同一管线);末轴仍走原来的 workgroup 树形快速路径。2D 的 `sum(0)` 从借道
  transpose 改为直连通用内核(单 pass,累加顺序不变)。
- 整型归约可用:`i32` / `u32` 的 `sum` / `max` / `min` 全部支持
  (全局 + 按轴);`argmax` / `argmin` 对整型本就可用。

### 修复

- **i32/u32 归约此前无法编译**(真 bug):累加器 `var acc = uniforms.identity`
  被推断为 f32,与整型输入 `combine` 产生 WGSL 类型错误。现在整型在 shader 内
  对 f32 identity 槽做**显式饱和转换**(`identityTyped`),max/min 的 identity 按
  dtype 取饱和边界(i32:±2147483647…8,u32:0/4294967295)。
- `mean` 对整型输入给出明确报错(提示先 `cast('f32')`),而不是难以理解的
  shader 编译失败。
- **调度器 temp 回收加固**:`runNode` 原用单槽 `prevTemp` 回收中间缓冲,若某算子
  出现 ≥3 个 step(多个中间 temp)会泄漏。现改为 `tempBufs` 数组,本节点分配的
  全部中间 temp 一律回收(当前算子行为不变,属预防性加固)。

### 测试

- GPU e2e +4:3D 任意轴(轴 0/1/2、负轴、mean)、i32 全局、u32 全局(含 0 元素)、
  i32 按轴;单测 +axisReduceOpDef outShape / normAxis 共 7 用例。

## [0.1.0] - 2026-10-02

首个公开版本。核心、算子、惰性计算图、测试基建与文档全部就绪;
**23 个 GPU 端到端用例在 SwiftShader(CPU 模拟 GPU)上全绿**。

### 新增

**核心(core/)**

- `mox.init()`:一行获取 `MoxContext`(device / queue / pool / pipelines / scheduler 全组装)。
- `BufferPool`:2 的幂字节桶内存池,**storage 与 uniform 严格分池**,
  `live` / `pooled` 统计,稳态 dispatch 几乎零 GPU buffer 分配。
- `PipelineCache`:WGSL 哈希 → compute pipeline 缓存;
  创建后显式 `getCompilationInfo()`,把 WebGPU 的静默失败变成带行列号的可读报错。
- `Kernel`:原始 WGSL 逃生舱,`dispatch()` 只下发 / `run()` 下发并读回。
- dtype 体系:`f32` / `i32` / `u32` + TypedArray 工厂。

**张量与算子(tensor/)**

- `Tensor`:嵌套数组 / TypedArray 构造,链式调用,惰性执行,`toArray` / `item` / `toBuffer` 读回。
- 逐元素:`add` `sub` `mul` `div` `pow` `clamp` + 标量反转 `rsub` `rdiv`,
  支持 1D / 2D 行广播 / 列广播,形状不合法直接抛错。
- 一元 14 个:`neg` `abs` `exp` `log` `sqrt` `sin` `cos` `tanh` `floor` `ceil`
  `relu` `sigmoid` `square` `sign`。
- 归约:全局 `sum` `mean` `max` `min` `argmax` `argmin`(两阶段树形,workgroup=64 / chunk=8);
  按最后一轴 `sum(-1)` 等;2D `sum(0)` 借道 transpose。
  max/min 的 ±Infinity identity 走 uniform 槽运行时传入(绕开 WGSL const-expression 限制)。
- `matmul`:16×16 workgroup 分块,workgroup 内存 tile 累加。
- 形状:`transpose`(2D)、`slice`(≤4D)、`concat`(任意轴,多 chunk 一次直写)、
  `reshape`(**零拷贝视图**,支持 `-1`)。
- `softmax`:数值稳定实现,单 workgroup 融合三遍(max → exp 求和 → 归一)。
- `cast`:f32 ↔ i32 ↔ u32,向零截断。
- 统一算子 API:`max()` 无参 = 归约、`max(-1)` = 按轴、`max(t)` = 元素级;`min` 同理,
  并提供 `maxReduce` / `minReduce` 无歧义别名。

**计算图(graph/)**

- 惰性 DAG:建图零 GPU 开销,`toArray()` / `item()` 时拓扑排序一次性下发;
- 中间 buffer 引用计数,消费者归零瞬间归还内存池;
- 已执行节点缓存结果,重复读回不重算;
- 单 `CommandEncoder` 多 compute pass,一次 `queue.submit`。

**工程与文档**

- 双构建:tsup 产出 ESM / CJS / IIFE(浏览器全局 `MoxWebGPU`)+ 类型声明,运行时零依赖。
- 测试:17 个单元测试(uniform 字节级校验 / 图 / OpDef)+
  **23 个 GPU e2e**(与 CPU 参考实现逐值比对)。
- SwiftShader 测试配方:xvfb + Chrome Vulkan ICD 自动化套壳(`pnpm test:gpu` 一条命令,
  无 GPU 机器 / CI 均可跑真 WebGPU)。
- 微基准:`pnpm bench`(中位数统计,含 add / matmul / 归约 / 链式)。
- 浏览器演示页:液态玻璃设计语言,四张卡片真跑 WebGPU(`pnpm demo`)。
- CI:GitHub Actions,类型检查 + 构建 + 单测 + SwiftShader GPU e2e 双 job。
- 文档:README(中文为主,含 API 参考 / 算子清单 / FAQ)、英文版、
  `docs/ARCHITECTURE.md`(uniform 字节表 / 调度算法 / 踩坑实录)、MIT LICENSE、FUNDING.yml。

### 修复

开发过程中被测试抓出来并固化为设计的缺陷(完整复盘见 docs/ARCHITECTURE.md §8):

- `shared` 是 WGSL 保留字 → workgroup 变量统一改名 `smem`。
- ±Infinity 作为 WGSL const-expression 字面量会报 shader-creation 错误
  → uniform 增加 `identity` 槽位运行时传入,`fmtF32()` 对特殊值直接抛错。
- uniform 地址空间数组 stride 跨实现不一致(SwiftShader 把 `array<u32,4>` 当 16B stride)
  → copy 系 struct 全部改 `vec4<u32>`(208 B → 64 B)。
- argmin 初始候选下标可越界(OOB 读回 0 被误判为最小值)
  → 候选 clamp 至 `min(start + l.x, n - 1u)`。
- BufferPool storage/uniform 混池导致 UNIFORM-only buffer 被当 storage 绑定
  (validation error = 静默 no-op)→ 严格分池 + `releaseUniform()`。
- 每次执行泄漏 scratch UBO → submit 后统一归还池。

[unreleased]: https://github.com/codecloud-dev/moxwebgpu/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/codecloud-dev/moxwebgpu/releases/tag/v0.2.0
[0.1.0]: https://github.com/codecloud-dev/moxwebgpu/releases/tag/v0.1.0
