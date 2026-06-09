# P1 知识分享 3: DCU vs NVIDIA GPU + HIP 编程模型

**录制人**: 队员 A
**目标听众**: 4 人队
**时长**: 30 min
**前置阅读**: spec §5.4 (FP8 支持矩阵)
**录制日期**: P1 第 1 周 (分享 1+2 之后)

> **录制前必做**:
> 1. 确认 P0 Task 0.1 验证结果 (gfx90a 还是 gfx942) — 在 §5:00 / §12:00 段必须念出具体硬件
> 2. 跑一次 `rocminfo | head -30`, 屏幕录制时贴终端输出
> 3. 把 spec §5.4 的 FP8 支持矩阵截图贴到 PPT 第 5 页

---

## 0:00-5:00 — DCU 是什么 + 为什么赛题选 DCU

### 开场 (1 min)

今天分享分 5 段, 总共 30 min, 末了留 3 min 问答。主题: **DCU vs NVIDIA GPU** + **HIP 编程模型**。和分享 1 (prefill/decode) + 分享 2 (vLLM 架构) 的内容是互补的: 那两个是 LLM 推理的"通用知识", 这个是**赛题特有的硬件/生态知识**。

### 国产化背景 (1.5 min)

"DCU" = Deep Computing Unit, 国产 GPGPU 的统称:

- **海光 (Hygon)** — x86 CPU 出身, DCU 用 CDNA 微架构, 对标 AMD Instinct 系列, 走 ROCm/HIP 生态
- **壁仞 (Biren)** — BR106, 自研 BIRENSUPA 架构, 自有软件栈 (BIREN Compute)
- **寒武纪 (Cambricon)** — MLU, 思元 590, CNCL 编程模型
- **摩尔线程 (Moore Threads)** — MTT S5000, 走 MUSAA 兼容 CUDA 路径

赛题选海光的原因:

- **ROCm/HIP 兼容路径最成熟** — 90% 兼容 CUDA, vLLM / PyTorch / Triton 都有 first-class 支持
- 海光 DCU 是国内**唯一**能跑 vLLM 0.18.x + Qwen3 的开箱即用平台
- 其他国产 GPGPU 软件栈要么 CUDA 兼容不完整, 要么 vLLM 后端没移植

### DCU 在 ROCm 生态中的位置 (1.5 min)

ROCm = AMD 的开源 GPU 计算栈, 核心组件:

- **HIP** — 90% 兼容 CUDA 的 C++ API (后面 §12:00-20:00 详讲)
- **HCC / HIPCC** — 编译器 (基于 LLVM/Clang)
- **MIOpen** — 类 cuDNN 库
- **rocBLAS / rocFFT** — 类 cuBLAS / cuFFT
- **ROCR-Runtime** — 类 CUDA Runtime
- **Triton-AMD** — Triton 的 AMD 后端 (ROCm 5.6+ 默认带)

我们拿到手的硬件 (P0 Task 0.1 决定):

- **如果 = K100**: CDNA2 (gfx90a), MI250X 同代, **无 FP8**
- **如果 = Z100**: CDNA3 (gfx942), MI300X 同代, **有 FP8 FNUZ**

(录的时候根据 P0 实际结果, 把上面两条之一划掉, 只留对的。)

### §5:00 小结 (1 min)

- DCU = 国产 GPGPU 统称, 赛题用海光
- 海光走 ROCm/HIP 生态
- 我们的硬件 = K100 (gfx90a) 或 Z100 (gfx942), P0 已确认

---

## 5:00-12:00 — CDNA2 vs CDNA3 微架构

### CDNA 架构总览 (2 min)

AMD 现在的 GPGPU 微架构分两大族:

- **CDNA** (Compute DNA) — 数据中心 / HPC, 砍掉图形硬件
- **RDNA** (RDNA DNA) — 消费 / 游戏, 保留图形硬件

DCU 都是 CDNA。我们关心两代:

| 架构 | 代次 | 典型 GPU | 海光 SKU | gfx target |
|------|------|----------|----------|------------|
| **CDNA2** | 第 2 代 | MI250X | **K100** | gfx90a |
| **CDNA3** | 第 3 代 | MI300X | **Z100** | gfx942 |
| CDNA4 | 第 4 代 | MI355X | (未来) | gfx950 |

### CDNA2 (gfx90a, MI200 系列) (2 min)

- **算力**: FP16/bf16 强 (理论 ~383 TFLOPS peak, MI250X)
- **显存**: HBM2e, 128 GB (MI250X), K100 大约 64 GB
- **FP8**: **❌ 不支持** — 这就是 K100 的硬伤
- **wave size**: 64 threads/wave (类比 NVIDIA warp)
- **LDS** (类比 shared memory): 64 KB per CU
- **寄存器文件**: 256 KB per CU
- **CU 数**: MI250X = 110 CU × 2 dies = 220 CU

### CDNA3 (gfx942, MI300 系列) (2 min)

- **算力**: FP16/bf16 比 CDNA2 高 ~1.5x; 新增 **FP8** (FNUZ 变体)
- **显存**: HBM3, 192 GB (MI300X), Z100 大约 80 GB
- **FP8**: **✅ 原生 FNUZ 变体** (`__hip_fp8_e4m3_fnuz`)
- **wave size**: 64 threads/wave (和 CDNA2 一样)
- **LDS**: 64 KB per CU (没变)
- **寄存器文件**: 256 KB per CU (没变)
- **CU 数**: MI300X = 304 CU

### 关键差异 (1 min)

(在 PPT 上画表格)

| 维度 | CDNA2 (gfx90a) | CDNA3 (gfx942) |
|------|----------------|----------------|
| FP8 | ❌ | ✅ FNUZ |
| 显存 | HBM2e | HBM3 |
| 带宽 | ~3.2 TB/s (MI250X) | ~5.3 TB/s (MI300X) |
| 算力 (bf16) | ~383 TFLOPS | ~1300 TFLOPS |
| 海光 SKU | K100 | Z100 |

### §12:00 小结 (1 min)

(录的时候, 念出我们的实际 SKU + 念出 1 句 "这是我们后面所有优化的硬件前提")

- CDNA2 = 无 FP8, 我们的 FP8 路线会降级为 INT8 / bf16
- CDNA3 = 有 FP8 FNUZ, 但和 NVIDIA OCP FP8 不兼容 (下面 §25:00 详讲)
- 显存带宽 + 算力 是后续 vLLM 优化的物理上限

---

## 12:00-20:00 — HIP vs CUDA 编程模型

### HIP 是什么 (1.5 min)

HIP = **H**eterogeneous-Compute **I**nterface for **P**ortability. AMD 出品, 设计目标:

- **90% 兼容 CUDA** — 大部分 CUDA kernel 可以**机械翻译** (sed 一遍) 成 HIP
- 一套代码, 编译时可走 CUDA (nvcc) 或 HIP (hipcc)
- **API 一对一映射**:
  - `cudaMalloc` → `hipMalloc`
  - `cudaMemcpy` → `hipMemcpy`
  - `cudaStream_t` → `hipStream_t`
  - `__global__ void kernel()` → `__global__ void kernel()` (不变!)
  - `threadIdx.x` → `threadIdx.x` (不变!)
  - `__shared__` → `__shared__` (不变!)

### 主要差异 (2 min)

虽然 API 大部分兼容, 编译流程和启动语法有差异:

**Kernel 启动**:
- CUDA: `kernel<<<grid, block>>>(args);` (三括号语法)
- HIP: 同上! **HIP 保留三括号语法** (为了兼容)

**编译**:
- CUDA: `nvcc -arch=sm_80 kernel.cu`
- HIP (DCU): `hipcc -target=gfx942 kernel.cpp` 或 `-target=gfx90a`

**库**:
- `cublas` → `rocblas`
- `cudnn` → `miopen`
- `cuda-runtime` → `hip-runtime`
- `nccl` (多卡) → `rccl`

**宏判断** (写可移植代码时):
```cpp
#ifdef __HIP_PLATFORM_AMD__
    hipLaunchKernelGGL(...);
#else
    myKernel<<<grid, block>>>(...);
#endif
```

### Hello-world HIP kernel (2.5 min)

(录的时候**实际**在 DCU 上跑一遍, 屏幕贴终端)

```cpp
// hello_hip.cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello_kernel(int *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = i * 2;
}

int main() {
    int *d_x;
    hipMalloc(&d_x, 1024 * sizeof(int));
    hello_kernel<<<4, 256>>>(d_x);  // 三括号语法, 和 CUDA 一模一样
    hipDeviceSynchronize();

    int h_x[1024];
    hipMemcpy(h_x, d_x, 1024 * sizeof(int), hipMemcpyDeviceToHost);
    printf("h_x[7] = %d\n", h_x[7]);  // 应该是 14
    hipFree(d_x);
    return 0;
}
```

编译 + 跑:
```bash
hipcc -target=gfx942 hello_hip.cpp -o hello_hip   # 录的时候 -target 改成我们的实际值
./hello_hip
```

### vLLM 和 HIP 的关系 (2 min)

vLLM 在 DCU 上能跑, 关键在两个 build-time 环境变量:

- **`PYTORCH_ROCM_ARCH=gfx90a` 或 `gfx942`** — PyTorch 编译时给 ROCm 后端指定的 target
- **`TORCH_CUDA_ARCH_LIST`** — 在 ROCm 路径下, vLLM 用这个判断启用哪些 kernel (讽刺地沿用了 NVIDIA 变量名)

实际工作流:

1. PyTorch 检测到 ROCm → 加载 `torch.version.hip`
2. vLLM `platforms/rocm.py` 选 HIP 后端
3. Attention backend 选择:
   - `FLASH_ATTN` → 走 ROCm/flash-attention fork (spec §5.3 提到)
   - `TRITON_ATTN` → Triton-AMD 编译
   - `XFORMERS` → xformers 的 HIP 后端 (可能有坑)
4. Linear / GEMM → rocBLAS 或 aiter

(录的时候, 翻 spec §5.2 找 vLLM 在 DCU 上 backend 选择的流程图, 贴屏幕上。)

### §20:00 小结 (1 min)

- HIP 90% 兼容 CUDA, 编译时换编译器, API 基本不动
- DCU 上写新 kernel = 写 CUDA 代码 + `hipcc` 编译
- vLLM 在 DCU 上能跑, 关键是 build-time env vars + 选对 backend

---

## 20:00-27:00 — 27B 模型的 DCU 适配

### 显存账 (2 min)

Qwen3.5-27B, 假设 bf16:

- **参数**: 27B × 2 bytes = **54 GB**
- **KV cache** (32K context, batch=4, 64 layers): ~16 GB (bf16)
- **激活 + workspace**: ~4 GB
- **总**: ~74 GB

DCU 显存:

- **K100 (gfx90a)**: ~64 GB HBM2e — **装不下 bf16 满载**
- **Z100 (gfx942)**: ~80 GB HBM3 — 装得下, 但紧

### FP8 量化: 显存减半 (2 min)

(重点, 这是 DCU 优化的核心)

| 量化方案 | 显存 | DCU 兼容性 |
|----------|------|------------|
| bf16 满载 | 54 GB | ✅ K100 + Z100 |
| INT8 量化 (per-tensor) | 27 GB | ✅ K100 + Z100 |
| **FP8 OCP** (NVIDIA 风格) | 27 GB | ❌ DCU 不认 |
| **FP8 FNUZ** (AMD 风格) | 27 GB | ✅ **仅 Z100** |

(录的时候, 念 spec §5.4 原文 "FNUZ = Finite + No inf + Unsigned zero", 然后说 "这就是为什么我们不能直接用 NVIDIA 训练好的 FP8 checkpoint, 必须重新校准 scale")

### 算力 / 带宽理论上限 (1.5 min)

batch=1 推理 (decode bound) 关键看 **HBM 带宽**:

- **K100**: ~3.2 TB/s (假设) → batch=1 decode ~80 tok/s (上限)
- **Z100**: ~5.3 TB/s (假设) → batch=1 decode ~130 tok/s (上限)

实际数字 (P2 Task 2.2 跑分) 大概只能达到理论上限的 60-70%, 这是因为:

- KV cache 读写不只是纯 HBM
- Attention 计算要 SM 周期
- 调度 / 同步开销

### 3 必做项在 DCU 上的实现路径 (1.5 min)

(过一遍, 不深讲, 后面 P3 详做)

1. **PagedAttention** (vLLM 内置, 无需改) — TRITON_ATTN / FLASH_ATTN 后端选 ROCm 兼容的
2. **动态 FP8 KV cache** (自定义算子) — **仅 Z100**; K100 改 INT8 或保留 bf16
3. **torch.compile** (高门槛路径) — 0.18.1 在 DCU 上**未验证**, P3 风险最高

### §27:00 小结 (1 min)

- 27B bf16 = 54 GB, K100 装不下, Z100 紧
- FP8 FNUZ 减半到 27 GB, **但仅 Z100**
- batch=1 decode 上限 ~80-130 tok/s, 实际 60-70%
- 3 必做项在 DCU 上需要 SKU-specific 决策

---

## 27:00-30:00 — Q&A + 总结

### 总结 (2 min)

(这段录的时候**慢点说**, 让队员有时间记笔记)

**DCU 不是 NVIDIA, 优化策略要改**:

1. **不要假设 max-autotune 模板可用** — Triton max-autotune 在 CDNA 上是 ROCm 5.6+ 才稳定, 而且给的候选 kernel 数比 NVIDIA 少
2. **Triton + DCU 已知坑** (spec §5.5):
   - `tl.atomic_*` for FP8 有 bug — 走 reduction 时可能 race condition
   - 部分 `tl.dot` scale 组合编译失败 — P0 期间已验证 5 行 matmul + FP8 store 最小 case
3. **优先用已验证 backend** (spec §5.3):
   - `TRITON_ATTN` — 已验证, 默认
   - `FLASH_ATTN` via ROCm fork — 已验证, 设 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` 启用 FP8 路径
   - 不要碰 `XFORMERS` — ROCm 后端有未合并 bug

**FP8 KV cache 的 FNUZ vs OCP** 是 DCU + NVIDIA 之间最大的坑, 任何从 NVIDIA 移植的 FP8 量化代码**必须**改 FNUZ 重新校准。

### 留 3 min 问答 (1 min)

(录的时候停在这, 留个尾巴, 实际问答现场做)

### 给队员的 3 个 action item

1. **队长**: 看 spec §5.4 + §5.5, 拍板 KV cache 选型
2. **队员 A** (Kernel owner): 跑 P0 Task 0.4 (vLLM backend 路径 smoke), 选 attention backend
3. **队员 B** (vLLM owner): 翻 ROCm/flash-attention fork 的 issue tracker, 看有没有 gfx942 / gfx90a 相关 bug
4. **队员 C** (浮动): 准备 P2 Task 2.2 baseline 跑分, 跑出来的 HBM 带宽数字 vs 今天说的理论上限

---

## 全员 ROCm precision-support 总结

URL: https://rocm.docs.amd.com/en/latest/reference/precision-support.html

> **填表说明**: 全员 (队长 + 队员 A + B + C) 各自读一遍上面这个 ROCm 官方 precision-support 文档, 然后在下面 4 个 `**待填**` 位置各写 1 段总结。每段 ~3-5 句话, 要点:
> - 你理解的 FNUZ vs OCP 含义是什么
> - 这个文档对我们这个赛题有什么具体影响
> - 至少 1 个 actionable conclusion (例如 "K100 上必须放弃 FP8 KV cache")

### 队长 (recoletas)

**待填** — 阅后写 1 段: FNUZ vs OCP 含义 / 我们的 DCU 在表中的哪一格 / FP8 KV cache 选型的含义

### 队员 A (Kernel owner)

**待填** — 阅后写 1 段: Triton 在 gfx90a / gfx942 上的兼容性 / `tl.atomic_*` FP8 bug 是否文档化

### 队员 B (vLLM owner)

**待填** — 阅后写 1 段: ROCm/flash-attention fork 在我们 SKU 上的支持情况 / CK backend vs aiter 的选择

### 队员 C (浮动)

**待填** — 阅后写 1 段: 影响精度评测 (OpenCompass) 的 FP8 容差 / Δ > 10% 红线
