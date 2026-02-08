---
topic: "Model Optimization for Production AI Systems"
date: 2026-02-07
status: research-brief
---

# Research Brief: Model Optimization for Production AI Systems

## Summary

Model optimization encompasses the full stack of techniques for making deep learning models—especially large language models—run faster, use less memory, and cost less to serve. The field spans compiler infrastructure (torch.compile, AOTInductor, TVM, XLA), GPU execution primitives (CUDA graphs, operator fusion), precision reduction (FP8/FP4 training, INT4/INT8 quantization, QAT), memory-efficient algorithms (FlashAttention), and production serving systems (TensorRT, ONNX Runtime, vLLM, SGLang).

The field has undergone six paradigm shifts: (1) symbolic graphs → eager execution → compilation, (2) hand-tuned kernels → compiler-generated kernels, (3) FLOP reduction → IO-awareness, (4) fixed quantization → adaptive quantization, (5) single-precision → mixed-precision → FP8/FP4, and (6) monolithic inference → disaggregated serving with KV cache as a first-class resource.

As of early 2026, the ecosystem is converging: torch.compile is standard for inference, FP8 training is validated at 671B scale (DeepSeek-V3), FlashAttention-3 reaches 75% H100 utilization, and FP4 training has been demonstrated at 12B scale. Key open problems include compilation cold-start overhead, the FlashAttention-PagedAttention composability gap, extreme low-precision (INT4/FP4) quality at frontier scale, and the compiler-first vs. kernel-first philosophical divide.

## Key Concepts

### Foundational (Prerequisites)

- **Roofline Model**: Visual framework plotting attainable GFLOPS against operational intensity (FLOPs/byte). Kernels are either memory-bound (limited by bandwidth) or compute-bound (limited by peak FLOPS). This framework explains *why* fusion matters (memory-bound ops) and *why* FlashAttention works (attention is memory-bound). Common misconception: that reducing FLOPs always improves speed. In practice, memory bandwidth is often the bottleneck.

- **Operator Fusion**: Combining multiple GPU kernel launches into one, eliminating intermediate memory reads/writes and launch overhead. XLA (2017) established this as the primary DL compiler optimization. TorchInductor generates fused Triton kernels automatically. Misconception: fusion is always beneficial—in practice, over-fusing can increase register pressure and reduce parallelism.

- **CUDA Graphs**: Capture a sequence of GPU operations as a graph, then replay it as a single unit. Eliminates per-kernel CPU launch overhead (5–15μs per launch). Critical for LLM autoregressive decoding where many small kernels dominate. Misconception: CUDA graphs work with dynamic shapes—they don't; any shape change requires a new graph capture.

### Core Techniques

- **Mixed Precision Training**: Use FP16/BF16 for compute, FP32 for master weights and accumulation, with loss scaling to prevent gradient underflow (Micikevicius et al., 2018). Nearly 2x speedup on Tensor Core GPUs with no accuracy loss. The universal standard since Volta.

- **torch.compile / TorchDynamo + TorchInductor**: PyTorch 2's compilation system. TorchDynamo captures FX graphs from unmodified Python via bytecode analysis; TorchInductor generates optimized Triton (GPU) or C++ (CPU) kernels. Three modes: `default` (balanced), `reduce-overhead` (CUDA graphs), `max-autotune` (Triton template search). 2.27x geometric mean inference speedup on 180+ models. Misconception: torch.compile always helps—graph breaks from unsupported ops can fragment and negate gains.

- **AOT Compilation (AOTInductor)**: Ahead-of-time compilation via `torch.export` + `aoti_compile_and_package()`, producing shared libraries deployable in non-Python environments. Eliminates JIT warmup overhead entirely. Standard practice at Meta for production inference. Beta status as of PyTorch 2.9–2.10.

- **FlashAttention**: IO-aware tiled attention that keeps intermediate results in GPU SRAM instead of HBM (Dao et al., 2022). Reduces HBM accesses from $O(N^2)$ to $O(N^2 d^2 / M)$. Uses online softmax (Milakov & Gimelshein, 2018) for block-wise computation and recomputes during backward instead of storing. FlashAttention-2 reaches 73% A100 utilization; FlashAttention-3 reaches 75% H100 utilization with FP8 and async execution. Enabled context lengths growing from 2–4K to 1M+.

- **Post-Training Quantization (PTQ)**: Quantize weights and/or activations after training. GPTQ uses second-order information for 3–4 bit weight quantization. SmoothQuant migrates quantization difficulty from activations to weights for W8A8. AWQ (MLSys 2024 Best Paper) protects activation-aware salient weights. Most models retain >99.5% accuracy with INT8 PTQ.

- **Quantization-Aware Training (QAT)**: Insert fake quantization nodes during training; the model learns to be robust to quantization noise. Jacob et al. (2018) defined the standard: quantize with affine mapping $q = \text{round}(\text{clamp}(r/S + Z, 0, 2^b - 1))$, use Straight-Through Estimator for gradients. Google's AQT produces bit-exact training-serving consistency.

### Advanced / Cutting Edge

- **FP8 Training**: E4M3 for forward pass, E5M2 for backward pass (Micikevicius et al., 2022). DeepSeek-V3 validated at 671B scale with <0.25% accuracy loss. DeepL achieved 80% MFU on 544 H100s after 15 months of optimization. Requires fine-grained blockwise scaling to work around Hopper Tensor Core accumulation limitations.

- **FP4 Training**: NVIDIA Blackwell introduces native FP4 (NVFP4). A 12B model was trained on 10T tokens in FP4 with loss curves matching FP8 baselines—the first successful billion-parameter FP4 training. The OCP Microscaling (MX) spec standardizes MXFP4/6/8 formats. Still early—unclear if it scales to 100B+.

- **FlexAttention**: PyTorch-native programmable attention API that generates fused FlashAttention-style Triton kernels via torch.compile. Supports causal, sliding window, ALiBi, document masking, and more—without hand-written kernels. Automatically generates backward passes.

- **Speculative Decoding**: Draft model proposes $k$ tokens; target model verifies all $k$ in one forward pass via rejection sampling. Lossless—output distribution is identical to the target. 2–3x speedup. EAGLE series (ICML/EMNLP/NeurIPS 2024–25) and Medusa are production-ready variants.

- **BitNet b1.58**: Ternary-weight LLMs ({-1, 0, +1}) matching half-precision Transformer performance while being 2.7–4.1x faster and using 3.5–7.2x less memory. Requires training from scratch—no post-hoc compression.

## Important Papers & Sources

### Compiler Infrastructure
- **[LLVM](https://llvm.org/pubs/2004-01-30-CGO-LLVM.html)** by Lattner & Adve (2004) — Typed SSA-based IR underpinning all modern ML compilers
- **[Theano](https://conference.scipy.org/proceedings/scipy2010/bergstra.html)** by Bergstra et al. (2010) — First symbolic-graph-to-GPU compiler for DL
- **[Halide](https://people.csail.mit.edu/jrk/halide12/)** by Ragan-Kelley et al. (2013) — Algorithm/schedule separation paradigm that influenced all DL compilers
- **[TVM](https://arxiv.org/abs/1802.04799)** by Chen et al. (OSDI 2018) — End-to-end DL compiler with learned cost models
- **[Triton](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)** by Tillet et al. (MAPL 2019) — Block-level GPU programming; now TorchInductor's backend
- **[MLIR](https://arxiv.org/abs/2002.11054)** by Lattner et al. (CGO 2021) — Multi-level dialect-based compiler infrastructure
- **[PyTorch 2 / torch.compile](https://dl.acm.org/doi/10.1145/3620665.3640366)** by Ansel et al. (ASPLOS 2024) — TorchDynamo + TorchInductor

### Quantization & Precision
- **[Mixed Precision Training](https://arxiv.org/abs/1710.03740)** by Micikevicius et al. (ICLR 2018) — FP16 + loss scaling + FP32 master weights
- **[QAT](https://arxiv.org/abs/1712.05877)** by Jacob et al. (CVPR 2018) — Definitive QAT methodology with integer-only inference
- **[Deep Compression](https://arxiv.org/abs/1510.00149)** by Han et al. (ICLR 2016 Best Paper) — Prune + quantize + Huffman coding (35–49x)
- **[FP8 Formats](https://arxiv.org/abs/2209.05433)** by Micikevicius et al. (2022) — E4M3/E5M2 specification
- **[GPTQ](https://arxiv.org/abs/2210.17323)** by Frantar et al. (ICLR 2023) — One-shot PTQ for 175B+ LLMs
- **[SmoothQuant](https://arxiv.org/abs/2211.10438)** by Xiao et al. (ICML 2023) — Activation-to-weight difficulty migration
- **[AWQ](https://arxiv.org/abs/2306.00978)** by Lin et al. (MLSys 2024 Best Paper) — Activation-aware weight quantization

### Attention & Serving
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** by Dao et al. (NeurIPS 2022) — IO-aware tiled attention
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)** by Dao (2023) — 50–73% A100 utilization
- **[FlashAttention-3](https://arxiv.org/abs/2407.08608)** by Dao & Shah (2024) — Hopper async + FP8, 75% H100 utilization
- **[PagedAttention/vLLM](https://arxiv.org/abs/2309.06180)** by Kwon et al. (SOSP 2023) — Virtual memory for KV cache
- **[Speculative Decoding](https://arxiv.org/abs/2211.17192)** by Leviathan et al. (ICML 2023) — Lossless 2–3x autoregressive speedup

### Modern Systems (2024–2026)
- **[FlexAttention](https://pytorch.org/blog/flexattention/)** (August 2024) — Programmable attention via torch.compile
- **[Flashlight](https://arxiv.org/abs/2511.02043)** (November 2025) — Compiler-native tiled attention generation
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** (MLSys 2025 Best Paper) — Customizable JIT attention engine for serving
- **[Liger-Kernel](https://arxiv.org/abs/2410.10989)** (October 2024) — Fused Triton kernels: 20% throughput gain, 60% memory reduction
- **[ThunderKittens](https://github.com/HazyResearch/ThunderKittens)** — Tile-based CUDA kernel DSL; v2.0 (Jan 2026) with Blackwell/FP4 support
- **[TorchAO](https://arxiv.org/html/2507.16099v1)** — PyTorch-native INT4/INT8/FP8 quantization (ICML 2025)
- **[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1)** — First 671B model trained in FP8
- **[NVFP4 Pretraining](https://arxiv.org/html/2509.25149v1)** — 12B model trained on 10T tokens in FP4
- **[State of torch.compile (Aug 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)** — Authoritative assessment by Edward Yang

## Technical Details

### Quantization: Affine Mapping

The standard quantization scheme (Jacob et al., 2018):

$$q = \text{round}\left(\text{clamp}\left(\frac{r}{S} + Z,\; 0,\; 2^b - 1\right)\right)$$

where $S$ is scale factor, $Z$ is zero-point, $b$ is bit-width. Dequantization: $r \approx S(q - Z)$.

### SmoothQuant: Difficulty Migration

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X}\hat{W}$$

Per-channel scaling: $s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$ with $\alpha \approx 0.5$.

### AWQ: Activation-Aware Scaling

$$s^* = \arg\min_s \| Q(W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot X) - WX \|$$

### FlashAttention: IO Complexity

Standard attention: $\Theta(Nd + N^2)$ HBM accesses.
FlashAttention: $\Theta(N^2 d^2 / M)$ where $M$ is SRAM size. For $N=2048$, $d=64$, $M=100$KB → ~10x reduction.

Online softmax update rule:

$$m_{i+1} = \max(m_i, x_{i+1}), \quad d_{i+1} = d_i \cdot e^{m_i - m_{i+1}} + e^{x_{i+1} - m_{i+1}}$$

### Mixed Precision Training

Loss scaling: $\hat{L} = L \cdot S$, compute gradients, then unscale: $\nabla W = \frac{1}{S} \nabla_{\hat{L}} W$. Weight update in FP32: $W_{\text{FP32}} \leftarrow W_{\text{FP32}} - \eta \cdot \nabla W_{\text{FP32}}$.

### Roofline Model

$$\text{Attainable GFLOPS} = \min(\text{Peak GFLOPS},\; \text{Operational Intensity} \times \text{Peak Bandwidth})$$

### Speculative Decoding: Rejection Sampling

Accept token $x_i$ with probability $\min\left(1,\; \frac{p_{\text{target}}(x_i)}{p_{\text{draft}}(x_i)}\right)$. If rejected at position $j$, resample from $\max(0,\; p_{\text{target}}(x_j) - p_{\text{draft}}(x_j))$.

## Code Opportunities

### 1. torch.compile modes comparison
Benchmark `default` vs `reduce-overhead` vs `max-autotune` on a transformer model. Show compilation time vs inference speedup tradeoff.

```python
import torch

model = TransformerBlock().cuda().eval()
x = torch.randn(32, 128, 768, device="cuda")

for mode in ["default", "reduce-overhead", "max-autotune"]:
    compiled = torch.compile(model, mode=mode)
    # Warmup, then benchmark
```

### 2. CUDA Graphs capture and replay
Show manual CUDA graph capture for a training step vs torch.compile's automatic CUDA graph integration.

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    optimizer.zero_grad(set_to_none=True)
    y = model(static_input)
    loss = F.mse_loss(y, static_target)
    loss.backward()
    optimizer.step()
# Replay: static_input.copy_(data); g.replay()
```

### 3. AOT compilation pipeline
Export → AOT compile → load and run without Python overhead.

```python
exported = torch.export.export(model, (example_input,))
package_path = torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")
compiled = torch._inductor.aoti_load_package(package_path)
```

### 4. Mixed precision training with AMP
Show `torch.amp.autocast` + `GradScaler` pattern.

### 5. Fused softmax kernel in Triton
Write a fused softmax from scratch in ~20 lines of Triton, demonstrating the memory bandwidth advantage over PyTorch's native implementation.

### 6. Quantization comparison
Compare TorchAO INT4, bitsandbytes NF4, and AWQ on the same model—measure memory, latency, and perplexity on WikiText-2.

### 7. FlashAttention vs standard attention benchmark
Side-by-side memory and speed comparison at various sequence lengths (512 to 16K).

## Open Questions

1. **Compilation cold-start**: torch.compile JIT adds 20+ seconds; AOTInductor helps for inference but training compilation on GPU clusters wastes expensive compute. No general solution.

2. **FlashAttention-PagedAttention composability**: FlashAttention assumes contiguous memory; PagedAttention fragments it into pages. 20–26% overhead. Each new FA version needs PagedAttention-aware ports. vAttention proposes OS-level virtual memory but adoption is limited.

3. **Extreme low-precision quality**: INT4 causes "significant accuracy drops for decoder-only models"—precisely the models most needing compression. FP4 training demonstrated at 12B but unproven at 100B+. BitNet b1.58 matches FP16 but requires training from scratch.

4. **Compiler-first vs kernel-first**: Google's XLA generalizes to new patterns; CUDA's kernel-first (FlashAttention, custom GEMM) achieves higher peak utilization. torch.compile faces "stiff competition from JAX, which is years ahead in compile-driven parallelism" (ezyang, Aug 2025).

5. **Graph breaks**: torch.compile can't trace C extensions, data-dependent control flow, or unsupported builtins. After 8 recompiles per frame, permanent eager fallback. Distributed collectives unoptimized by default.

6. **KV cache at long contexts**: As windows expand to 1M+, KV cache dominates memory. Compression techniques are model-specific. Disaggregated architectures help but add complexity.

7. **LLM-generated kernel correctness**: KernelBench shows LLMs struggle to produce functionally correct GPU kernels. Generated kernels exploit benchmark loopholes and fail to generalize.

8. **Training-serving precision mismatch**: Models trained in FP8 but served in INT4/FP4—compounding precision mismatches are poorly understood, especially for reasoning models.

## Suggested Post Structure

```
## Introduction
Why model optimization matters now: LLMs cost $1M+/day to serve;
optimization is the difference between viable and bankrupt.

## The Optimization Stack: A Mental Model
Roofline model → memory-bound vs compute-bound → where each technique fits.
Diagram: compiler ↔ kernel ↔ precision ↔ memory ↔ serving.

## Compilation: From Python to Fast Code
torch.compile internals (TorchDynamo → FX graph → TorchInductor → Triton).
AOTInductor for production. CUDA graphs: capture-and-replay.
Code: torch.compile modes benchmark.

## FlashAttention: The IO-Awareness Revolution
Why attention is memory-bound. Online softmax → tiling → FlashAttention.
FA-1 → FA-2 → FA-3 evolution. FlexAttention for programmable patterns.
Code: FlashAttention vs standard attention benchmark.

## Precision: From FP32 to FP4
Mixed precision training (AMP). FP8 at scale (DeepSeek-V3, DeepL).
FP4 and Microscaling formats. The precision frontier.
Code: AMP training pattern.

## Quantization: Compressing Trained Models
PTQ landscape: GPTQ, SmoothQuant, AWQ. QAT methodology.
TorchAO for PyTorch-native quantization. When to use which.
Code: TorchAO INT4 vs AWQ comparison.

## Kernel Optimization: Writing Fast GPU Code
Triton programming model. Liger-Kernel for drop-in LLM optimization.
ThunderKittens for hand-tuned kernels. KernelBench evaluation.
Code: Fused softmax in Triton.

## Production Serving
TensorRT vs torch.compile. vLLM vs SGLang. PagedAttention.
Speculative decoding. Cold-start and dynamic shapes.

## Practical Guide: What to Use When
Decision tree by use case: training vs inference, latency vs throughput,
GPU generation, model size. The 80/20 of optimization.

## What's Next
FP4 scaling, compiler-kernel convergence, LLM-generated kernels,
disaggregated inference, the end of manual optimization?
```

## Raw Notes

### Industry Benchmarks
- Meta: 5T+ inferences/day across 50 data centers
- DeepSeek-V3: 671B params, FP8 training, <0.25% accuracy loss vs BF16
- DeepL: 44.6% → 67% → 80% MFU over 15 months of FP8 optimization on 544 H100s
- Fireworks AI: 2.3x speedup from CUDA graphs on LLaMA-v2-7B; FireAttention v2 up to 8x
- SGLang: 29% throughput advantage over vLLM (16,215 vs 12,553 tok/s)
- NVIDIA B200: 20 PFLOPS FP4, 9 PFLOPS FP8; DGX B200: 3x training, 15x inference over H100

### Key Libraries and Tools
- **torch.compile**: [pytorch/pytorch](https://github.com/pytorch/pytorch) — Production-grade, PyTorch 2.x default compiler
- **TorchAO**: [pytorch/ao](https://github.com/pytorch/ao) — INT4/INT8/FP8 quantization, ICML 2025
- **Triton**: [triton-lang/triton](https://github.com/triton-lang/triton) — Block-level GPU kernel language
- **FlashAttention**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — Gold standard attention
- **FlashInfer**: [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) — JIT attention engine (MLSys 2025 Best Paper)
- **vLLM**: [vllm-project/vllm](https://github.com/vllm-project/vllm) — Industry-standard LLM serving
- **SGLang**: [sgl-project/sglang](https://github.com/sgl-project/sglang) — High-performance serving, 400K+ GPUs
- **Liger-Kernel**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — Fused Triton kernels for LLM training
- **ThunderKittens**: [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) — Tile-based CUDA kernel DSL
- **TensorRT**: [pytorch/TensorRT](https://github.com/pytorch/TensorRT) — NVIDIA inference optimizer
- **ONNX Runtime**: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) — Cross-platform inference
- **AWQ**: [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) — MLSys 2024 Best Paper
- **bitsandbytes**: [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — 4/8-bit quantization
- **Apache TVM**: [apache/tvm](https://github.com/apache/tvm) — End-to-end DL compiler

### Key Tutorials
- [torch.compile intro](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [AOTInductor tutorial](https://docs.pytorch.org/tutorials/recipes/torch_export_aoti_python.html)
- [CUDA Graphs in PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [GPU-Mode lectures](https://github.com/gpu-mode/lectures) — Community lecture series on CUDA/Triton
- [depyf](https://depyf.readthedocs.io/) — Debug torch.compile internals

### Datasets for Benchmarking
- MLPerf Inference/Training (industry standard)
- WikiText-2/103 (perplexity for quantization quality)
- HellaSwag (commonsense reasoning for QAT evaluation)
- ImageNet, CIFAR-10 (CNN optimization benchmarks)
- SQuAD (NLP inference benchmarks)

### Validation Status Table
| Technology | Status | Production? |
|---|---|---|
| torch.compile (inference) | Mature | Yes (vLLM, SGLang, Meta) |
| torch.compile (training) | Active dev | Partial (Meta, limited patterns) |
| AOTInductor | Beta | Yes (Meta production inference) |
| FlashAttention-3 | Stable | Yes (widely deployed) |
| FlexAttention | Stable | Yes (vLLM, PyTorch) |
| CUDA Graphs | Mature | Yes (vLLM, SGLang, Fireworks) |
| FP8 Training | Validated | Yes (DeepSeek-V3, DeepL) |
| FP4 Training | Early | Research (12B demonstrated) |
| AWQ/GPTQ (INT4 PTQ) | Mature | Yes (widely deployed) |
| TorchAO | Stable | Yes (Meta, SGLang, vLLM) |
| TensorRT-LLM | Mature | Yes (NVIDIA ecosystem) |
| Speculative Decoding | Mature | Yes (vLLM, TRT-LLM, SGLang) |
| Liger-Kernel | Stable | Yes (LinkedIn, community) |
| ThunderKittens 2.0 | Stable | Yes (Together AI, Cursor) |
| BitNet b1.58 | Research | Limited (Microsoft bitnet.cpp) |

### Paradigm Shifts (Historical)
1. Symbolic graphs → eager execution (2010–2019)
2. Eager execution → compilation (2019–2024)
3. Hand-tuned kernels → compiler-generated kernels (2014–2022)
4. FLOP reduction → IO-awareness (2017–2022)
5. Fixed quantization → adaptive quantization (2018–2024)
6. Single-precision → mixed-precision → FP8/FP4 (2015–present)

### Timeline
- 2004: LLVM
- 2009: Roofline Model
- 2010: Theano
- 2013: Halide
- 2014: cuDNN
- 2015: Deep Compression, BinaryConnect, limited precision training
- 2016: Deep Compression (ICLR Best Paper), TensorRT
- 2017: XLA, ONNX
- 2018: Mixed Precision Training, QAT, TVM, CUDA Graphs, Online Softmax, ONNX Runtime
- 2019: Triton, MLIR, PyTorch
- 2020: Ansor, Rammer
- 2022: FlashAttention, FP8 formats
- 2023: GPTQ, SmoothQuant, FlashAttention-2, PagedAttention/vLLM, Speculative Decoding
- 2024: torch.compile (ASPLOS), FlashAttention-3, AWQ (MLSys Best Paper), FlexAttention, Liger-Kernel, DeepSeek-V3 FP8 training
- 2025: FlashInfer (MLSys Best Paper), TorchAO (ICML), Flashlight, ThunderKittens 2.0, FP4 pretraining
- 2026: QuantLRM, D2Quant, ongoing convergence
