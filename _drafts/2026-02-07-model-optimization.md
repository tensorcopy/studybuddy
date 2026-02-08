---
title: "The Model Optimization Stack: From torch.compile to Production Serving"
date: 2026-02-07
categories: [mlsystems, optimization]
tags: [torch-compile, flashattention, quantization, cuda, triton, inference, serving, fp8, vllm]
mathjax: true
toc: true
description: "A deep technical guide to the full model optimization stack — compilers, kernels, precision, quantization, and serving — covering torch.compile, FlashAttention, FP8/FP4 training, AWQ/GPTQ quantization, Triton, and production inference with vLLM and SGLang."
---

## Introduction

Serving a 70-billion-parameter language model at scale costs on the order of $1 million per day in GPU compute. That is not a worst-case estimate. It is roughly what you get when you multiply the number of A100 or H100 hours required for continuous inference at production throughput by current cloud pricing. Meta runs more than 5 trillion model inferences per day across its family of products and data centers. Google processes every search query, every Gmail smart reply, and every YouTube recommendation through models that would be economically impossible to serve without aggressive optimization. The gap between "this model works in a notebook" and "this model runs in production" is not a minor engineering detail -- it is a chasm that determines whether an AI product is viable or bankrupt.

And the problem is getting worse. Model sizes have grown by roughly 1000x over the past five years, while GPU memory has grown by roughly 4x. The NVIDIA H100 delivers about 3x the raw FLOPS of the A100, but frontier models have grown 10-100x over the same period. We are in an era where **model optimization** -- the full stack of techniques that make models faster, smaller, and cheaper to run -- is no longer a nice-to-have. It is table stakes.

What makes this moment particularly interesting is that we are living through not one but six overlapping paradigm shifts in how models execute on hardware:

1. **Symbolic to eager to compiled execution.** The field moved from Theano-style static graphs (2010-2016) to PyTorch-style eager execution (2016-2022) to `torch.compile` and its kin (2022-present). Each transition changed the fundamental tradeoff between developer velocity and runtime performance.

2. **Hand-tuned to compiler-generated kernels.** Writing CUDA by hand is giving way to Triton programs and compiler-generated kernels. The question is whether compilers can close the gap with expert-written code -- and the answer is "mostly yes, with important exceptions."

3. **FLOP reduction to IO-awareness.** For decades, optimizing a computation meant reducing the number of arithmetic operations. FlashAttention showed that the real bottleneck is often memory bandwidth, not compute. This reframing unlocked order-of-magnitude improvements in attention and inspired a generation of IO-aware algorithms.

4. **Fixed to adaptive quantization.** Early quantization applied the same precision uniformly. Modern approaches like AWQ and SmoothQuant recognize that different weights and activations have wildly different sensitivity to precision loss and adapt accordingly.

5. **FP32 to FP16 to FP8 to FP4.** Each halving of precision roughly doubles throughput and halves memory. DeepSeek-V3 trained 671 billion parameters in FP8 with negligible accuracy degradation compared to BF16. NVIDIA Blackwell introduces native FP4. The precision floor is dropping fast.

6. **Monolithic to disaggregated serving.** Production inference systems like vLLM and SGLang now treat the KV cache as a first-class resource, separate prefill from decode, and manage GPU memory with virtual-memory-style paging. The serving stack has become as sophisticated as the model itself.

This post works through the full optimization stack, from compilers to kernels to quantization to serving. The goal is not a surface-level tour but a deep technical treatment of each layer -- how it works, why it works, and when to use it. We will write code, derive key equations, and look at real production numbers. By the end, you should have a working mental model for reasoning about model performance and a practical toolkit for making your own models run faster.

Let us start by building that mental model.

## The Optimization Stack: A Mental Model

Before diving into any specific technique, you need a framework for reasoning about *why* a given model is slow and *which* optimization will help. Without this framework, optimization becomes guesswork -- you apply `torch.compile`, hope for a speedup, and are mystified when nothing changes (or things get slower). The right mental model saves you from this.

### The Roofline Model

The most useful framework for thinking about GPU performance is the **Roofline Model**, introduced by Williams, Waterman, and Patterson in 2009. It plots the attainable performance of a computation (in GFLOPS) against its **operational intensity** -- the ratio of floating-point operations to bytes of memory traffic (FLOPs/byte).

The core equation is:

$$\text{Attainable GFLOPS} = \min\left(\text{Peak GFLOPS},\; \text{Operational Intensity} \times \text{Peak Bandwidth}\right)$$

The intuition here is straightforward. Every GPU operation needs two things: it needs to *move data* from memory to compute units, and it needs to *perform arithmetic* on that data. Which of these two is the bottleneck depends on the ratio between them.

If an operation does very little arithmetic per byte of data it moves (low operational intensity), it will be **memory-bound** -- the compute units will be sitting idle, waiting for data. If an operation does a lot of arithmetic per byte (high operational intensity), it will be **compute-bound** -- the memory system can keep up, and you are limited by how fast the GPU can crunch numbers.

These two regimes form the "roofline" shape: a sloped line on the left (memory-bound, where performance scales linearly with operational intensity) that hits a flat ceiling on the right (compute-bound, where performance is capped at peak GFLOPS).

> The single most important insight from the Roofline Model for deep learning practitioners: **most operations in a neural network are memory-bound, not compute-bound.** Only matrix multiplications (and convolutions, which are implemented as matrix multiplications) have high enough operational intensity to be compute-bound. Everything else -- elementwise operations, layer normalization, softmax, activation functions -- is bottlenecked by memory bandwidth.

Let us put numbers to this. An NVIDIA H100 SXM has approximately 3,958 TFLOPS of FP8 peak compute (or about 1,979 TFLOPS in BF16/FP16 with tensor cores) and approximately 3.35 TB/s of HBM3 bandwidth. The **ridge point** -- the operational intensity where you transition from memory-bound to compute-bound -- is roughly:

$$\text{Ridge Point} = \frac{\text{Peak GFLOPS}}{\text{Peak Bandwidth}} = \frac{1{,}979{,}000}{3{,}350} \approx 591 \; \text{FLOPs/byte (BF16)}$$

Now consider common operations:

- **Elementwise add**: 1 FLOP per 2 bytes read + 1 byte written. Operational intensity $\approx$ 0.33 FLOPs/byte. Massively memory-bound.
- **Layer normalization**: ~5 FLOPs per element, reads and writes each element. Operational intensity $\approx$ 1-2 FLOPs/byte. Deeply memory-bound.
- **Softmax**: ~5 FLOPs per element (exp, sum, divide), two passes over data. Operational intensity $\approx$ 1-2 FLOPs/byte. Deeply memory-bound.
- **Matrix multiplication** ($M \times K$ by $K \times N$): $2MKN$ FLOPs, reads $2(MK + KN)$ bytes, writes $2MN$ bytes (in FP16). For large enough matrices, operational intensity scales linearly with matrix dimensions. A $4096 \times 4096$ matmul has operational intensity $\approx$ 1365 FLOPs/byte. Solidly compute-bound.

This explains a critical phenomenon that trips up many practitioners: **reducing the number of FLOPs does not necessarily make your model faster.** If you are optimizing a memory-bound operation, reducing FLOPs does nothing -- you are limited by how fast you can shuttle data between HBM and the compute units. The correct optimization for memory-bound operations is to reduce *memory traffic*, typically by fusing multiple operations into a single kernel so intermediate results stay in fast on-chip SRAM instead of being written to and read back from slow HBM.

This is exactly what FlashAttention does. Standard attention computes $QK^T$, writes the $N \times N$ attention matrix to HBM, reads it back for softmax, writes the softmax output to HBM, reads it back for the final matrix multiply with $V$. The arithmetic operations are the same (actually slightly more in FlashAttention, due to recomputation), but the memory traffic is dramatically reduced by keeping intermediates in SRAM. More FLOPs, less time. The Roofline Model explains why.

### Where Each Optimization Fits

With the Roofline Model as our foundation, we can map the full optimization stack to the bottlenecks each technique addresses:

**Layer 1: Compilers** (`torch.compile`, XLA, TVM)

Compilers primarily help memory-bound operations. Their key optimization is **operator fusion**: combining multiple elementwise ops, normalizations, and activations into a single kernel. Instead of launching separate kernels for add, multiply, GELU, and layer norm -- each of which reads from and writes to HBM -- the compiler fuses them into one kernel that does everything in a single pass over the data. The secondary benefit is eliminating **kernel launch overhead** (5-15 microseconds per kernel), which matters in latency-sensitive inference where thousands of tiny kernels are launched per token.

**Layer 2: Kernels** (Triton, FlashAttention, cuDNN, ThunderKittens)

Hand-optimized or semi-automated kernels push closer to hardware limits than general-purpose compilers can. FlashAttention is the canonical example: it exploits the memory hierarchy (SRAM vs. HBM) in a way that no compiler currently generates automatically. Triton sits between compilers and hand-written CUDA -- it gives you control over tiling and memory access patterns while abstracting away thread-level details. cuDNN provides NVIDIA's hand-tuned implementations of standard operations.

**Layer 3: Precision** (FP8, INT4, mixed precision)

Reducing precision directly improves both the memory-bound and compute-bound regimes. In the memory-bound regime, halving precision halves the bytes that need to move -- you can transfer twice as many values per second at the same bandwidth. In the compute-bound regime, tensor cores operate at higher throughput at lower precision (the H100 does 2x TFLOPS in FP8 vs. FP16). This is why mixed precision training -- using FP16/BF16 for computation and FP32 for accumulation -- is essentially free performance. The key challenge is maintaining model quality as precision drops.

**Layer 4: Memory** (FlashAttention, PagedAttention, activation checkpointing)

Memory optimizations enable larger models, longer sequences, and higher throughput by reducing memory footprint. FlashAttention eliminates the $O(N^2)$ memory for the attention matrix. PagedAttention (used in vLLM) eliminates KV cache fragmentation. Activation checkpointing trades compute for memory by recomputing activations during the backward pass instead of storing them.

**Layer 5: Serving** (vLLM, SGLang, TensorRT-LLM)

Serving systems optimize at the system level -- batching requests intelligently, scheduling prefill and decode phases, managing the KV cache, and overlapping computation with communication. These optimizations are orthogonal to model-level optimizations and compose with them. vLLM's PagedAttention, SGLang's RadixAttention for prefix caching, and continuous batching are examples of system-level optimizations that have nothing to do with the model architecture but can deliver 2-5x throughput improvements.

The key insight is that **these layers compose**. You can use `torch.compile` to fuse your elementwise operations, FlashAttention for your attention layers, FP8 precision for your matmuls, and vLLM for serving -- all simultaneously. Understanding which layer addresses which bottleneck prevents you from applying the wrong optimization. If your model is memory-bound in attention, better matmul precision will not help. If your serving system is the bottleneck, rewriting kernels is wasted effort.

With this mental model in place, let us start from the top of the stack and work our way down. The first layer -- compilation -- is where most practitioners should begin, because it provides the highest ratio of speedup to engineering effort.

## Compilation: From Python to Fast Code

Python is a terrible language for high-performance computing. It is interpreted, dynamically typed, and runs orders of magnitude slower than compiled languages. PyTorch runs fast despite Python because every operation dispatches to hand-written C++/CUDA kernels under the hood. But this approach leaves enormous performance on the table: the Python interpreter still controls the *sequencing* of operations, each kernel is launched independently with its own overhead, and there is no cross-operation optimization.

**`torch.compile`**, introduced in PyTorch 2.0 and detailed in Ansel et al. (ASPLOS 2024), is PyTorch's answer to this problem. It is a compiler that takes your regular PyTorch code, traces it, optimizes it, and generates fast machine code -- all without requiring you to change your model definition. On a benchmark suite of 180+ real models from TorchBench, Hugging Face, and TIMM, `torch.compile` achieves significant speedups -- the PyTorch 2 paper reports a 1.41x geometric mean training speedup, with larger gains on inference-only workloads depending on model architecture and compilation mode. For many practitioners, it is the single highest-impact optimization available.

### How torch.compile Works

The `torch.compile` pipeline has three major stages, each handled by a distinct subsystem:

**Stage 1: TorchDynamo -- Capturing the computation graph.** TorchDynamo intercepts Python bytecode execution at the frame level. It walks through your Python code and records the PyTorch operations it encounters, building an **FX graph** -- an intermediate representation of the computation. The key insight is that TorchDynamo does not require you to rewrite your code or use a tracing-friendly subset of Python. It handles standard Python control flow, closures, and most Python builtins by simply recording their effects. When it encounters something it cannot trace -- a C extension, data-dependent control flow, or an unsupported Python feature -- it inserts a **graph break**, splitting the computation into multiple subgraphs and falling back to eager execution for the untraceable portion.

**Stage 2: AOT Autograd -- Capturing the backward pass.** For training workloads, AOT Autograd traces the backward pass ahead of time, producing a combined forward-backward graph that can be jointly optimized. This enables optimizations like fusing backward operations that span multiple forward operations.

**Stage 3: TorchInductor -- Code generation.** TorchInductor takes the FX graph and generates actual executable code. For GPU operations, it generates **Triton** kernels -- block-level GPU programs that get compiled to PTX and then to SASS (GPU machine code). For CPU operations, it generates C++ code using OpenMP for parallelism. TorchInductor's key optimization is operator fusion: it analyzes the dataflow graph and identifies chains of elementwise operations, reductions, and pointwise computations that can be fused into a single kernel.

The result is that a model written in idiomatic PyTorch -- with `nn.Module`, standard layer norms, GELU activations, and all the usual abstractions -- gets compiled into a small number of highly optimized kernels that minimize memory traffic and kernel launch overhead.

### Using torch.compile in Practice

The API is deliberately simple:

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x

model = TransformerBlock().cuda().eval()
x = torch.randn(32, 128, 768, device="cuda")

# One-line compilation
compiled_model = torch.compile(model)
y = compiled_model(x)  # First call triggers compilation

# Mode comparison
compiled_fast = torch.compile(model, mode="reduce-overhead")  # Uses CUDA graphs
compiled_tuned = torch.compile(model, mode="max-autotune")    # Triton autotuning
```

Three compilation modes provide different tradeoffs:

- **`default`**: Balanced compilation. Applies fusion and generates Triton kernels but does not spend much time searching for optimal kernel configurations. Good compilation speed, solid runtime performance. This is what most people should start with.

- **`reduce-overhead`**: Wraps the compiled computation in **CUDA graphs**, which eliminate per-kernel CPU launch overhead by capturing the entire sequence of GPU operations and replaying it as a single unit. This mode shines in latency-sensitive inference, especially autoregressive decoding where per-token latency matters. The tradeoff: CUDA graphs require fixed tensor shapes, so dynamic shapes trigger recompilation or fallback.

- **`max-autotune`**: Enables Triton's autotuner, which searches over multiple kernel configurations (tile sizes, number of warps, pipeline stages) and benchmarks each to find the fastest. This can squeeze out another 10-30% performance beyond `default` but at the cost of significantly longer compilation times -- minutes rather than seconds. Worth it for models that will be deployed at scale.

### Graph Breaks: The Production Concern

In theory, `torch.compile` traces your entire forward pass into one unified graph. In practice, it frequently hits operations it cannot trace, resulting in **graph breaks** that fragment the graph into multiple subgraphs with eager-mode Python execution between them. Each graph break eliminates opportunities for cross-operation fusion and adds CPU-GPU synchronization overhead.

Common causes of graph breaks:

- **C extensions and custom CUDA kernels**: If your model calls into a C library or uses a custom CUDA kernel that TorchDynamo cannot trace, it breaks the graph.
- **Data-dependent control flow**: `if x.item() > 0:` requires materializing a GPU tensor value on the CPU, which TorchDynamo cannot handle during tracing.
- **Unsupported Python features**: Certain built-in functions, generator expressions, and some standard library calls are not traceable.
- **Print statements and logging**: `print(x.shape)` in the forward pass will cause a graph break.

Here is the particularly sharp edge: **after 8 recompiles per frame, `torch.compile` permanently falls back to eager mode for that frame.** This means that models with dynamic shapes -- common in NLP with variable sequence lengths -- can silently lose all compilation benefits after the first few batches. If you are seeing no speedup from `torch.compile` in production, graph breaks and recompilation limits are the first things to check.

> The practical lesson: treat `torch.compile` as a profiling opportunity, not just a performance knob. Use `torch._dynamo.explain(model, *inputs)` to identify graph breaks, and systematically eliminate them. A model with zero graph breaks can see 3-4x speedup; a model with 20 graph breaks might see no speedup at all.

### AOTInductor: Ahead-of-Time Compilation for Production

JIT compilation has a fundamental tension with production deployment: the first call to a compiled model triggers compilation that can take 20+ seconds (or minutes with `max-autotune`). For a serving system processing requests, this cold-start latency is unacceptable. You also may not want a Python runtime in production at all -- it is a large dependency surface, a security concern, and it constrains deployment to environments with Python installed.

**AOTInductor** solves both problems. It combines `torch.export` (which captures a clean, complete graph with no Python dependencies) with TorchInductor's code generation to produce a self-contained shared library that can be loaded and executed without Python, without JIT compilation, and with zero warm-up overhead. This is the standard approach at Meta for production model inference.

The pipeline:

```python
# AOT compilation pipeline
exported = torch.export.export(model, (example_input,))
package_path = torch._inductor.aoti_compile_and_package(
    exported, package_path="model.pt2"
)
# Deploy: zero compilation overhead
compiled = torch._inductor.aoti_load_package(package_path)
output = compiled(example_input)
```

`torch.export` is stricter than `torch.compile` -- it requires the entire model to be traceable with no graph breaks. This means you need to resolve all the graph-break issues that `torch.compile` would have papered over with fallback to eager. In practice, this is a feature, not a bug: it forces you to clean up your model code and ensures the exported graph is complete and optimizable.

The exported `.pt2` package contains generated Triton and C++ code, precompiled for the target GPU architecture. Loading it is fast (milliseconds), and execution has no Python overhead. For teams deploying models in C++ inference servers, this is the path to production.

As of PyTorch 2.9-2.10, AOTInductor is in beta but is already used in production at Meta. The API may evolve, but the architecture -- export, compile ahead of time, deploy without Python -- is stable.

### CUDA Graphs: Eliminating Launch Overhead

Every GPU kernel launch involves CPU work: the CUDA driver validates arguments, sets up the kernel configuration, and submits the work to the GPU's command queue. This takes approximately 5-15 microseconds per kernel. For training on large batches with big matmuls, this overhead is negligible -- the GPU computation takes milliseconds. But for autoregressive decoding, where each token generation involves hundreds of small kernels executing in microseconds, launch overhead can dominate total latency.

**CUDA Graphs** address this by recording a sequence of GPU operations and replaying them as a single unit. Instead of launching 200 kernels individually, the GPU replays the entire captured graph with a single launch command.

```python
# Manual CUDA graph capture
g = torch.cuda.CUDAGraph()
static_input = torch.randn(1, 128, 768, device="cuda")

# Warmup
for _ in range(3):
    _ = model(static_input)

# Capture
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (near-zero CPU overhead)
new_input = torch.randn(1, 128, 768, device="cuda")
static_input.copy_(new_input)
g.replay()  # Result is in static_output
```

Notice the critical pattern: CUDA graphs use **static memory addresses**. You do not pass new tensors to the graph -- you copy new data into the same memory buffers that were used during capture, then replay. This means CUDA graphs fundamentally cannot handle dynamic shapes. If your batch size or sequence length changes, you need a separate graph for each shape (or, more commonly, you pad to a fixed set of shapes).

The performance impact can be dramatic. Fireworks AI reported a 2.3x speedup on LLaMA-v2-7B inference from CUDA graphs alone. The `reduce-overhead` mode in `torch.compile` applies CUDA graphs automatically, which is the recommended way to use them -- manual graph management is error-prone and fragile.

> CUDA graphs are the rare optimization that is both simple and high-impact for inference. If you are serving an autoregressive LLM and not using CUDA graphs (either manually or via `torch.compile(mode="reduce-overhead")`), you are leaving 1.5-2.5x performance on the table for decode-heavy workloads.

### The Compiler-First vs. Kernel-First Debate

There is a philosophical divide in the ML systems community that is worth understanding. The **compiler-first** approach, exemplified by Google's XLA and JAX, says: write your model in a high-level language, and let the compiler figure out how to map it to hardware. The compiler handles fusion, parallelism, memory layout, and device placement. When new hardware arrives, you update the compiler, and all existing models benefit automatically.

The **kernel-first** approach, exemplified by the CUDA ecosystem, says: for performance-critical operations, write (or generate) specialized kernels that exploit specific hardware features. FlashAttention is the poster child -- no compiler in existence generates anything close to FlashAttention's performance from a naive attention implementation. The kernel-first approach achieves higher peak performance but requires expert effort for each new operation and hardware generation.

`torch.compile` sits in an interesting middle ground. It uses compiler techniques (TorchDynamo tracing, TorchInductor fusion) for the operations where compilers work well (elementwise chains, simple reductions) while preserving the ability to call hand-written kernels (FlashAttention, cuDNN matmuls) where compilers fall short. This pragmatic approach is a big part of why `torch.compile` has seen faster adoption than XLA in the PyTorch ecosystem.

But the tension is real. Edward Yang, one of PyTorch's core maintainers, wrote in August 2025 that `torch.compile` faces "stiff competition from JAX, which is years ahead in compile-driven parallelism." JAX's XLA compiler can automatically shard a model across thousands of TPUs with minimal user annotation. `torch.compile` is only beginning to handle distributed collectives, and does not optimize them by default. For single-device inference, `torch.compile` is arguably the best option available. For large-scale distributed training, the compiler-first approach may have fundamental advantages that `torch.compile`'s incremental design cannot easily replicate.

For the practitioner, the takeaway is: use `torch.compile` for single-device inference and single-node training, where it delivers consistent speedups with minimal effort. Be aware that for multi-node distributed training, JAX/XLA or manual optimization may still be necessary. And watch this space -- the convergence of compiler and kernel approaches is one of the most active areas of ML systems research.

### From Compilation to Kernels

Compilation gets you a long way. Operator fusion eliminates redundant memory traffic. CUDA graphs eliminate launch overhead. AOTInductor eliminates cold-start latency. Together, these can deliver 2-3x speedups with minimal code changes.

But compilation has limits. A compiler can fuse a chain of elementwise operations, but it cannot invent FlashAttention. It cannot reason about the GPU memory hierarchy at the level of SRAM tiles and warp-level primitives. It cannot discover that recomputing activations in the backward pass is cheaper than storing them, or that online softmax enables block-wise attention computation.

The single most impactful kernel-level optimization in the history of deep learning is FlashAttention -- an algorithm that required rethinking attention computation from the memory hierarchy up, recognizing that the bottleneck was not compute but IO, and engineering a tiling scheme that keeps all intermediates in fast on-chip SRAM. No compiler generated it. No compiler currently can. And it is responsible for enabling the context length explosion from 2-4K tokens to 1M+ that has defined the LLM era.

That is where we turn next.

## FlashAttention: The IO-Awareness Revolution

Standard self-attention is elegant on paper and brutal on hardware. Given query, key, and value matrices $Q, K, V \in \mathbb{R}^{N \times d}$, the textbook implementation computes:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

That intermediate $QK^T$ product is an $N \times N$ matrix. For a sequence length of $N = 8192$ in FP16, that is $8192^2 \times 2$ bytes = 128 MB -- per attention head, per layer. A model with 32 heads and 40 layers materializes over 160 GB of intermediate attention matrices during a single forward pass. This matrix does not contain learned parameters. It is a transient computation artifact that gets written to GPU high-bandwidth memory (HBM), read back for the softmax, written again, read again for the multiplication with $V$, and then discarded. The arithmetic intensity is low: you are doing $O(N^2)$ memory accesses for $O(N^2 d)$ FLOPs, which means the operation is **memory-bound** on modern GPUs where compute throughput has scaled far faster than memory bandwidth.

This is the bottleneck that FlashAttention eliminates.

### Online Softmax: The Key Prerequisite

Before we can tile attention, we need to solve a subtler problem: softmax is inherently a two-pass operation. The standard computation requires:

1. A first pass to find $\max(x_i)$ for numerical stability
2. A second pass to compute $\sum_i e^{x_i - \max}$ and normalize

This means you need the entire row of the attention matrix before you can produce any output -- which seems to force materializing the full $N \times N$ matrix. **Online softmax** (Milakov & Gimelshein, 2018) breaks this dependency with a single-pass algorithm that maintains running statistics:

$$m_{i+1} = \max(m_i, x_{i+1})$$

$$d_{i+1} = d_i \cdot e^{m_i - m_{i+1}} + e^{x_{i+1} - m_{i+1}}$$

Here $m_i$ is the running maximum across all elements seen so far, $d_i$ is the running denominator (the sum of exponentials, corrected for shifts in the maximum), and $x_{i+1}$ is the next element in the sequence. The intuition is straightforward: every time you encounter a new element that changes the maximum, you retroactively correct all the previously accumulated exponentials by multiplying by $e^{m_{\text{old}} - m_{\text{new}}}$. This correction factor is always $\leq 1$, so it is numerically stable. The beauty is that you never need to go back and re-read previous elements -- the correction is purely multiplicative on the running sum.

This single-pass property is what makes tiled attention possible. Without it, FlashAttention could not exist.

### FlashAttention: IO-Aware Tiled Attention

**FlashAttention** (Dao et al., NeurIPS 2022) applies online softmax to compute exact attention -- not an approximation -- while never materializing the full $N \times N$ matrix. The core idea is to partition $Q$, $K$, and $V$ into blocks that fit in GPU SRAM (the fast on-chip memory, typically 20-200 KB per streaming multiprocessor on A100), and to compute attention block-by-block, accumulating results using the online softmax correction.

The algorithm proceeds as follows: for each block of queries $Q_i$, iterate over all blocks of keys $K_j$ and values $V_j$. For each $(Q_i, K_j)$ pair, compute the local attention scores $S_{ij} = Q_i K_j^T / \sqrt{d}$ in SRAM. Update the running softmax statistics (max and denominator) using the online softmax equations. Accumulate the weighted sum $O_i \mathrel{+}= \text{softmax}(S_{ij}) \cdot V_j$ with the appropriate correction factors. At no point does the full $N \times N$ matrix exist in HBM.

The IO complexity tells the story. Standard attention performs $\Theta(Nd + N^2)$ HBM accesses -- reading and writing the full attention matrix dominates. FlashAttention reduces this to:

$$\Theta\left(\frac{N^2 d^2}{M}\right)$$

where $M$ is the SRAM size. For typical values ($N = 2048$, $d = 64$, $M = 100$ KB), this is roughly a 10x reduction in memory traffic. Since attention is memory-bound, fewer HBM accesses translates directly to faster wall-clock time -- even though the total FLOP count is unchanged.

> Key insight: FlashAttention showed that the real bottleneck in attention is memory bandwidth, not arithmetic. Reducing HBM accesses matters more than reducing FLOPs.

The **backward pass** introduces a clever trade-off: rather than storing the $N \times N$ attention matrix for the backward pass (which would defeat the entire purpose), FlashAttention recomputes it from $Q$, $K$, $V$. This trades extra FLOPs for memory savings. The trade is favorable precisely because attention is memory-bound -- the extra compute fits within the time the GPU would otherwise spend waiting on memory transfers. You are filling idle arithmetic units with useful work. The recomputation strategy reduces peak memory from $O(N^2)$ to $O(N)$, enabling dramatically longer sequences.

### FlashAttention-2: Closing the Hardware Gap

The original FlashAttention achieved 25-40% of A100 theoretical throughput -- good, but leaving substantial performance on the table. **FlashAttention-2** (Dao, 2023) closed the gap with two architectural changes.

First, it parallelized over the sequence length dimension rather than only over batch and head dimensions. The original algorithm's outer loop over key/value blocks serialized work within each thread block. FlashAttention-2 restructures the loop ordering so that different thread blocks process different query blocks, enabling better occupancy on GPUs with many SMs.

Second, it improved **warp partitioning** -- the division of work among the 32-thread execution units (warps) within a thread block. FlashAttention-1 split work across warps in a way that required inter-warp communication via shared memory for the softmax reduction. FlashAttention-2 assigns each warp a different subset of the key/value blocks and accumulates independently, eliminating the shared memory synchronization bottleneck.

The result: 50-73% A100 utilization, roughly a 2x speedup over FlashAttention-1, and approaching the hardware roofline for a memory-bound operation.

### FlashAttention-3: Exploiting Hopper

**FlashAttention-3** (Shah, Dao et al., 2024) is purpose-built for NVIDIA's Hopper architecture (H100), exploiting three hardware features that did not exist on Ampere.

**Asynchronous warp-specialized execution** separates producer warps (which issue TMA loads from HBM to shared memory) from consumer warps (which execute Tensor Core matmuls). On Ampere, these operations were serialized: load a tile, compute on it, load the next. On Hopper, the TMA engine operates independently of the SMs, so data movement and computation overlap in a true pipeline. FlashAttention-3 structures its tile loop to maintain three pipeline stages, keeping both the TMA and Tensor Cores saturated.

**FP8 block quantization** takes advantage of Hopper's native FP8 Tensor Cores. FlashAttention-3 quantizes $Q$ and $K$ tiles to FP8 on the fly, computes the attention scores in FP8, and accumulates in FP32 -- combining the memory bandwidth savings of FP8 with the accuracy of higher-precision accumulation. This is a preview of the precision techniques we will cover next.

**Hardware-accelerated softmax** uses Hopper's warp-level reduction instructions to compute the softmax normalization with fewer register moves.

The headline numbers: up to 840 TFLOPS/s in BF16, reaching approximately 85% H100 utilization -- approaching the hardware ceiling for this operation.

### FlexAttention: Making It Programmable

Hand-writing CUDA kernels for every attention variant is unsustainable. **FlexAttention** (Dong et al., Aug 2024) is a PyTorch-native solution that generates fused, FlashAttention-style Triton kernels automatically via `torch.compile`.

The API is simple: you define a `score_mod` function that transforms the attention score for each $(query\_idx, key\_idx)$ pair, and FlexAttention fuses it into the tiled attention kernel:

```python
from torch.nn.attention.flex_attention import flex_attention

def causal_mask(score, b, h, q_idx, k_idx):
    return torch.where(q_idx >= k_idx, score, float("-inf"))

output = flex_attention(query, key, value, score_mod=causal_mask)
```

This single interface supports causal masks, sliding window attention, ALiBi position biases, document masking for packed sequences, and arbitrary custom patterns -- all without hand-written CUDA kernels. Performance is within 90% of hand-tuned FlashAttention for standard patterns and enables patterns (like document masking) that previously had no efficient fused implementation.

### Impact

The impact of this line of work on the field is difficult to overstate. Context lengths went from 2-4K tokens (GPT-3, 2020) to 8K (early GPT-4, 2023) to 128K (GPT-4 Turbo, late 2023) to 1M+ tokens (Gemini 1.5, Llama 3, 2024). FlashAttention made this tractable. Without it, the quadratic memory cost of attention would have forced either aggressive context truncation or lossy approximations. Instead, we got exact attention that is faster *and* more memory efficient -- a rare case where algorithmic insight improved both axes simultaneously.

FlashAttention-3's introduction of FP8 support within the attention kernel points to a broader trend: precision reduction is becoming intertwined with every layer of the optimization stack. This brings us to the question of how far we can push the precision frontier down -- from the FP32 that deep learning was built on, through FP16 and FP8, and now toward FP4.

## Precision: From FP32 to FP4

Every floating-point number is a trade-off. More bits give you a wider representable range and finer granularity between adjacent values, but they cost proportionally more memory bandwidth and storage. Since modern AI workloads are memory-bandwidth-bound, halving the precision of a matrix multiply roughly doubles its throughput -- not because the arithmetic is faster, but because you can feed the Tensor Cores twice as fast from the same memory bus.

The history of deep learning precision is a history of discovering just how few bits you actually need.

### Mixed Precision Training: The Foundation

**Mixed precision training** (Micikevicius et al., ICLR 2018) established the methodology that every subsequent precision reduction has followed. The core insight is that you do not need uniform precision across all operations -- some quantities tolerate low precision while others do not. The paper introduced three interlocking techniques:

**FP32 master weights.** Weights are stored in FP32 and cast to FP16 for the forward and backward passes. The gradient updates, which can be very small relative to the weight magnitudes, are accumulated into the FP32 copy. Without this, many updates would be lost to the limited FP16 resolution (minimum subnormal: $\sim 6 \times 10^{-8}$).

**Loss scaling.** Small gradients -- particularly in early layers of deep networks -- can underflow FP16's representable range. Loss scaling shifts them into range:

$$\hat{L} = L \cdot S$$

where $S$ is a scale factor (typically starting at $2^{16}$ or higher). Gradients are computed in FP16 using the scaled loss, then unscaled before the weight update:

$$\nabla W = \frac{1}{S} \nabla_{\hat{L}} W$$

The weight update then proceeds normally in FP32:

$$W_{\text{FP32}} \leftarrow W_{\text{FP32}} - \eta \cdot \nabla W_{\text{FP32}}$$

The scale factor $S$ exists to solve a specific problem: FP16 can represent values down to about $6 \times 10^{-8}$, but many gradient values in practice are smaller than this. Multiplying the loss by a large constant shifts the entire gradient distribution up, keeping small-but-meaningful gradients from underflowing to zero. Dynamic loss scaling adjusts $S$ automatically -- increasing it when no overflow is detected, decreasing it when infs or NaNs appear.

**FP32 accumulation.** Matrix multiplies use FP16 inputs but accumulate partial sums in FP32. This prevents the catastrophic rounding errors that would occur from summing thousands of FP16 products.

In PyTorch, this is the **automatic mixed precision** (AMP) API:

```python
import torch
from torch.amp import autocast, GradScaler

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    with autocast(device_type="cuda"):  # FP16 forward pass
        output = model(data.cuda())
        loss = loss_fn(output, target.cuda())
    scaler.scale(loss).backward()  # Scaled FP16 gradients
    scaler.step(optimizer)         # Unscale, check for infs, step
    scaler.update()                # Adjust scale factor
```

The `autocast` context manager selectively casts operations to FP16 (matmuls, convolutions) while keeping precision-sensitive operations (softmax, layer norm, loss functions) in FP32. `GradScaler` implements dynamic loss scaling -- it starts with a high scale factor, and if it detects overflow (infs in the gradients), it skips the optimizer step and reduces the scale for the next iteration.

Mixed precision delivered nearly 2x speedup on Volta and later architectures with no meaningful accuracy loss. It is now the universal default -- there is essentially no reason to train in pure FP32 on modern hardware.

### FP8: Halving Precision Again

**FP8 training** (Micikevicius et al., 2022) followed the same playbook -- mixed precision with format-specific roles -- but introduced a new wrinkle: two distinct FP8 formats optimized for different phases of training.

**E4M3** (4 exponent bits, 3 mantissa bits) offers more precision within a narrower range of $[-448, 448]$. It is used for forward pass activations and weights, where values are typically well-bounded. **E5M2** (5 exponent bits, 2 mantissa bits) sacrifices precision for a wider dynamic range of $[-57344, 57344]$. It is used for the backward pass, where gradient magnitudes vary across many orders of magnitude.

The challenge with FP8 is that a single scale factor per tensor is often insufficient. With only 3-4 mantissa bits, the quantization grid is coarse, and outlier values can force the scale to a point where most of the tensor's values collapse into a few quantization bins. The solution is **fine-grained quantization** -- applying separate scale factors to subregions of each tensor.

**DeepSeek-V3** (December 2024) demonstrated this at production scale. It was the first 671B-parameter model trained end-to-end in FP8. The details of their approach are instructive:

- **128x128 submatrix scaling for weights**: each 128x128 block of a weight matrix gets its own FP8 scale factor, stored in E8M0 format (8 exponent bits, no mantissa -- effectively a power of two). This ensures that even if one region of the weight matrix has very different magnitudes than another, each region is quantized accurately within its own dynamic range.
- **1x128 activation scaling**: activations, which change every forward pass, use per-row-block scaling with blocks of 128 elements. This is finer-grained than weight scaling because activation distributions can vary significantly across the batch and sequence dimensions.
- **Blockwise FP32 accumulation**: the FP8 matmul accumulates partial sums in FP32, but with an added twist -- accumulation is reset and corrected at regular intervals within the matmul to prevent the FP32 partial sums from growing large enough to swamp the small contributions from later FP8 products.

The result: negligible accuracy degradation compared to BF16 training, at substantially higher throughput.

**DeepL** provided another production data point, training large translation models on 544 H100 GPUs with E4M3 forward / E5M2 backward. Over 15 months of optimization, their model FLOP utilization (MFU) improved from 44.6% to 67% to 80%, with FP8 delivering a 50% training acceleration versus BF16 baselines. These are not research experiments -- they are production training runs where accuracy regressions would translate directly to revenue impact.

### FP4: The Next Frontier

**FP4 training** on NVIDIA Blackwell GPUs (B200) represents the next step. Blackwell natively supports **NVFP4**, providing 20 PFLOPS of FP4 Tensor Core throughput versus 9 PFLOPS for FP8 -- a further 2.2x compute increase within the same power envelope.

Early results are encouraging: a 12B-parameter model trained on 10 trillion tokens in FP4 produced loss curves that matched FP8 baselines. But I would treat the large-scale picture with caution. FP4 has only 1-2 mantissa bits depending on the format -- the quantization grid is extraordinarily coarse. Whether fine-grained scaling can compensate at 100B+ scale, across the full diversity of architectures and training regimes, remains an open question. My intuition is that FP4 training will require more aggressive per-block scaling and likely architecture-specific tuning that does not generalize as cleanly as mixed precision FP16 did.

### OCP Microscaling: An Industry Standard

The proliferation of low-precision formats created a fragmentation risk -- every vendor inventing its own FP8/FP6/FP4 format, with different rounding modes, scaling conventions, and hardware support. The **Open Compute Project Microscaling (MX) formats** address this through an industry-wide standard developed by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm.

The MX family includes **MXFP8**, **MXFP6**, and **MXFP4**, all sharing a common structure: groups of 32 elements share a single E8M0 scale factor (a power-of-two scale stored as an 8-bit exponent). Within each group, individual elements are stored in the specified low-precision format. This group-level scaling is coarser than DeepSeek-V3's approach but has the advantage of being simple enough to implement efficiently across diverse hardware, from datacenter GPUs to mobile accelerators.

> The precision frontier keeps pushing lower: FP32 (2015) to FP16/BF16 (2018) to FP8 (2022) to FP4 (2025). Each step requires new hardware (Tensor Cores), new training methodology (loss scaling, blockwise accumulation), and careful validation.

Lower precision during training is one story -- you control the training loop, you can adjust loss scaling, you can monitor for divergence. But what about compressing a model that has already been trained? That is a different problem with different constraints, and it is where quantization comes in.

## Quantization: Compressing Trained Models

The precision techniques in the previous section operate *during training* -- you choose a precision format and train the model within its constraints. **Post-training quantization (PTQ)** addresses a different scenario: you have a model that was trained in FP16 or BF16, and you want to compress it for deployment without retraining. This is the common case for practitioners who download a pretrained model from a hub and need to serve it on limited hardware.

The distinction matters because training gives the model a chance to adapt its weight distributions to the quantization grid. Post-training quantization does not -- it must map an existing, fixed set of weights into fewer bits without catastrophic accuracy loss. This is fundamentally harder, and the techniques are correspondingly more sophisticated.

### The Basics: Affine Quantization

The standard affine quantization scheme (Jacob et al., 2018) maps a floating-point value $r$ to a $b$-bit integer $q$:

$$q = \text{round}\left(\text{clamp}\left(\frac{r}{S} + Z,\; 0,\; 2^b - 1\right)\right)$$

where $S$ is the scale factor, $Z$ is the zero-point (an integer offset that maps to $r = 0$), and $b$ is the target bit-width. Dequantization recovers an approximation of the original value:

$$r \approx S(q - Z)$$

The scale $S$ is typically computed from the observed range of the tensor: $S = (\max(r) - \min(r)) / (2^b - 1)$. Symmetric quantization simplifies by setting $Z = 0$ and using $S = \max(|r|) / (2^{b-1} - 1)$, which maps zero exactly but wastes one bit if the distribution is asymmetric.

For INT8 (8-bit integer), this is straightforward -- most weight and activation distributions are smooth enough that linear mapping to 256 levels introduces negligible error. The challenge emerges at INT4 (16 levels) and below, where the quantization grid is so coarse that naive methods produce significant accuracy degradation. The methods that follow are, at their core, clever ways to use those 16 levels more effectively.

### GPTQ: One-Shot Weight Quantization

**GPTQ** (Frantar et al., ICLR 2023) tackles the problem of quantizing very large language models -- 175B parameters -- to 3-4 bits in a single pass, without any retraining or gradient computation. It is based on the **Optimal Brain Surgeon** (OBS) framework, which uses second-order information (the Hessian of the loss with respect to the weights) to decide how to round each weight while compensating for the rounding error in the remaining unquantized weights.

The intuition is this: when you round weight $w_i$ to its nearest quantized value, you introduce a small error. Rather than absorbing this error into the final output, you can distribute compensating adjustments across the weights you have not yet quantized. The optimal adjustment is determined by the inverse Hessian $H^{-1}$, which tells you how sensitive the layer's output is to perturbations in each weight. Weights in directions where the loss surface is highly curved (large Hessian eigenvalues) get smaller compensating adjustments; weights in flat directions get larger ones.

GPTQ makes this tractable at scale through two key engineering choices: processing weights in large blocks (128 columns at a time) for better GPU utilization, and using a lazy batch update that defers Hessian updates within each block. The result: a 175B model quantized to 3 or 4 bits in approximately 4 GPU-hours, with perplexity degradation of 1-3 points depending on the model and bit-width.

The limitation is that GPTQ only quantizes weights -- activations remain in FP16 during inference. This means you get the memory savings from smaller weight storage but not the full throughput benefit of integer matmuls. For consumer GPU deployment where memory is the binding constraint, this is often sufficient.

### SmoothQuant: Taming Activation Outliers

Quantizing both weights *and* activations to INT8 (**W8A8**) would unlock integer Tensor Core throughput, but activations are much harder to quantize than weights. **SmoothQuant** (Xiao et al., ICML 2023) identified the core difficulty: large language models develop **outlier channels** -- specific feature dimensions where activation magnitudes are 10-100x larger than the rest. A single outlier in a channel forces the quantization scale for that entire tensor upward, crushing the dynamic range available for the majority of well-behaved values.

SmoothQuant's solution is mathematically elegant. Rather than trying to quantize the difficult activation distribution directly, it migrates the quantization difficulty from activations to weights via a per-channel scaling:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X}\hat{W}$$

The scaling factors are computed per channel:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

with $\alpha \approx 0.5$ balancing the difficulty between the two matrices. The intuition: channels where activations have large outliers get divided by a large $s_j$, bringing the activations into a more quantization-friendly range. The corresponding weight columns get multiplied by $s_j$, absorbing the scaling. Since weight distributions are typically much smoother than activation distributions, the weights remain easy to quantize even after absorbing the scaling.

The result: the smoothed activations $\hat{X}$ and the adjusted weights $\hat{W}$ are both easy to quantize to INT8, enabling full W8A8 inference for 175B+ parameter models with negligible accuracy loss. This is a preprocessing step -- the scaling is applied to the weights offline, with no per-token overhead at inference time.

### AWQ: Protecting What Matters

**AWQ** -- Activation-Aware Weight Quantization (Lin et al., MLSys 2024 Best Paper) -- takes a different approach to the same problem. Instead of smoothing everything uniformly, AWQ observes that only about 1% of weights are truly critical for model accuracy -- specifically, the weights that correspond to channels with large activation magnitudes. If $X_j$ is consistently large for channel $j$, then even a small quantization error in $W_j$ gets amplified by the large activation, producing a disproportionate output error.

AWQ protects these salient weights with per-channel scaling that is optimized to minimize the quantization error on the output:

$$s^* = \arg\min_s \| Q(W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot X) - WX \|$$

where $Q(\cdot)$ denotes the quantization function. The scaling factor $s$ is chosen to minimize the end-to-end error of each linear layer: scaling up a salient weight channel by $s_j$ effectively increases its precision (by making it a larger number relative to the quantization step size), while the corresponding inverse scaling on the activations is a simple rescaling that does not affect quantization difficulty.

AWQ achieves strong INT4 weight quantization for large language models, outperforming GPTQ on most benchmarks while being simpler to implement. It has become one of the most widely used quantization methods in production LLM serving (supported by vLLM, TensorRT-LLM, and most major inference frameworks).

### Quantization-Aware Training

When post-training methods are not sufficient -- for instance, when you need INT4 weights *and* INT4 or INT8 activations -- **Quantization-Aware Training (QAT)** inserts simulated quantization operations into the training graph. During the forward pass, weights and/or activations are quantized and immediately dequantized ("fake quantization"), so the model sees the quantization noise during training and learns to produce weight distributions that are robust to rounding.

The backward pass uses the **straight-through estimator** (STE) to propagate gradients through the non-differentiable rounding operation -- essentially treating the round function as the identity for gradient purposes. This is a theoretically questionable but practically effective approximation.

Google's **AQT** (Accurate Quantized Training) pushes this further with bit-exact consistency: the fake quantization during training exactly matches the integer arithmetic used at serving time, eliminating the train-serve skew that plagues many quantization approaches. If your serving stack uses asymmetric INT8 with specific rounding modes, AQT ensures training sees exactly those same rounding decisions.

The downside of QAT is cost: it requires training (or fine-tuning) the model, which for large language models can be prohibitively expensive. PTQ methods like AWQ exist precisely because the community needed quantization without this cost.

### TorchAO: Composable PyTorch-Native Quantization

**TorchAO** (ICML 2025) integrates quantization directly into the PyTorch ecosystem with an API that composes naturally with `torch.compile`. Rather than requiring custom inference runtimes, TorchAO represents quantized tensors as subclasses of `torch.Tensor` and generates optimized kernels through the compiler stack:

```python
import torch
import torchao

model = load_model().cuda().eval()

# One-line INT4 weight-only quantization
torchao.quantize_(model, torchao.quantization.int4_weight_only())

# Compile for additional speedup
model = torch.compile(model, mode="max-autotune")
```

This composability is the key innovation. Because quantized tensors are proper PyTorch tensors, they interact correctly with `torch.compile`, FSDP for distributed inference, and the broader PyTorch ecosystem. On Llama-3-8B, INT4 weight-only quantization via TorchAO delivers 1.89x inference speedup and 58% memory reduction with minimal accuracy impact.

### Practical Guidance

The quantization landscape is dense, so here is a decision framework:

| Method | Bits | Target | Accuracy Impact | Best For |
|--------|------|--------|-----------------|----------|
| PTQ (INT8) | 8 | W+A | Negligible | General deployment |
| SmoothQuant | 8 | W+A | <1% | LLM W8A8 serving |
| GPTQ | 3-4 | W only | 1-3% perplexity | Consumer GPU inference |
| AWQ | 4 | W only | <1% | Production LLM serving |
| QAT | 4-8 | W+A | Recovers most loss | Precision-critical apps |

A few patterns emerge. For **weight-only quantization** (W4A16 or W4A8), AWQ is the current default for production deployments -- it is fast, accurate, and widely supported. GPTQ remains useful when you need 3-bit quantization to fit a model on consumer hardware where the marginal accuracy loss is acceptable. For **weight-and-activation quantization** (W8A8), SmoothQuant is the established approach, enabling full utilization of INT8 Tensor Cores at inference time. And for cases where post-training accuracy is insufficient, QAT provides a path to recovery -- at the cost of retraining.

> INT4 is safe for encoder models but "causes significant accuracy drops for decoder-only models" -- precisely the models that most need compression. This is the fundamental tension in LLM quantization.

The autoregressive structure of decoder models amplifies quantization error across the generation sequence. Each token's prediction is conditioned on all previous tokens, so a small shift in a single layer's output compounds through subsequent layers and generation steps. Encoder models, which process all tokens in parallel without this sequential dependency, are more forgiving. This is why production LLM serving stacks like vLLM overwhelmingly use INT4 *weight-only* quantization (keeping activations in higher precision) rather than aggressive W4A4 schemes.

These quantization methods produce compressed models, but someone has to write the kernels that execute dequantization and mixed-precision arithmetic efficiently on actual hardware. That is the world of kernel optimization.

## Kernel Optimization: Writing Fast GPU Code

We have spent the last several sections talking about *what* to compute more efficiently -- lower precision, compressed weights, fused attention patterns. But at some point, someone has to sit down and write the actual GPU code that executes these operations. For decades, that meant writing CUDA C++, a notoriously unforgiving exercise involving shared memory management, warp-level synchronization, memory coalescing patterns, and register pressure tuning. A single misplaced `__syncthreads()` could silently produce wrong results; a suboptimal memory access pattern could leave 90% of your GPU bandwidth on the table.

The good news: we are living through a revolution in how GPU kernels get written. The bad news: you still need to understand *why* certain kernels are fast to make good optimization decisions. Let us start with the tool that changed the game.

### Triton: Block-Level GPU Programming

**Triton** (Tillet et al., 2019) is a Python-based language and compiler for writing GPU kernels at the *block level* rather than the thread level. Instead of reasoning about individual CUDA threads, warps, and shared memory -- the traditional CUDA programming model -- you reason about blocks of data that get loaded into SRAM, transformed, and written back. Triton's compiler handles the low-level details: shared memory allocation, memory coalescing, warp synchronization, and register allocation.

This is not a minor convenience. It is a fundamental shift in abstraction level that makes GPU programming accessible to ML researchers who know Python and NumPy but have no interest in becoming CUDA experts.

Here is a fused softmax kernel in Triton -- roughly 20 lines of actual logic:

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
                   output_row_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load entire row into SRAM — no HBM roundtrips for intermediate results
    row = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))

    # Compute softmax in SRAM
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    # Single write back to HBM
    out_start = output_ptr + row_idx * output_row_stride
    tl.store(out_start + col_offsets, softmax_out, mask=mask)
```

Why is this faster than `torch.softmax`? The answer is memory traffic. A standard PyTorch softmax implementation decomposes into three separate CUDA kernels:

1. **Find the row maximum** -- read entire row from HBM, write max back to HBM
2. **Compute exponentials and sum** -- read row + max from HBM, write exp values and sum back
3. **Normalize** -- read exp values + sum from HBM, write final result

That is three full round-trips to HBM (high bandwidth memory). On an A100, HBM bandwidth is ~2 TB/s, which sounds fast until you realize that SRAM bandwidth is ~19 TB/s -- nearly 10x higher. The Triton kernel does **one read from HBM, all computation in SRAM, and one write back**. For memory-bound operations like softmax (where the arithmetic intensity is low), reducing memory traffic *is* the optimization. Everything else is noise.

> The central insight of kernel fusion is simple: if you can keep data in SRAM between operations instead of writing it back to HBM and reading it again, you eliminate the dominant bottleneck for memory-bound operations. Triton makes this pattern easy to express.

Triton is not just an academic project anymore. It is the compilation backend for **TorchInductor**, PyTorch 2's default compiler. When you call `torch.compile()`, TorchInductor traces your model, fuses operations where possible, and generates Triton kernels automatically. This means every PyTorch user is already benefiting from Triton, whether they know it or not.

### Liger-Kernel: Drop-In Fused Kernels for LLM Training

Writing custom Triton kernels is powerful but requires expertise. What if someone already wrote the fused kernels you need for standard LLM architectures?

**Liger-Kernel** (LinkedIn, October 2024) is a collection of hand-optimized Triton kernels for the operations that dominate LLM training: **RMSNorm**, **RoPE** (rotary positional embeddings), **SwiGLU**, **CrossEntropyLoss**, and a particularly clever **FusedLinearCrossEntropy** that avoids materializing the full logit tensor. The results are striking: ~20% throughput increase and ~60% peak memory reduction on standard LLM training workloads.

The integration could not be simpler:

```python
from liger_kernel.transformers import apply_liger_kernel_to_llama
from transformers import LlamaForCausalLM

# One-line optimization: monkey-patch with fused Triton kernels
apply_liger_kernel_to_llama()
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# Training proceeds normally — 20% faster, 60% less memory
```

The FusedLinearCrossEntropy kernel deserves special attention. In standard LLM training, the final layer projects hidden states to vocabulary size (often 32,000-128,000), producing a massive logit tensor that exists only to be immediately consumed by the cross-entropy loss. Liger fuses the linear projection and loss computation, never materializing the full logit matrix. For a model with vocabulary size 128,000 and batch size 32 with sequence length 2048, this avoids allocating a tensor of ~32 GB in FP32 -- memory that was allocated, written, read once, and immediately discarded.

Liger-Kernel is fully compatible with **FSDP** (Fully Sharded Data Parallel) and **DeepSpeed**, so you can combine it with distributed training strategies without modification. This is the kind of optimization that should be a default: zero code changes to your training loop, significant performance gains, no accuracy impact.

### ThunderKittens: Tile-Primitive CUDA for the Next Generation

While Triton raises the abstraction level, **ThunderKittens** (Stanford HAI, 2024) takes a different approach: it provides a C++ template DSL built around **16x16 tile primitives** that map directly to the hardware granularity of Tensor Cores. The philosophy is that most GPU operations can be expressed as operations on small matrix tiles -- loads, stores, matrix multiplies, element-wise transforms -- and that making these tile operations first-class citizens simplifies kernel development while maintaining near-peak hardware utilization.

ThunderKittens v2.0 (January 2026) added **Blackwell architecture support**, including MXFP8 and NVFP4 data types, warpgroup-level primitives for Blackwell's new 128-thread warpgroups, and fifth-generation Tensor Core support. The ecosystem has expanded beyond NVIDIA: **ThunderMittens** targets Apple Silicon (M-series chips), and **HipKittens** targets AMD GPUs via HIP/ROCm.

The adoption is telling. Together AI uses ThunderKittens for production inference kernels. Jump Trading uses it for latency-sensitive workloads. Cursor -- the AI code editor -- uses ThunderKittens kernels in their serving stack. When organizations whose revenue depends on GPU performance choose your framework, that is a strong signal.

### LLM-Generated Kernels: Can AI Write Its Own Fast Code?

An increasingly fascinating question: can LLMs write their own optimized GPU kernels? **KernelBench** (Stanford, February 2025) attempted to answer this with a benchmark of 250 kernel generation tasks. The results were instructive but sobering. LLMs showed reasonable ability to fix execution failures -- syntax errors, launch configuration issues, basic correctness -- but **struggled badly with functional correctness** on non-trivial kernels. Worse, generated kernels frequently exploited benchmark loopholes: producing outputs that passed correctness checks by accident or through degenerate solutions that did not actually implement the requested operation.

More recent work has been more encouraging. **CUDA-L1** uses reinforcement learning to generate and iteratively optimize CUDA kernels, achieving an average **3.12x speedup** over baseline implementations. **CudaForge** reports 97.6% functional correctness with a 1.68x average speedup. These are not replacing expert kernel engineers yet, but the trajectory is clear: AI-assisted kernel optimization is moving from "interesting research" to "useful tool."

> The irony is not lost on anyone: we are using large language models to write the optimized GPU kernels that make large language models faster. Whether this becomes a virtuous cycle or hits fundamental correctness ceilings remains an open question.

## Production Serving: From Research to Reality

Training an optimized model is one thing. Serving it to millions of users with low latency, high throughput, and reasonable cost is a different engineering discipline entirely. The production serving stack has its own set of optimization techniques, and choosing the right serving framework can easily make a 3-5x difference in cost efficiency.

### TensorRT-LLM: NVIDIA's Deep Inference Optimizer

**TensorRT-LLM** (NVIDIA) is the most aggressive inference optimization engine available. It performs layer and tensor fusion, automatic kernel selection and tuning, precision calibration across FP16/INT8/FP8/FP4, and extensive graph-level optimizations. The performance numbers are typically the best you will find on NVIDIA hardware -- but at the cost of a complex build process and NVIDIA-only deployment.

A key feature is **speculative decoding integration**: TensorRT-LLM can coordinate a small draft model with the main model, achieving up to 3.6x throughput improvement on generation-heavy workloads. The framework has moved toward a **PyTorch-first architecture**, making it easier to integrate with existing training pipelines rather than requiring full model re-export through ONNX or custom graph formats.

The trade-off is build complexity. TensorRT engines must be compiled for specific GPU architectures, batch sizes, and sequence lengths. Changing any of these often requires rebuilding, which can take minutes. For organizations with stable deployment configurations, this is fine. For teams iterating rapidly on model architectures, the friction is real.

### ONNX Runtime: The Cross-Platform Play

**ONNX Runtime** (Microsoft) takes the opposite approach: maximum portability via a unified runtime with pluggable **execution providers** -- CUDA, TensorRT, DirectML (Windows), OpenVINO (Intel), CoreML (Apple), and many more. Export your model to ONNX format once, deploy it anywhere.

The honest assessment: ONNX Runtime **significantly lags behind TensorRT and native PyTorch in both latency and throughput** on NVIDIA GPUs. The abstraction penalty is real. Where ONNX Runtime shines is edge deployment, CPU inference, and heterogeneous environments where you need the same model running on NVIDIA GPUs, Intel CPUs, and Apple Neural Engines without maintaining three separate serving stacks. If your deployment target is a laptop or a mobile device, ONNX Runtime is often the pragmatic choice.

### vLLM: PagedAttention and the Open-Source Standard

**vLLM** (Kwon et al., 2023) introduced **PagedAttention**, which applies the operating system concept of virtual memory to KV cache management. Instead of pre-allocating contiguous memory for the maximum possible sequence length (which wastes enormous amounts of GPU memory on short sequences), PagedAttention allocates KV cache in small, fixed-size blocks and maps them through a page table. This simple idea yielded **2-4x throughput improvements** over FasterTransformer by dramatically improving memory utilization and enabling larger batch sizes.

vLLM has become the **de facto open-source LLM serving engine**. Its V1 architecture (January 2025) made `torch.compile` the default compilation backend, enabling smart graph fusions without the build-time complexity of TensorRT. The V1 rewrite also simplified the internal architecture, consolidating separate scheduling and execution paths into a more maintainable codebase.

However, PagedAttention is not free. The indirection through page tables adds overhead: benchmarks show **20-26% slower attention computation** compared to contiguous memory layouts. **vAttention** (Microsoft) proposes an alternative that leverages OS-level virtual memory (using CUDA's `cuMemMap` API) to provide the memory efficiency benefits of paging without the attention computation overhead. The KV cache appears contiguous to the attention kernel while the OS handles physical memory management transparently.

### SGLang: When the Bottleneck Is Not the Kernel

**SGLang** (Zheng et al.) comes from a different angle. Its key innovations are **RadixAttention** -- which uses a radix tree to share KV cache prefixes across requests with common prompts -- and deeply integrated **constrained decoding** that can enforce JSON schemas, regex patterns, or context-free grammars during generation with minimal overhead.

On raw throughput benchmarks, SGLang shows a notable advantage: **16,215 tokens/s versus vLLM's 12,553 tokens/s** (a 29% gap) on comparable configurations. The SGLang team's analysis points to an important insight about where the bottleneck has moved:

> The bottleneck is no longer the mathematical kernel, but the engine's internal orchestration overhead -- scheduling, memory management, request routing, and tokenizer throughput. Optimizing the "plumbing" matters as much as optimizing the "math."

This is a sign of maturity in the field. When kernel implementations are fast enough that scheduling overhead dominates, the optimization frontier shifts from CUDA engineering to systems engineering.

### Speculative Decoding: Getting More Tokens per Forward Pass

**Speculative decoding** (Leviathan et al., 2023) is one of the most elegant optimization techniques in production serving. The core idea: autoregressive generation is bottlenecked by sequential forward passes, but verification can be parallelized. A small, fast **draft model** proposes $k$ candidate tokens. The large **target model** then scores all $k$ tokens in a single forward pass -- the same cost as generating one token. Each proposed token $x_i$ is accepted with probability:

$$p_{\text{accept}}(x_i) = \min\left(1, \frac{p_{\text{target}}(x_i)}{p_{\text{draft}}(x_i)}\right)$$

This acceptance criterion guarantees that the output distribution is **exactly identical** to sampling from the target model alone -- speculative decoding is mathematically lossless. If the draft model is good (high acceptance rate), you get 2-3x speedup. If the draft model is poor, you simply fall back to standard generation with minimal overhead.

The **EAGLE** series (Li et al., 2024) improves on naive speculative decoding by training the draft head to predict hidden states rather than tokens, achieving higher acceptance rates. **Medusa** (Cai et al., 2024) adds multiple parallel prediction heads to the target model itself, eliminating the need for a separate draft model. **ReDrafter** (Apple, 2024) combines draft models with beam search for even higher acceptance rates.

In practice, speculative decoding is most effective for tasks where the draft model can predict well -- code completion (highly structured), continuation of existing text (predictable patterns), and constrained generation (limited token space). It is less effective for creative or highly unpredictable outputs where the draft model's acceptance rate drops.

### The Cold Start Problem

A persistent pain point in production: **cold start latency**. TensorRT engine compilation can take minutes for large models. `torch.compile` JIT compilation adds 20+ seconds on the first forward pass. For serverless or auto-scaling deployments, this means new instances take unacceptably long to start serving traffic.

**AOTInductor** (Ahead-of-Time Inductor) addresses this by performing the `torch.compile` compilation step offline and serializing the optimized artifact. At serving time, you load the pre-compiled model directly -- no JIT overhead. One organization reported reducing cold start from **6+ minutes to 40 seconds** using AOTInductor combined with pre-built TensorRT engines and model weight caching.

The broader lesson: production optimization is not just about steady-state throughput. Startup time, graceful degradation under load, memory fragmentation over long-running sessions, and tail latency all matter. The fastest kernel in the world does not help if your service takes five minutes to cold start during a traffic spike.

## Practical Guide: What to Use When

After all this theory, let us get practical. The optimization landscape is large, but most real-world scenarios map to a handful of well-understood recipes. Here is my recommended starting point for the most common situations:

| Scenario | Start Here | Then Try | Advanced |
|----------|-----------|----------|----------|
| Training (any model) | AMP (BF16) | `torch.compile` | FP8 (Hopper+) |
| Inference (latency) | `torch.compile` + CUDA graphs | TensorRT-LLM | Speculative decoding |
| Inference (throughput) | vLLM or SGLang | AWQ INT4 | `torch.compile` + FlashAttention |
| Edge / mobile | ONNX Runtime | INT8 PTQ | ExecuTorch |
| Memory-constrained | AWQ or GPTQ INT4 | Liger-Kernel | 2:4 sparsity + INT4 |

### The 80/20 Rule of Model Optimization

If you remember nothing else from this post, remember these four steps. They cover the vast majority of practical optimization gains with minimal effort:

1. **Use BF16 mixed precision** -- this is essentially a free 2x speedup on any modern GPU (Ampere or newer). There is almost no reason to train or serve in FP32 anymore. BF16 maintains the dynamic range of FP32 while halving memory and doubling arithmetic throughput. If you are still using FP32 by default, this single change will have the largest impact of anything in this post.

2. **Add `torch.compile`** -- wrapping your model in `torch.compile(model)` typically yields 30-50% additional speedup with one line of code. The compiler fuses operations, eliminates Python overhead, and generates optimized Triton kernels. For training, use `mode="reduce-overhead"` to get CUDA graph integration. For inference, `mode="max-autotune"` spends more compilation time searching for the fastest kernel configurations.

3. **Use FlashAttention** -- this is the default in most modern frameworks (Hugging Face Transformers, PyTorch 2.x SDPA), so you may already be using it. Verify by checking that `torch.nn.functional.scaled_dot_product_attention` is being called instead of manual attention computation. FlashAttention gives 2-4x speedup on the attention computation with significantly reduced memory usage.

4. **Quantize for serving** -- once your model is trained, serve it in AWQ INT4 for most LLM workloads. This cuts memory by ~4x (enabling larger batch sizes or smaller GPUs) with minimal quality degradation on instruction-following and generation tasks. For tasks requiring higher precision (math, code, reasoning), INT8 is a safer choice.

> These four optimizations -- BF16, torch.compile, FlashAttention, and INT4 quantization -- will capture roughly 80% of the available performance gains for roughly 20% of the engineering effort. Everything beyond this is diminishing returns (sometimes important diminishing returns, but diminishing nonetheless).

### Common Mistakes

Having worked through optimizing models in various settings, here are the mistakes I see most frequently:

**Optimizing before profiling.** The number one mistake. People spend days writing custom CUDA kernels for an operation that accounts for 3% of their runtime, while ignoring a data loading bottleneck that accounts for 40%. Always profile first. `torch.profiler`, NVIDIA Nsight Systems, and even simple `torch.cuda.Event` timing will tell you where your time actually goes. My intuition is that at least half of all "GPU optimization" time is wasted on the wrong bottleneck.

**Applying CUDA graphs with dynamic shapes.** CUDA graphs record a fixed sequence of GPU operations and replay them with minimal CPU overhead -- but they require fixed tensor shapes. If your sequence lengths vary (they almost always do in LLM serving), naive CUDA graph usage either fails or requires padding to the maximum length, wasting computation. The solution is to bucket sequence lengths into a small number of sizes and maintain separate CUDA graphs per bucket, which is what vLLM and SGLang do internally.

**INT4 quantization without validation.** I have seen teams quantize a model to INT4, observe that perplexity on a held-out set looks fine, and deploy. Then they discover that the model produces subtly wrong outputs on domain-specific tasks -- hallucinating numbers in financial reports, generating syntactically invalid code, or losing multilingual capability. Always validate on your *actual* downstream task, not just perplexity. And for tasks requiring precision -- math reasoning, structured data extraction, code generation -- consider INT8 or even FP8 instead.

**Skipping warmup before benchmarking.** The first few forward passes through a model involve CUDA context initialization, kernel compilation, memory allocation, and caching. If you include these in your timing, your numbers will be misleading. Standard practice: run 10-50 warmup iterations, then time the next 100-1000 iterations. Report the median, not the mean (to exclude GC pauses and other outliers).

**Mixing optimization techniques without understanding interactions.** Not all optimizations compose cleanly. `torch.compile` with certain custom CUDA extensions can fail silently or produce incorrect results. Quantization with LoRA adapters requires careful handling of which parameters are quantized. FlashAttention with non-standard attention masks may require falling back to the standard implementation. Test each optimization individually before combining them.

## Conclusion

The full optimization stack -- compilers, kernels, precision, quantization, and serving -- is what stands between a model that works in a notebook and one that runs profitably in production. The techniques in this post compose: BF16 mixed precision, `torch.compile` for operator fusion, FlashAttention for IO-aware attention, AWQ for weight compression, and vLLM or SGLang for serving. Together, they can deliver order-of-magnitude improvements in throughput and cost.

The optimization landscape is evolving fast enough that anything I write here risks obsolescence within months. That said, several trends seem durable enough to bet on.

**FP4 at scale.** NVIDIA's Blackwell architecture provides hardware support for FP4 computation, and DeepSeek-V3 has demonstrated FP8 training at 671B parameters. The obvious question is whether FP4 can work at the 100B+ scale for training -- not just inference. The hardware capability is there; the training methodology is not mature. My suspicion is that we will see FP4 work for fine-tuning and the later stages of training before it becomes viable for pretraining from scratch, following the same adoption pattern as FP8.

**Compiler-kernel convergence.** The boundary between "compiler-generated code" and "hand-written kernel" is blurring rapidly. **Flashlight** (November 2025) demonstrated the ability to generate FlashAttention-style IO-aware kernels from standard PyTorch code -- no manual kernel engineering required. **FlexAttention** (PyTorch) makes custom attention patterns (sliding window, sparse, block-sparse) expressible as simple Python mask functions that the compiler optimizes into efficient kernels. We are moving toward a world where expressing *what* you want to compute is sufficient, and the compiler figures out *how* to compute it efficiently.

**LLM-generated kernels.** CUDA-L1 and CudaForge show that AI-assisted kernel optimization is viable for achieving meaningful speedups. The correctness challenge identified by KernelBench remains real, but the rapid improvement trajectory suggests that within a few years, LLM-generated kernels will be a standard part of the optimization workflow -- likely with human review and verification, similar to how LLM-generated code is used in software engineering today.

**Disaggregated inference.** The current standard -- running prefill and decode on the same GPU -- is suboptimal because these phases have very different computational profiles. Prefill is compute-bound (large matrix multiplications on the full input); decode is memory-bound (sequential token generation). **Disaggregated inference** separates these onto different nodes optimized for each workload. **LMCache** and similar projects treat the KV cache as a first-class distributed resource, enabling prefill results to be shared across decode nodes, cached across requests, and even persisted across sessions.

**BitNet and ternary quantization.** Microsoft's **BitNet b1.58** demonstrated that models with ternary weights $\{-1, 0, +1\}$ can match FP16 quality at smaller scales while being dramatically faster (no floating-point multiply, only additions and subtractions). If this scales -- and that remains a big "if" -- it could render most post-training quantization techniques obsolete. Why compress a model from FP16 to INT4 after training if you can train directly in 1.58 bits? The counterargument is that ternary training requires fundamentally different optimization techniques and does not benefit from the massive existing infrastructure built for floating-point training.

**The end of manual optimization?** I suspect we are approaching a phase transition. As compilers like Flashlight and FlexAttention close the gap with hand-written kernels, as hardware like Blackwell and TPU v7 provides higher raw throughput, and as AI-assisted tools handle more of the low-level optimization work, the skill that matters shifts from "write clever CUDA code" to "express computational intent clearly and choose the right abstractions." The best GPU programmer of 2028 may not write a single line of CUDA -- but they will deeply understand memory hierarchies, arithmetic intensity, and hardware capabilities, because those are the concepts you need to guide a compiler toward the right solution.

> The optimization techniques in this post will not all survive the next five years. The *mental models* -- roofline analysis, memory-boundedness, the arithmetic intensity spectrum, the fusion principle -- are timeless. Learn the principles, and the specific tools become interchangeable.

## References

- [Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures", 2009](https://doi.org/10.1145/1498765.1498785)
- [Han et al., "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding", ICLR 2016](https://arxiv.org/abs/1510.00149)
- [Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018](https://arxiv.org/abs/1712.05877)
- [Micikevicius et al., "Mixed Precision Training", ICLR 2018](https://arxiv.org/abs/1710.03740)
- [Milakov & Gimelshein, "Online Normalizer Calculation for Softmax", 2018](https://arxiv.org/abs/1805.02867)
- [Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018](https://arxiv.org/abs/1802.01946)
- [Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", MAPL 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation", CGO 2021](https://arxiv.org/abs/2002.11054)
- [Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022](https://arxiv.org/abs/2205.14135)
- [Micikevicius et al., "FP8 Formats for Deep Learning", 2022](https://arxiv.org/abs/2209.05433)
- [Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023](https://arxiv.org/abs/2307.08691)
- [Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers", ICLR 2023](https://arxiv.org/abs/2210.17323)
- [Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2023](https://arxiv.org/abs/2211.10438)
- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023](https://arxiv.org/abs/2309.06180)
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023](https://arxiv.org/abs/2211.17192)
- [Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", MLSys 2024](https://arxiv.org/abs/2306.00978)
- [Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation", ASPLOS 2024](https://arxiv.org/abs/2403.20107)
- [Dao & Shah, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024](https://arxiv.org/abs/2407.08608)
- [DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024](https://arxiv.org/abs/2412.19437)
- [Hsu et al., "Liger-Kernel: Efficient Triton Kernels for LLM Training", LinkedIn, 2024](https://arxiv.org/abs/2410.10989)
- [Spector et al., "ThunderKittens: Simple, Fast, and Adorable AI Kernels", Stanford HAI, 2024](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- [Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024](https://arxiv.org/abs/2401.15077)
- [Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024](https://arxiv.org/abs/2401.10774)
- [Scala et al., "KernelBench: Can LLMs Write GPU Kernels?", Stanford, 2025](https://arxiv.org/abs/2502.10517)
- [Fireworks AI, "Speed, Python: Pick Two. How CUDA Graphs Enable Fast Python Code for Deep Learning", 2023](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning)
- [Dong et al., "FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention", 2024](https://pytorch.org/blog/flexattention/)
- [Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)", 2024](https://arxiv.org/abs/2402.17764)
- [Prabhu et al., "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention", 2024](https://arxiv.org/abs/2405.04437)
- [Apple, "ReDrafter: Fast Speculative Decoding with Recurrence Draft Model", 2024](https://arxiv.org/abs/2403.09919)
