I had two days. I wanted to know how fast my M4 could push molecular dynamics.

Lennard-Jones N-body. Same physics, same particles, fifteen different force kernels. NEON intrinsics on the P-cores. Metal compute shaders on the GPU. The Apple Neural Engine via private APIs CoreML doesn't expose. SVE2 streaming on the SME coprocessor. CUDA on a borrowed L4. Axion SVE2 on a GCP VM for cross-arch sanity.

I wasn't trying to ship a fast MD code. I wanted to see how the design space changes when iteration is minutes, not days. Numbers below, with the parts that surprised me.

## The fifteen kernels

| Kernel | Hardware | Description |
|---|---|---|
| Scalar | CPU (1 core) | Plain C, Newton's 3rd law, real division. Baseline. |
| NEON | CPU SIMD | Hand-written ARM NEON intrinsics, float32x4, 4 particles/op |
| NEON+N3L | CPU SIMD | NEON with Newton's 3rd law. Half the pairs, SIMD scatter-write |
| NEON f64 | CPU SIMD | Double-precision NEON (float64x2), 2 particles/op |
| OpenMP | CPU multi-core | NEON kernel parallelized across P-cores |
| SME2 | SME coprocessor | ARM Scalable Matrix Extension 2, streaming SVE (16-wide) |
| Tiled | CPU multi-core | Cache-tiled OpenMP+NEON with CPU pinning |
| Metal | M4 GPU | Metal compute, tiled all-pairs, Horner polynomial |
| NEON+CL | CPU SIMD | NEON with cell list neighbor lists. O(N) scaling |
| OMP+CL | CPU multi-core | OpenMP+NEON with cell lists |
| Metal+CL | M4 GPU | Metal compute with cell lists. O(N) GPU |
| Metal+BVH | M4 GPU | Two-pass: quantized BVH traversal, then force compute |
| Metal+NBNXM | M4 GPU | GROMACS-style 8x8 cluster pair lists. O(N) with SIMD coherence |
| ANE Direct | M4 Neural Engine | Exact LJ via private APIs. FP16 on ANE, NEON accumulation on CPU |

A CUDA kernel for the L4 and an Axion SVE2 build round it out. Same algorithm, different silicon, useful for cross-arch reasoning.

## NEON ceiling, and why scalar+N3L beats it

Single-core SIMD on the M4 tops out around **29 GFLOPS** for this kernel. Four FMA pipes, four floats per lane. Here's the inner loop, unedited:

```c
float32x4_t r2 = vmulq_f32(dx, dx);
r2 = vfmaq_f32(r2, dy, dy);    /* r2 += dy*dy */
r2 = vfmaq_f32(r2, dz, dz);    /* r2 += dz*dz */

float32x4_t inv_r2 = vrecpeq_f32(safe_r2);                /* ~12-bit estimate */
inv_r2 = vmulq_f32(inv_r2, vrecpsq_f32(safe_r2, inv_r2)); /* refine to ~24-bit */

float32x4_t inv_r6  = vmulq_f32(vmulq_f32(inv_r2, inv_r2), inv_r2);
float32x4_t inv_r12 = vmulq_f32(inv_r6, inv_r6);

float32x4_t f_over_r = vmulq_f32(veps24, inv_r2);
f_over_r = vmulq_f32(f_over_r, vsubq_f32(vmulq_f32(vtwo, inv_r12), inv_r6));

fxi = vfmaq_f32(fxi, f_over_r, dx);
fyi = vfmaq_f32(fyi, f_over_r, dy);
fzi = vfmaq_f32(fzi, f_over_r, dz);
```

No `pow`, no division. `vrecpeq_f32` plus one Newton-Raphson step gets `1/r²` to ~24 bits in two cycles. The LJ chain becomes pure multiplies. That's where the 29 GFLOPS comes from.

Newton's 3rd law cuts the pair count in half. For every (i, j) you'd compute, you skip (j, i) and write the negated force back. Done naively, this wrecks SIMD: the lanes scatter to four different j-particle force accumulators, and the scatter writes hurt. Plain scalar with N3L runs at 17.8 GFLOPS but does half the work. NEON without N3L runs at 28.7 GFLOPS but does all of it. Wall-clock at N=10976: scalar+N3L wins.

**SIMD width is not throughput.** The instruction with fewer lanes can win if it touches less work. Measure end to end, not in GFLOPS.

## Metal GPU crossover and the cell-list flip

The Metal kernel does the same physics on GPU threadgroups, but the LJ chain is no longer an analytical evaluation. It's a piecewise polynomial fit to `f/r` over the relevant `r²` range. Sixteen intervals, degree-4, Horner's method:

```metal
float scaled = (safe_r2 - CHEB_R2_MIN) * CHEB_SCALE;
uint  idx    = min(uint(scaled), uint(CHEB_N - 1));
float t      = scaled - float(idx);

/* 4 chained FMAs replace the 8-cycle analytical chain */
float f_over_r = fma(fma(fma(fma(
    cheb_c4[idx], t, cheb_c3[idx]),
    t, cheb_c2[idx]),
    t, cheb_c1[idx]),
    t, cheb_c0[idx]);
```

Coefficients live in `constant` address space. No memory fetch, baked into the shader binary. Half-precision storage for the position tile, single-precision for accumulation. Everything that touches the M4's unified memory does so zero-copy.

All-pairs Metal hits **810 GFLOPS** at 87K particles. Roughly 19% of the chip's 4.26 TFLOPS theoretical. Reasonable. The crossover with OpenMP happens around 3K particles; below that, kernel launch overhead and the tiled-load setup dominate, and the four P-cores beat the GPU.

Then cell lists changed the shape entirely. O(N²) becomes O(N). Linked-list cells, gather neighbors into a contiguous buffer per cell, run the same Horner inner loop on the gathered list. At N=70304, Metal+CL is **45x faster** than Metal all-pairs.

The non-obvious result: **OMP+CL on 4 P-cores beats all-pairs Metal GPU above 32K particles.** Once the algorithm flips from O(N²) to O(N), per-particle compute drops below the GPU's launch and sync overhead. The CPU's cache-friendly access pattern, with NEON in the inner loop, wins. Right hardware depends on the algorithm, not the chip.

## ANE direct, the actual interesting kernel

The Apple Neural Engine is 15.8 TFLOPS of FP16. CoreML hides it behind a model-loading API meant for ML inference: build a `.mlmodelc`, hand it to a compiler, dispatch via Vision or CoreML at roughly 200ms launch overhead per inference. Non-starter for a hot loop.

There's another way. `H11ANEServices.framework` and `ANECompilerService` are private frameworks shipped with macOS. They expose a lower-level dispatch path: build an ANE program directly, hand it raw FP16 buffers, dispatch in microseconds. Apple does not document this. The headers aren't in the SDK. You assemble them from `class-dump` output and the function signatures present in the binary.

So I assembled them. Built an LJ force chain as a sequence of FP16 ops the ANE accepts (multiplies, FMAs, reciprocal-via-Newton-step), compiled via the private compiler service, dispatched via the device controller. NEON computes the distance matrix on the CPU. The ANE evaluates the LJ chain in FP16. NEON accumulates the forces back.

Measured: **15 GFLOPS. 0.1% of the ANE's peak.**

That number is the point. The ANE is not the bottleneck. The CPU↔ANE data path is. Each dispatch ships an FP16 buffer of `r²` values to the ANE and reads an FP16 buffer of `f/r` back. The bandwidth ceiling on that round trip is what caps throughput. Not the ANE's 15.8 TFLOPS. Not the LJ chain's arithmetic intensity. Not the CPU's NEON pipes.

The interesting result here isn't the GFLOPS. It's that direct ANE dispatch from C, bypassing CoreML, works at all. For workloads where the data path is amortized over enough ANE work, like large-batch FP16 chains, the path is real. For LJ specifically, where each pair is one Horner-equivalent evaluation, it isn't worth it. But you don't know until you build it.

## Wall-clock, all in one place

GFLOPS lies. Wall-clock is the truth. Cell-list kernels, M4 only, ms per step:

| N | NEON | NEON+CL | OMP+CL (4P) | Metal AP | Metal+CL |
|---|---|---|---|---|---|
| 864 | 0.58 | 0.73 | n/a | 1.5 | 1.2 |
| 4,000 | 10.9 | 4.3 | n/a | 1.33 | 0.64 |
| 10,976 | 83.1 | 9.6 | 2.65 | 3.70 | 0.64 |
| 32,000 | 698 | 26.1 | 7.56 | 26.5 | 2.29 |
| 70,304 | n/a | 56.3 | 15.7 | ~190 | 4.26 |

Read across at a fixed N to see where the algorithm-plus-hardware combination flips. At 10K particles, the right answer is OMP+CL. At 70K particles, the right answer is Metal+CL. The gap is two orders of magnitude.

A few side notes from the rest of the matrix. f64 NEON is 2.2x slower than f32 NEON, with 0.02% energy drift over 10⁶ steps versus 0.009% for f64. Timestep error dominates either way, so float32 is sufficient for LJ MD. SME2 ran at 5.5 GFLOPS on this kernel: wrong tool. SME2 is designed for FMOPA matrix outer products, not element-wise pair sweeps; the streaming SVE path through L2 is bandwidth-limited for this access pattern. Useful to know, useful not to use.

## The point

The point is not any single kernel. The point is that the iteration cycle for hardware design-space exploration collapsed.

Five GPU strategies in a single sitting: all-pairs, cell lists, BVH traversal, NBNXM cluster pair lists, with and without Newton's 3rd law. Three architectures locally (M4 NEON, M4 Metal, M4 ANE). Two more cross-compiled and benchmarked on cloud (NVIDIA L4 via CUDA, GCP Axion via SVE2). Most of these are dead ends. SME2 was the wrong instruction set. ANE direct was a 0.1%-of-peak curiosity. NBNXM cluster pairs didn't beat plain cell lists at this workload size.

Things I didn't try: octree spatial decomposition with cell-pair pruning to skip the far-field cells entirely, Ewald summation for long-range correctness, GPU/CPU overlap, SIMD-aware cluster sizing on the Metal NBNXM kernel, FP16 accumulation experiments on the GPU. A thousand things I could chase. Two days was the budget. The work is done when the curve flattens, not when the design space is exhausted.

But you don't know which is a dead end until you measure. Measuring used to be the slow part. Set up an HPC job, wait for the queue, parse the output, repeat tomorrow. With a workstation that has the silicon you care about and tooling that compiles in seconds, it isn't anymore. Write a kernel, benchmark it, profile it, understand why it's slow from the hardware, pivot. The design space is the same size it always was. The cost of walking through it is what changed.
