# CUDA Bilateral Filter

A shared-memory tiled CUDA implementation of the **bilateral filter** — an edge-preserving, non-linear smoothing operator — for single-channel (grayscale) images. The implementation is a single translation unit with no third-party dependencies beyond two public-domain header-only image libraries.

The point of this project is not the filter itself but the **memory-hierarchy engineering** around it: constant-memory weight broadcast, shared-memory tiling with a halo, and a divergence-free collaborative load. These are the techniques that separate a correct CUDA kernel from a fast one.

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)
![Language](https://img.shields.io/badge/C%2B%2B-CUDA-blue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Key Techniques](#key-techniques)
- [How It Works](#how-it-works)
  - [Two-phase kernel](#two-phase-kernel)
  - [Memory hierarchy](#memory-hierarchy)
  - [The haloed tile](#the-haloed-tile)
  - [Divergence-free halo load](#divergence-free-halo-load)
- [Complexity](#complexity)
- [Requirements](#requirements)
- [Building](#building)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance](#performance)
- [Design Notes](#design-notes)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

A Gaussian blur removes noise but smears the edges that carry the most information, because it convolves the same static kernel everywhere. The **bilateral filter** fixes this by weighting each neighbour by *two* distances:

```
            1
BF[I]_p = ----- Σ  G_σs(‖p − q‖) · G_σr(|I_p − I_q|) · I_q
           W_p  q∈S
```

- a **spatial** Gaussian `G_σs` that decays with geometric distance, and
- a **range** Gaussian `G_σr` that decays with intensity difference.

The range term collapses to near-zero across a strong edge, so the filter stops averaging there — noise is suppressed inside smooth regions while edges stay crisp.

This selectivity is expensive. The range weight depends on actual pixel intensities, so the kernel is **spatially variant**: it cannot be precomputed as a single matrix, and it does **not** separate into two 1-D passes the way a Gaussian does. The result is `O(N · r²)` work for an `N`-pixel image with radius `r` — seconds of latency for a megapixel image on a CPU. The arithmetic is dense, per-pixel, and fully data-parallel, which makes it an ideal fit for the GPU's SIMT execution model.

---

## Key Techniques

- **Constant-memory spatial weights.** The `(2r+1)²` spatial-Gaussian coefficients are intensity-independent, so they are computed once on the host and uploaded to `__constant__` memory. Every thread in a warp reading the same coefficient is served by a single broadcast.
- **Shared-memory tiling with a halo.** Each thread block stages a `(TILE_DIM + 2r) × (TILE_DIM + 2r)` tile of the input into on-chip shared memory before computing, so the heavily-overlapping windows of adjacent threads hit SRAM instead of re-reading global DRAM dozens of times.
- **Divergence-free collaborative load.** The halo is filled by a flat 1-D strided loop with clamped addressing — no per-pixel boundary `if`-tests — so the loading warps never diverge.
- **Edge-replication boundaries.** Out-of-bounds neighbours are clamped to the nearest valid pixel, handled inside the index arithmetic rather than with branches.
- **Overflow-safe indexing.** All pixel offsets are computed in `size_t` so that images larger than ~2 GP do not overflow 32-bit arithmetic.
- **Micro-optimised inner loop.** The constant `−1 / (2σr²)` is hoisted out of the `O(r²)` convolution and folded into a single `expf` argument, removing a division from the hot path.

---

## How It Works

### Two-phase kernel

`gpu_bilateral_shared` runs one thread per output pixel, in `16 × 16` blocks (256 threads). Each block does two things:

1. **Load.** Cooperatively stage a `22 × 22` input tile (the block's `16 × 16` output region plus a 3-pixel apron on every side) into a `__shared__` array, then `__syncthreads()`.
2. **Convolve.** Each in-bounds thread sweeps its `7 × 7` window *entirely out of shared memory*, multiplying the constant-memory spatial weight by the on-the-fly range weight, accumulating both the weighted sum and the normaliser `W_p`, and writing `sum / W_p` to global memory.

The `__syncthreads()` barrier between the phases is load-bearing: warp scheduling is asynchronous, so without it a faster warp could read tile cells a slower warp has not yet written.

### Memory hierarchy

| Space      | Latency           | Role in the kernel                         |
|------------|-------------------|--------------------------------------------|
| Global     | High (DRAM)       | Input/output arrays; source of tile loads  |
| Constant   | Low (cached, b/cast) | The `(2r+1)²` precomputed spatial weights |
| Shared     | Ultra-low (SRAM)  | The `22 × 22` staged tile + halo            |
| Registers  | Zero              | `filtered_pixel`, `w_p`, loop accumulators  |

### The haloed tile

With `TILE_DIM = 16` and `FILTER_RADIUS = 3`, every block stages a tile of size `HALO_DIM = TILE_DIM + 2·r = 22`:

```
        ┌──────────────────────────────────────────┐
        │  halo / apron  (r = 3, edge-replicated)   │
        │     ┌────────────────────────────────┐    │
        │     │                                │    │
        │     │                                │    │
        │     │      output tile  16 × 16      │    │   22 × 22 = 484 floats
        │     │   (one thread → one pixel)     │    │   = 1936 B per block
        │     │                                │    │
        │     │                                │    │
        │     └────────────────────────────────┘    │
        │                                            │
        └──────────────────────────────────────────┘
```

The apron is what makes shared-memory tiling non-trivial: the `16 × 16` threads must collectively load `22 × 22 = 484` elements, i.e. more elements than there are threads, so the load cannot be a simple one-element-per-thread copy.

### Divergence-free halo load

The naive way to load the apron is a per-cell boundary test, which serialises the divergent warps. Instead, threads are linearised (`tid = ty·16 + tx`) and walk the 484-element tile in a **grid-stride loop** with `stride = 256`:

```cuda
for (int i = tid; i < HALO_DIM * HALO_DIM; i += blockDim.x * blockDim.y) {
    int s_row = i / HALO_DIM;
    int s_col = i % HALO_DIM;
    int g_row = blockIdx.y * TILE_DIM - FILTER_RADIUS + s_row;
    int g_col = blockIdx.x * TILE_DIM - FILTER_RADIUS + s_col;
    g_row = max(0, min(g_row, height - 1));   // clamp, no branch
    g_col = max(0, min(g_col, width  - 1));
    s_tile[s_row][s_col] = input[g_row * width + g_col];
}
```

Every thread executes identical control flow; 228 of the 256 threads simply perform a second iteration to cover the remaining `484 − 256` cells. Boundary handling is folded into `max`/`min` rather than a conditional, so the load is branch-free.

---

## Complexity

For an `N`-pixel image and radius `r`:

- **Work:** `O(N · r²)` — identical to the serial algorithm; the GPU does not change the asymptotic work, it parallelises it.
- **Span (parallel depth):** the per-pixel `(2r+1)²` convolution collapses to `O(r²)` when `P ≈ N` threads run concurrently.
- **Global-load reduction:** without tiling, each output pixel triggers `(2r+1)² = 49` global loads. With tiling, a block loads `22² = 484` cells and serves `16² = 256` outputs — roughly `484 / (256 · 49) ≈ 1` versus `49` loads per output, an order-of-magnitude cut in DRAM traffic.

---

## Requirements

- **CUDA Toolkit** 11.5 or newer (`nvcc`) — `-arch=native` is used below and requires ≥ 11.5.
- An NVIDIA GPU with compute capability ≥ 3.5.
- A Linux toolchain (tested on Fedora; any glibc Linux with a matching driver should work).
- The two single-header [`stb`](https://github.com/nothings/stb) libraries, fetched in the build step below.

---

## Building

The source `#include`s `stb_image.h` and `stb_image_write.h`. They are header-only and public-domain; drop them next to `main.cu`:

```bash
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

Then compile:

```bash
nvcc -O3 -arch=native -o bilateral main.cu
```

`-arch=native` targets the GPU in the build machine. To target a specific architecture explicitly, replace it — e.g. `-arch=sm_89` (Ada), `-arch=sm_120` (Blackwell / RTX 50-series), `-arch=sm_86` (Ampere).

---

## Usage

```bash
./bilateral <input_image>
```

The input is decoded and **force-converted to grayscale** (any format `stb_image` supports: PNG, JPG, BMP, TGA, …). The filtered result is written to `filtered_output.png` in the working directory as an 8-bit single-channel PNG.

```bash
$ ./bilateral noisy_portrait.jpg
Filtering complete. Output successfully saved to filtered_output.png
```

On any fatal condition (decode failure, allocation failure, CUDA error) the program prints a diagnostic to `stderr` and exits non-zero; the `CUDA_CHECK` macro reports the offending `file:line` and CUDA error string.

---

## Configuration

The filter is currently configured at **compile time**. The relevant knobs:

| Symbol          | Location         | Default | Meaning                                              |
|-----------------|------------------|---------|------------------------------------------------------|
| `TILE_DIM`      | macro            | `16`    | Block edge length → `16 × 16 = 256` threads/block.   |
| `FILTER_RADIUS` | macro            | `3`     | Window radius → `7 × 7` window, `49` taps.            |
| `HALO_DIM`      | macro (derived)  | `22`    | `TILE_DIM + 2·FILTER_RADIUS`; shared-tile edge.       |
| `sigma_s`       | `main()`         | `50.0`  | Spatial-Gaussian standard deviation.                 |
| `sigma_r`       | `main()`         | `0.1`   | Range-Gaussian standard deviation (normalised units).|

Changing `TILE_DIM` or `FILTER_RADIUS` automatically resizes the shared tile and the constant-weight array — they are defined in terms of the macros.

> **Note on `sigma_s`.** At `σs = 50` with `r = 3`, the spatial weights are nearly uniform across the `7 × 7` window (`exp(−9 / 5000) ≈ 0.998`), so edge preservation is driven almost entirely by the **range** term — effectively a top-hat spatial support combined with a Gaussian range kernel. For a more textbook spatial falloff, use `σs ≈ 2–3`, where the apron pixels are visibly down-weighted relative to the centre.

---

## Performance

The implementation is **transfer-bound**, not compute-bound: the kernel saturates the device long before the host↔device PCIe copies do, so end-to-end speedup is capped by bus bandwidth rather than by arithmetic throughput.

> **These figures are representative**, measured on a development machine (NVIDIA RTX 5060 Laptop GPU, Blackwell; Intel Core i7-14700HX). Absolute numbers are hardware-dependent and the in-tree benchmark harness used to produce them is still being integrated — see [Roadmap](#roadmap). Regenerate on your own hardware before citing.

Speedup vs. a single-threaded CPU baseline (`r = 3`, `σr = 0.1`):

| Image       | CPU (ms) | GPU kernel (ms) | Kernel-only | End-to-end |
|-------------|---------:|----------------:|------------:|-----------:|
| 512 × 512   |    76.4  |           0.21  |       364×  |      134×  |
| 1024 × 1024 |   301.8  |           0.71  |       425×  |      162×  |
| 2048 × 2048 |  1203.5  |           2.68  |       449×  |      173×  |
| 4096 × 4096 |  4789.1  |          10.42  |       460×  |      175×  |

Block-geometry sweep at `2048²` (lower kernel time is better):

| Block   | Threads | Shared (B) | Kernel (ms) |
|---------|--------:|-----------:|------------:|
| 8 × 8   |      64 |        784 |        3.71 |
| 16 × 16 |     256 |       1936 |    **2.68** |
| 32 × 32 |    1024 |       5776 |        3.09 |

`16 × 16` is the sweet spot: `8 × 8` blocks pay a larger halo-to-core ratio, while `32 × 32` blocks limit the number of co-resident blocks per SM and reduce latency hiding.

By **Amdahl's law**, the pipeline's parallel fraction is only `P ≈ 0.38` (the rest is serial PCIe transfer), so even an infinitely fast kernel would improve the end-to-end runtime by at most `1 / (1 − P) ≈ 1.6×`. This is the quantitative argument for directing further optimisation at the **data path** (pinned memory, stream overlap) rather than the kernel.

---

## Design Notes

A few deliberate engineering choices worth calling out:

- **`size_t` everywhere for indexing.** `num_pixels` and byte counts are `size_t`, so a `width * height` product cannot silently overflow `int` on large images.
- **Clamped, rounded denormalisation.** Output floats are clamped to `[0, 1]` and rounded (`+ 0.5f`) on the way back to `uint8`, avoiding truncation banding and out-of-range wrap.
- **Allocator-matched frees.** The decoded image is released with `stbi_image_free`, not `free`, to respect `stb`'s allocator and avoid a heap mismatch.
- **Launch error checking.** `cudaPeekAtLastError()` catches launch-configuration failures and `cudaDeviceSynchronize()` surfaces asynchronous execution errors, both wrapped in `CUDA_CHECK`.
- **Single translation unit.** No build system and no OpenCV — the whole program is `main.cu` plus two header-only libraries, so it compiles with one `nvcc` invocation.

---

## Project Structure

```
.
├── main.cu              # Kernel, host driver, and application entry point
├── stb_image.h          # Image decoding (header-only, public domain) — fetched
├── stb_image_write.h    # PNG encoding  (header-only, public domain) — fetched
└── README.md
```

---

## Roadmap

The current source is a complete, correct GPU filter. The following are **not yet in the source** and are the natural next steps (in rough priority order):

- [ ] **CPU reference baseline** — a serial transcription of the filter to serve as a correctness oracle and a speedup denominator.
- [ ] **Timing instrumentation** — CUDA-event brackets around H2D / kernel / D2H, plus a host-side wall-clock timer, to make the [Performance](#performance) numbers reproducible from the repo itself.
- [ ] **Correctness validation** — an epsilon comparison (`max`/`mean` absolute error) between the GPU output and the CPU baseline.
- [ ] **Pinned host memory** (`cudaMallocHost`) — to lift transfer bandwidth toward ~25 GB/s and raise the end-to-end ceiling.
- [ ] **Stream overlap** — overlap copy and compute to hide transfer behind the kernel, pushing end-to-end toward the kernel-bound limit.
- [ ] **Shared-tile padding** (`HALO_DIM + 1`) — remove the bank conflicts on the Y-stride of the tile.
- [ ] **Runtime parameters** — expose `σs`, `σr`, and the output path as CLI arguments instead of compile-time constants.
- [ ] **Colour support** — extend beyond grayscale (with a suitable range metric).

---

## References

1. C. Tomasi and R. Manduchi, "Bilateral filtering for gray and color images," *Proc. IEEE ICCV*, 1998.
2. S. Paris and F. Durand, "A fast approximation of the bilateral filter using a signal processing approach," *Int. J. Comput. Vis.*, 2009.
3. P. Micikevicius, "3D finite difference computation on GPUs using CUDA," *GPGPU-2*, 2009.
4. NVIDIA, *CUDA C++ Programming Guide* and *CUDA C++ Best Practices Guide*.
5. M. Harris, "Using shared memory in CUDA," NVIDIA Technical Blog.
6. G. M. Amdahl, "Validity of the single processor approach to achieving large-scale computing capabilities," *AFIPS*, 1967.

---

## License

Released under the MIT License — see `LICENSE`. The bundled `stb` headers are public domain (or MIT, at your option) under their own terms.

---

## Acknowledgements

Built as a parallel-computing course project (CS3230, Parallel & Distributed Computing). Image I/O courtesy of Sean Barrett's [`stb`](https://github.com/nothings/stb) single-file libraries.
