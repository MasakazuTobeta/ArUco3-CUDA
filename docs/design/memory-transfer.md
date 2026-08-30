# Passing memory between host and device

## Purpose

This document measures how host and device share memory on the three target machines and records which method to choose. It shows, using device properties and measured values, the conditions under which the intuition that "transfers are unnecessary on an integrated GPU" holds and the conditions under which it does not.

## Scope

The scope is the route from a kernel writing a result to the host reading it. Input transfers from host to device follow the same properties, but the measurements were made on the result readback.

## Why transfers still occur on an integrated GPU

On an integrated GPU, host and device use the same physical memory. Even so, a region allocated with `cudaMalloc` and ordinary host memory are **separate allocations**, and `cudaMemcpy` really does copy. It is only a copy within the same DRAM, but it still costs something.

To skip the transfer, you must use an allocation method through which host and device can both reference the same region.

| Method | Allocation | Property |
| --- | --- | --- |
| A. Device memory + explicit copy | `cudaMalloc` + `cudaMemcpy` | A copy always occurs |
| B. Managed memory | `cudaMallocManaged` | Both sides reference the same pointer. Whether migration occurs depends on the device's properties |
| C. Mapped host memory | `cudaHostAlloc(cudaHostAllocMapped)` | The device references host memory directly |

## Properties of the target machines

These are measured values from `cudaGetDeviceProperties`.

| Machine | `integrated` | `canMapHostMemory` | `managedMemory` | `concurrentManagedAccess` | `pageableMemoryAccess` |
| --- | --- | --- | --- | --- | --- |
| DGX Spark GB10 | 1 | 1 | 1 | **1** | **1** |
| Jetson AGX Orin | 1 | 1 | 1 | **0** | **0** |
| GeForce RTX 5070 Ti | 0 | 1 | 1 | 1 | 1 |

**`integrated` alone is not enough to decide.** The Jetson Orin is an integrated GPU, yet its `concurrentManagedAccess` is 0. In that case managed memory is attached to the device side, and migration occurs when the host touches it. Even though the same physical memory is shared, the driver's unit of management is separate.

On the DGX Spark GB10, `pageableMemoryAccess` is 1, so the device can access ordinary host memory directly.

## Measurements

These are the times for a kernel to write 1.46 MB and for the host to **finish reading every byte**. Median over 170 runs after 30 warmup runs.

| Machine | A. Copy | B. Managed | C. Mapped |
| --- | --- | --- | --- |
| DGX Spark GB10 | 0.170 ms | **0.110 ms** | 0.128 ms |
| Jetson AGX Orin | 2.430 ms | 2.156 ms | **2.065 ms** |
| GeForce RTX 5070 Ti | 0.402 ms | **4.831 ms** | 0.396 ms |

These values include the time for the host to scan 1.46 MB. The host reads the results under every method, so that portion is a shared lower bound. The difference between the methods appears in the part that gets the data ready to read.

Measuring the copy stage alone gives the following on the DGX Spark.

| Stage | Time |
| --- | --- |
| Kernel + synchronization only (with B and C the host can read at this point) | 0.048 ms |
| + `cudaMemcpy` (A) | 0.120 ms |

**The copy itself is 0.071 ms**, and on an integrated GPU B or C eliminates it entirely.

## Choice per machine

| Machine | Recommendation | Reason |
| --- | --- | --- |
| DGX Spark GB10 | B (managed) | No migration occurs and no copy is needed. `pageableMemoryAccess` is 1, so ordinary memory can be used as well |
| Jetson AGX Orin | C (mapped) | `concurrentManagedAccess` is 0, so managed memory involves migration. Zero copy avoids it |
| GeForce RTX 5070 Ti | A or C | A discrete GPU, so there is no free sharing. **B is 12x slower** |

B is extremely slow on the RTX 5070 Ti because pages migrate from device to host every time the host reads. Choosing managed memory on a discrete GPU because it "is convenient" is substantially slower than an explicit copy.

## Difference from the current implementation

The hybrid route of plan C uses method A and returns 1499 KB per frame through 8 synchronous transfers (5 pyramid levels and 3 thresholded images). The measured time on the DGX Spark was 0.269 ms.

The gap against the 0.071 ms measured for a single copy is **the per-call cost of each of the 8 calls** (about 25 µs each). There are two opportunities for improvement.

1. Use B or C on an integrated GPU and eliminate the copy itself
2. Put the 8 blocking calls on a stream and batch them asynchronously

## Measurement pitfalls

**When measuring managed memory, you must read the entire allocated region.**

In the first measurement only 1 byte was read, and managed came out 21x faster than the copy on the RTX 5070 Ti. Managed memory migrates only the pages it needs, lazily, so reading 1 byte migrated only 1 page. Once corrected to read every byte, it turned out to be 12x slower instead.

For the same reason, measuring managed memory without `cudaMemPrefetchAsync` depends strongly on the host-side access pattern. When comparing, read as much as the actual use does.

## Open questions

- Whether to run the same comparison for input (host to device). Only the result readback is currently measured.
- The behavior of managed memory when combined with `cudaMemPrefetchAsync`. Whether it improves on a discrete GPU is unmeasured.
- How much slower kernel-side access becomes when mapped host memory is used on the Jetson. Zero copy is, from the kernel's point of view, access to host memory.
- Whether a `cv::Mat` buffer can be handed to the device as is when `pageableMemoryAccess` is 1 on an integrated GPU. It is possible in principle on the DGX Spark, but unverified.

## On an integrated GPU the page cache eats into the "device's free memory"

On the DGX Spark GB10, `ctest -j 8` began failing intermittently. In 5 out of 15 runs, a different test each time failed with `out of memory`. The cause was **the environment, not the implementation**.

```
GPU:  free 3513.6 MiB / total 122572.2 MiB
host: MemTotal 119 GB / MemFree 5 GB / MemAvailable 107 GB / Cached 100 GB
```

On an integrated GPU, device memory is the same memory as host memory. The "free" figure returned by `cudaMemGetInfo` **corresponds to `MemFree`, not `MemAvailable`**. The page cache is reclaimable, but a CUDA allocation fails without waiting for that reclamation.

Over a long working session, the page cache grows to 100 GB from build output, corpora, and docker layers. At that point even a single process can allocate only 3.5 GB, and lining up processes as `ctest -j 8` does exhausts it easily.

Reclaiming it brings the memory back.

```
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

In measurement it went from 3.5 GB back to **108.9 GB**, and `ctest -j 8` passed 10 times in a row.

**Drop the page cache before measuring.** Measuring without doing so causes not only allocation failures but also fluctuation in the measured values themselves, from contention over host-side memory bandwidth.

## See also

- [Detection pipeline design](detector-pipeline.md)
- [Evaluation plan](../evaluation-plan.md)
- [Benchmark report](../benchmark-report.md)
- [Architecture](../architecture.md)
