// 目的: 統合 GPU で device から host へ結果を渡す 3 方式を比べる。
//   A. cudaMalloc + cudaMemcpy2D  (現在の実装)
// 公平に比べるため、いずれも host が全 byte を読み終えるまでを測る。
// 1 byte だけ読むと managed は 1 page しか移送せず、比較にならない。
//   B. cudaMallocManaged          (統合 GPU では移送不要のはず)
//   C. cudaHostAlloc(Mapped)      (zero copy)
// kernel が書いた内容を host が読めるまでの時間を測る。
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

__global__ void fill_kernel(std::uint8_t* p, std::size_t n, std::uint8_t v) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = static_cast<std::uint8_t>(v + (i & 31));
}

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s\n", prop.name);
    std::printf("  integrated=%d  canMapHostMemory=%d  managedMemory=%d\n",
                prop.integrated, prop.canMapHostMemory, prop.managedMemory);
    std::printf("  concurrentManagedAccess=%d  pageableMemoryAccess=%d  unifiedAddressing=%d\n",
                prop.concurrentManagedAccess, prop.pageableMemoryAccess, prop.unifiedAddressing);

    // 現在の hybrid が毎 frame 戻している量に合わせる。
    const std::size_t bytes = 1499u * 1024u;
    const int n = 200;
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((bytes + threads - 1) / threads);

    std::vector<double> ta, tb, tc;
    volatile std::uint64_t sink = 0;
    // A. cudaMalloc + cudaMemcpy2D 相当 (1 次元の cudaMemcpy で測る)
    {
        void* dev = nullptr; cudaMalloc(&dev, bytes);
        std::vector<std::uint8_t> host(bytes);
        for (int i = 0; i < n; ++i) {
            const auto t0 = clk::now();
            fill_kernel<<<blocks, threads>>>(static_cast<std::uint8_t*>(dev), bytes,
                                             static_cast<std::uint8_t>(i));
            cudaDeviceSynchronize();
            cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost);
            std::uint64_t sum = 0;
            for (std::size_t k = 0; k < bytes; ++k) sum += host[k];
            sink += sum;
            const auto t1 = clk::now();
            if (i >= 30) ta.push_back(ms(t0, t1));
        }
        std::printf("  検算 A: %llu\n", (unsigned long long)(sink % 1000));
        cudaFree(dev);
    }
    // B. cudaMallocManaged
    {
        void* man = nullptr;
        if (cudaMallocManaged(&man, bytes) == cudaSuccess) {
            std::uint8_t* p = static_cast<std::uint8_t*>(man);
            for (int i = 0; i < n; ++i) {
                const auto t0 = clk::now();
                fill_kernel<<<blocks, threads>>>(p, bytes, static_cast<std::uint8_t>(i));
                cudaDeviceSynchronize();
                // host から全 byte を読む。移送が要る実装ではここで発生する。
                std::uint64_t sum = 0;
                for (std::size_t k = 0; k < bytes; ++k) sum += p[k];
                sink += sum;
                const auto t1 = clk::now();
                if (i >= 30) tb.push_back(ms(t0, t1));
            }
            std::printf("  検算 B: %llu\n", (unsigned long long)(sink % 1000));
            cudaFree(man);
        }
    }
    // C. cudaHostAlloc(Mapped) zero copy
    {
        void* hostp = nullptr;
        if (cudaHostAlloc(&hostp, bytes, cudaHostAllocMapped) == cudaSuccess) {
            void* devp = nullptr;
            cudaHostGetDevicePointer(&devp, hostp, 0);
            std::uint8_t* h = static_cast<std::uint8_t*>(hostp);
            for (int i = 0; i < n; ++i) {
                const auto t0 = clk::now();
                fill_kernel<<<blocks, threads>>>(static_cast<std::uint8_t*>(devp), bytes,
                                                 static_cast<std::uint8_t>(i));
                cudaDeviceSynchronize();
                std::uint64_t sum = 0;
                for (std::size_t k = 0; k < bytes; ++k) sum += h[k];
                sink += sum;
                const auto t1 = clk::now();
                if (i >= 30) tc.push_back(ms(t0, t1));
            }
            std::printf("  検算 C: %llu\n", (unsigned long long)(sink % 1000));
            cudaFreeHost(hostp);
        }
    }
    std::printf("  %.2f MB を kernel が書き、host が全 byte を読み終えるまで:\n", bytes / 1048576.0);
    std::printf("    A cudaMalloc + cudaMemcpy  %7.3f ms\n", ta.empty()?-1:median(ta));
    std::printf("    B cudaMallocManaged        %7.3f ms\n", tb.empty()?-1:median(tb));
    std::printf("    C cudaHostAlloc(Mapped)    %7.3f ms\n", tc.empty()?-1:median(tc));
    return 0;
}
