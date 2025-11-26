
# Aoi Kajitsu

Interested in **concurrent programming** and **parallel computing**.  
Mostly work with **C/C++** and **CUDA** on **Linux**, learning by building things from scratch.

**Focus:** C/C++ · CUDA · Concurrency · Parallelism · Linux  

Still learning - aiming for high-performance systems with C/C++ and CUDA, yet my talent seems strictly single-threaded.
---

## Contact

If you want to reach me, try decoding this with CUDA or C++20:

```cpp
constexpr char decode_char(int y) {
    constexpr int A_inv = 45;
    constexpr int B = 19;
    constexpr int K = 128;
    return static_cast<char>((A_inv * ((y + K) - B)) % K);
}

template<int... Values>
constexpr auto decode_email_cpp(std::integer_sequence<int, Values...>) {
    return std::array<char, sizeof...(Values)>{ decode_char(Values)... };
}

constexpr auto email_cpp = decode_email_cpp(std::integer_sequence<int,
    24, 30, 64, 77, 40, 40, 80, 6, 77, 40, 43, 60,
    114, 83, 118, 84, 24, 64, 47, 57, 98, 30, 84>{});

#ifdef USE_CUDA
__global__ void decode_email_cuda(char* out, const uint8_t* encoded, int N) {
    const uint32_t A_inv = 45; 
    const uint32_t B = 19;
    const uint32_t K = 128;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        uint32_t y = encoded[idx];
        uint32_t x = (A_inv * ((y + K) - B)) % K;
        out[idx] = static_cast<char>(x);
    }
}
#endif

int main() {
    const int N = 23;
    uint8_t encoded[N] = {
        24, 30, 64, 77, 40, 40, 80, 6, 77, 40, 43, 60,
        114, 83, 118, 84, 24, 64, 47, 57, 98, 30, 84
    };

#ifdef USE_CUDA
    char* d_out;
    uint8_t* d_encoded;
    cudaMalloc(&d_out, N);
    cudaMalloc(&d_encoded, N);
    cudaMemcpy(d_encoded, encoded, N, cudaMemcpyHostToDevice);
    decode_email_cuda<<<1, N>>>(d_out, d_encoded, N);
    cudaDeviceSynchronize();
    char result[N + 1];
    cudaMemcpy(result, d_out, N, cudaMemcpyDeviceToHost);
    result[N] = '\0';
    std::cout << result << "\n";
    cudaFree(d_out);
    cudaFree(d_encoded);
#else
    for (char c : email_cpp) std::cout << c;
    std::cout << '\n';
#endif

    return 0;
}
