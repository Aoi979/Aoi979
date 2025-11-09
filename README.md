
# Aoi Kajitsu

Interested in **concurrent programming** and **parallel computing**.  
Mostly work with **C/C++** and **CUDA** on **Linux**, learning by building things from scratch.

**Focus:** C/C++ · CUDA · Concurrency · Parallelism · Linux  

---

## Contact

If you want to reach me, try decoding this with CUDA:

```cpp
__global__ void decode_email(char* out, const uint8_t* encoded, int N) {
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

int main() {
    const int N = 23;
    uint8_t encoded[N] = {
        24, 30, 64, 77, 40, 40, 80, 6, 77, 40, 43, 60,
        114, 83, 118, 84, 24, 64, 47, 57, 98, 30, 84
    };
    char* d_out;
    uint8_t* d_encoded;
    cudaMalloc(&d_out, N);
    cudaMalloc(&d_encoded, N);
    cudaMemcpy(d_encoded, encoded, N, cudaMemcpyHostToDevice);
    decode_email<<<1, N>>>(d_out, d_encoded, N);
    cudaDeviceSynchronize();
    char result[N + 1];
    cudaMemcpy(result, d_out, N, cudaMemcpyDeviceToHost);
    result[N] = '\0';
    std::cout << "Decoded string: " << result << "\n";
    cudaFree(d_out);
    cudaFree(d_encoded);
}
