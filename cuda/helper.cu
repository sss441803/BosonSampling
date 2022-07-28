#include <cstdio>
#include <cstdint>

// Fill array with random floats
void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

// Fill array with random ints between a ranges
void random_init(int* array, size_t size, int low_limit, int high_limit)
{
    int range = high_limit - low_limit + 1;
    for(size_t i=0;i<size;i++)
    {
        array[i] = rand()%range + low_limit;
    }
}

// Checking if GPU computed results agree with the CPU results
bool check( const float *U,
            const float *A,
            const float *B,
            const float *LL,
            const float *LC,
            const float *LR,
            const float *C,
            int m, int n, int k) {

    for (int i = m-1; i >= max(m-100,0); --i) {
        for (int j = n-1; j >= max(n-100,0); --j) {
            float result = 0.f;
            for (int p = 0; p < k; ++p) {
                result += A[i * k + p] * B[j + p * n] * LC[p];
            }
            result *= LL[i] * LR[j];

            if (std::fabs(result - C[i * n + j]) / std::fabs(result) > 1e-5f) {
                printf("Size %i %i C[%d][%d] not match, %f vs %f\n", m, n, i, j, result, C[i * n + j]);
                return false;
            }
        }
    }
    printf("Size %i %i match\n", m, n);
    return true;
}

// Turn shared memory pointer into uint32_t address.
__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

// Non-coheret load from global memory to register
__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

// Set global memory from register
__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

// Load from shared memory to register
__device__ __forceinline__
void lds32(float &reg,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.f32 {%0}, [%1];\n"
        : "=f"(reg)
        : "r"(addr)
    );
}

__device__ __forceinline__
void lds32(int &reg,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.s32 {%0}, [%1];\n"
        : "=r"(reg)
        : "r"(addr)
    );
}

__device__ __forceinline__
void lds64(float &reg0, float &reg1,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v2.f32 {%0, %1}, [%2];\n"
        : "=f"(reg0), "=f"(reg1)
        : "r"(addr)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

// Set shared memory from register
__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts64(const float &reg0, const float &reg1,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v2.f32 [%0], {%1, %2};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}