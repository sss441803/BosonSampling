#include <cstdint>

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

/*
 * matrix Glc, Gcr and T: row-major
 *
 * mma block:
 * thread block tile: m128n128k8
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixGlc: 8x1 FP32
 *     matrixGcr: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                64
 *                    --|---------------------|
 *           Gcr_tile  8|                     |
 *                    --|---------------------|
 *
 *  Glc_tile | 8 |      |    32    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              16               ||
 *   Gcr_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * Glc_frag     | 2 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */
// CUDA kernel
extern "C" __global__ __launch_bounds__(256, 2)
void kernel(const int d,
            const int cc0,
            const int cc1,
            const float *U_r,
            const float *U_i,
            const float *Glc_r,
            const float *Glc_i,
            const float *Gcr_r,
            const float *Gcr_i,
            const int *CL0,
            const int *CL1,
            const int *CR0,
            const int *CR1,
            const int changes,
            const int *chcC0,
            const int *chcC1,
            const int *idxcC,
            float *T_r,
            float *T_i,
            uint32_t m,
            uint32_t n,
            uint32_t k,
            uint32_t Glc_ldg_step,    // k * sizeof(float)
            uint32_t Gcr_ldg_step) {  // n * sizeof(float) * 8
    /*
     * matrix Glc & Gcr thread block tile shared memory (double buffer)
     * matrix Glc: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
     * matrix Gcr: 64 * 8 * 4Byte/item * double buffer = 4KB
     *
     * for double buffer faster switch, Glc_smem requires 8KB * 2 shared memory
     * and 16KB aligned, Gcr_smem should be 8KB aligned, then the double buffer
     * can be switched by only 1 xor instruction:
     *     (uint32_t &)Glc_smem ^= 0x2000;
     *     (uint32_t &)Gcr_smem ^= 0x0800;
     */

// Shared memory declaration
    __shared__ __align__(32 * 1024) char smem[43 * 1024];
    float *Glc_r_smem = reinterpret_cast<float *>(smem);
    float *Glc_i_smem = reinterpret_cast<float *>(smem + 16 * 1024);
    float *Gcr_r_smem = reinterpret_cast<float *>(smem + 32 * 1024);
    float *Gcr_i_smem = reinterpret_cast<float *>(smem + 36 * 1024);
    int *chcC0_smem = reinterpret_cast<int *>(smem + 40 * 1024);
    int *chcC1_smem = reinterpret_cast<int *>(smem + 41 * 1024);
    int *idxcC_smem = reinterpret_cast<int *>(smem + 42 * 1024);

    // Glc, Gcr, U and T register fragment
    float Glc_r_frag[2][8];
    float Glc_i_frag[2][8];
    float Gcr_r_frag[2][4];
    float Gcr_i_frag[2][4];
    float U_r_frag[2][2];
    float U_i_frag[2][2];
    float T_r_frag[8][4];
    float T_i_frag[8][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            U_r_frag[i][j] = 0;
            U_i_frag[i][j] = 0;
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            T_r_frag[i][j] = 0;
            T_i_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;

    // 4x8 threads each warp for FFMA
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // Load left and right charge values to register
    int m_idx = blockIdx.y * 128 + warp_id / 2 * 32 + mma_tid_y * 4;
    int n_idx = blockIdx.x * 64 + warp_id % 2 * 32 + mma_tid_x * 2;
    int cl0[2], cr0[2], cl1[2], cr1[2];
    if (m_idx < m) { cl0[0] = CL0[m_idx]; }
    else { cl0[0] = -1; }
    if (m_idx + 16 < m) { cl0[1] = CL0[m_idx + 16]; }
    else { cl0[1] = -1; }
    if (n_idx < n) { cr0[0] = CR0[n_idx]; }
    else { cr0[0] = -1; }
    if (n_idx + 16 < n) { cr0[1] = CR0[n_idx + 16]; }
    else { cr0[1] = -1; }
    if (m_idx < m) { cl1[0] = CL1[m_idx]; }
    else { cl1[0] = -1; }
    if (m_idx + 16 < m) { cl1[1] = CL1[m_idx + 16]; }
    else { cl1[1] = -1; }
    if (n_idx < n) { cr1[0] = CR1[n_idx]; }
    else { cr1[0] = -1; }
    if (n_idx + 16 < n) { cr1[1] = CR1[n_idx + 16]; }
    else { cr1[1] = -1; }

    // Gamma tile ldg (load from global) pointer
    const char *Glc_r_ldg_ptr = (const char *)(
        Glc_r + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8); // 32 x 8
    const char *Glc_i_ldg_ptr = (const char *)(
        Glc_i + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8); // 32 x 8
    const char *Gcr_r_ldg_ptr = (const char *)(
        Gcr_r + (threadIdx.x / 32) * n + blockIdx.x * 64 + threadIdx.x % 32); // 8 x 32
    const char *Gcr_i_ldg_ptr = (const char *)(
        Gcr_i + (threadIdx.x / 32) * n + blockIdx.x * 64 + threadIdx.x % 32); // 8 x 32
    const char *chcC0_ldg_ptr = (const char *)(chcC0 + threadIdx.x);
    const char *chcC1_ldg_ptr = (const char *)(chcC1 + threadIdx.x);
    const char *idxcC_ldg_ptr = (const char *)(idxcC + threadIdx.x);

    // Gamma tile sts/lds (set shared memory/load shared memory) pointer
    // using uint32_t pointer for faster double buffer switch
    uint32_t Glc_r_sts_addr = smem_u32addr(
        Glc_r_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
    uint32_t Glc_i_sts_addr = smem_u32addr(
        Glc_i_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
    uint32_t Gcr_r_sts_addr = smem_u32addr(
        Gcr_r_smem + (threadIdx.x / 32) * 64 + (threadIdx.x % 32));
    uint32_t Gcr_i_sts_addr = smem_u32addr(
        Gcr_i_smem + (threadIdx.x / 32) * 64 + (threadIdx.x % 32));
    uint32_t Glc_r_lds_addr = smem_u32addr(
        Glc_r_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t Glc_i_lds_addr = smem_u32addr(
        Glc_i_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t Gcr_r_lds_addr = smem_u32addr(
        Gcr_r_smem + (warp_id % 2) * 32 + mma_tid_x * 2);
    uint32_t Gcr_i_lds_addr = smem_u32addr(
        Gcr_i_smem + (warp_id % 2) * 32 + mma_tid_x * 2);
    uint32_t chcC0_sts_addr = smem_u32addr(chcC0_smem + threadIdx.x);
    uint32_t chcC1_sts_addr = smem_u32addr(chcC1_smem + threadIdx.x);
    uint32_t idxcC_sts_addr = smem_u32addr(idxcC_smem + threadIdx.x);
    uint32_t chcC0_lds_addr = smem_u32addr(chcC0_smem);
    uint32_t chcC1_lds_addr = smem_u32addr(chcC1_smem);
    uint32_t idxcC_lds_addr = smem_u32addr(idxcC_smem);

    // ldg_guard to avoid LDG out of bound
    uint32_t Glc_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
        if (m_idx < m) {
            Glc_ldg_guard |= (1u << i);
        }
    }
    uint32_t Gcr_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n_idx = blockIdx.x * 64 + threadIdx.x % 32 + i * 32;
        if (n_idx < n) {
            Gcr_ldg_guard |= (1u << i);
        }
    }

    // Register to store values loaded from global memory before putting them into shared memory
    float Glc_r_ldg_reg[4];
    float Glc_i_ldg_reg[4];
    float Gcr_r_ldg_reg[2];
    float Gcr_i_ldg_reg[2];
    float chcC0_ldg_reg = 0;
    float chcC1_ldg_reg = 0;
    float idxcC_ldg_reg = 0;

    // 1'st Glc & Gcr tile loaded before the k_tile loop
    uint32_t k_tiles = (k + 7) / 8 - 1;
    uint32_t first_k_tile = k - k_tiles * 8;

    // load 1'st tile to shared memory
    { 
        // Glc
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (Glc_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            ldg32_nc_0(Glc_r_ldg_reg[i],
                       Glc_r_ldg_ptr + i * Glc_ldg_step,
                       guard);
            ldg32_nc_0(Glc_i_ldg_reg[i],
                       Glc_i_ldg_ptr + i * Glc_ldg_step,
                       guard);
        }
        sts128(Glc_r_ldg_reg[0], Glc_r_ldg_reg[1], Glc_r_ldg_reg[2], Glc_r_ldg_reg[3],
               Glc_r_sts_addr);
        sts128(Glc_i_ldg_reg[0], Glc_i_ldg_reg[1], Glc_i_ldg_reg[2], Glc_i_ldg_reg[3],
               Glc_i_sts_addr);
        // Gcr
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            bool guard = (Gcr_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 32 < first_k_tile;
            ldg32_nc_0(Gcr_r_ldg_reg[i],
                       Gcr_r_ldg_ptr + i * 32 * sizeof(float),
                       guard);
            ldg32_nc_0(Gcr_i_ldg_reg[i],
                       Gcr_i_ldg_ptr + i * 32 * sizeof(float),
                       guard);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            sts32(Gcr_r_ldg_reg[i], Gcr_r_sts_addr + i * 32 * sizeof(float));
            sts32(Gcr_i_ldg_reg[i], Gcr_i_sts_addr + i * 32 * sizeof(float));
        }
        bool guard = threadIdx.x < changes;
        ldg32_nc_0(chcC0_ldg_reg, chcC0_ldg_ptr, guard);
        sts32(chcC0_ldg_reg, chcC0_sts_addr);
        ldg32_nc_0(chcC1_ldg_reg, chcC1_ldg_ptr, guard);
        sts32(chcC1_ldg_reg, chcC1_sts_addr);
        ldg32_nc_0(idxcC_ldg_reg, idxcC_ldg_ptr, guard);
        sts32(idxcC_ldg_reg, idxcC_sts_addr);

        __syncthreads();

        // switch double buffer
        Glc_r_sts_addr ^= 0x2000;
        Glc_i_sts_addr ^= 0x2000;
        Gcr_r_sts_addr ^= 0x0800;
        Gcr_i_sts_addr ^= 0x0800;

        // ldg pointer for next tile
        Glc_r_ldg_ptr += first_k_tile * sizeof(float);
        Glc_i_ldg_ptr += first_k_tile * sizeof(float);
        Gcr_r_ldg_ptr += n * first_k_tile * sizeof(float);
        Gcr_i_ldg_ptr += n * first_k_tile * sizeof(float);
    }

    // load 1'st fragment
    lds128(Glc_r_frag[0][0], Glc_r_frag[0][1], Glc_r_frag[0][2], Glc_r_frag[0][3],
           Glc_r_lds_addr);
    lds128(Glc_i_frag[0][0], Glc_i_frag[0][1], Glc_i_frag[0][2], Glc_i_frag[0][3],
           Glc_i_lds_addr);
    lds128(Glc_r_frag[0][4], Glc_r_frag[0][5], Glc_r_frag[0][6], Glc_r_frag[0][7],
           Glc_r_lds_addr + 16 * sizeof(float));
    lds128(Glc_i_frag[0][4], Glc_i_frag[0][5], Glc_i_frag[0][6], Glc_i_frag[0][7],
           Glc_i_lds_addr + 16 * sizeof(float));
    lds64(Gcr_r_frag[0][0], Gcr_r_frag[0][1],
          Gcr_r_lds_addr);
    lds64(Gcr_i_frag[0][0], Gcr_i_frag[0][1],
          Gcr_i_lds_addr);
    lds64(Gcr_r_frag[0][2], Gcr_r_frag[0][3],
          Gcr_r_lds_addr + 16 * sizeof(float));
    lds64(Gcr_i_frag[0][2], Gcr_i_frag[0][3],
          Gcr_i_lds_addr + 16 * sizeof(float));
    // Load the center singular value lambda center
    int c = 0;
    // Load U
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (cl0[i] >= 0 && 0 >= cr0[j] && cl0[i] != -1 && cr0[j] != -1 && cl1[i] >= 0 && 0 >= cr1[j] && cl1[i] != -1 && cr1[j] != -1) {
                U_r_frag[i][j] = U_r[((cl0[i] - cc0) * d + cl1[i] - cc1) * d * d * d * d 
                                    + ((cc0 - cr0[j]) * d + cc1 - cr1[j]) * d * d
                                    + cl0[i] * d + cl1[i]];
                U_i_frag[i][j] = U_i[((cl0[i] - cc0) * d + cl1[i] - cc1) * d * d * d * d 
                                    + ((cc0 - cr0[j]) * d + cc1 - cr1[j]) * d * d
                                    + cl0[i] * d + cl1[i]];
            }
            else { U_r_frag[i][j] = U_i_frag[i][j] = 0; }
        }
    }
    __syncthreads();
    // Find the next center charge and index that is higher than the current charge
    /* Charge array:
    *  |---|---|---|---|---|---|---|---|---|
    *  | 0 | 0 | 0 | 2 | 2 | 3 | 4 | 6 | 6 |  ...
    *  |---|---|---|---|---|---|---|---|---|
    *  incC array: If no indices corresponds to a charge value, incC = -1 at that charge value
    *  |---|---|---|---|---|---|---|
    *  | 0 |-1 | 3 | 5 | 6 |-1 | 8 | ...
    *  |---|---|---|---|---|---|---|            */
    c = 0;
    int charge_id = 0;
    int next_c = 0;

    // k_tiles loop
    for (int k_tile = k_tiles; k_tile > 0; --k_tile) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            if (k_tile < k_tiles || k_frag < first_k_tile){
                // Load next center charge increment index
                if (c == next_c) {
                    int charge_local_0;
                    int charge_local_1;
                    lds32(charge_local_0, chcC0_lds_addr + charge_id * sizeof(float));
                    lds32(charge_local_1, chcC1_lds_addr + charge_id * sizeof(float));
                    charge_id += 1;
                    if (charge_id < changes) {
                        lds32(next_c, idxcC_lds_addr + charge_id * sizeof(float));
                    }
                    __syncthreads();
                    // Load needed U fragments
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            if (cl0[i] >= charge_local_0 && charge_local_0 >= cr0[j] && cl0[i] != -1 && cr0[j] != -1 && cl1[i] >= charge_local_1 && charge_local_1 >= cr1[j] && cl1[i] != -1 && cr1[j] != -1) {
                                U_r_frag[i][j] = U_r[((cl0[i] - cc0) * d + cl1[i] - cc1) * d * d * d * d 
                                                    + ((cc0 - cr0[j]) * d + cc1 - cr1[j]) * d * d
                                                    + (cl0[i] - charge_local_0) * d + cl1[i] - charge_local_1];
                                U_i_frag[i][j] = U_i[((cl0[i] - cc0) * d + cl1[i] - cc1) * d * d * d * d 
                                                    + ((cc0 - cr0[j]) * d + cc1 - cr1[j]) * d * d
                                                    + (cl0[i] - charge_local_0) * d + cl1[i] - charge_local_1];
                            }
                            else { U_r_frag[i][j] = U_i_frag[i][j] = 0; }
                        }
                    }
                    __syncthreads();
                }
                // Increase c
                c += 1;
            }

            // store next Glc&Gcr tile to shared memory
            if (k_frag == 7) {
                sts128(Glc_r_ldg_reg[0], Glc_r_ldg_reg[1], Glc_r_ldg_reg[2], Glc_r_ldg_reg[3],
                       Glc_r_sts_addr);
                sts128(Glc_i_ldg_reg[0], Glc_i_ldg_reg[1], Glc_i_ldg_reg[2], Glc_i_ldg_reg[3],
                       Glc_i_sts_addr);
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    sts32(Gcr_r_ldg_reg[i], Gcr_r_sts_addr + i * 32 * sizeof(float));
                    sts32(Gcr_i_ldg_reg[i], Gcr_i_sts_addr + i * 32 * sizeof(float));
                }
                __syncthreads();
                // switch double buffer
                Glc_r_lds_addr ^= 0x2000;
                Glc_i_lds_addr ^= 0x2000;
                Gcr_r_lds_addr ^= 0x0800;
                Gcr_i_lds_addr ^= 0x0800;
                Glc_r_sts_addr ^= 0x2000;
                Glc_i_sts_addr ^= 0x2000;
                Gcr_r_sts_addr ^= 0x0800;
                Gcr_i_sts_addr ^= 0x0800;
                // ldg pointer for next tile
                Glc_r_ldg_ptr += 8 * sizeof(float);
                Glc_i_ldg_ptr += 8 * sizeof(float);
                Gcr_r_ldg_ptr += Gcr_ldg_step;
                Gcr_i_ldg_ptr += Gcr_ldg_step;
            }

            // load next Glc&Gcr fragment from shared memory to register
            lds128(Glc_r_frag[(k_frag + 1) % 2][0],
                   Glc_r_frag[(k_frag + 1) % 2][1],
                   Glc_r_frag[(k_frag + 1) % 2][2],
                   Glc_r_frag[(k_frag + 1) % 2][3],
                   Glc_r_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(Glc_i_frag[(k_frag + 1) % 2][0],
                   Glc_i_frag[(k_frag + 1) % 2][1],
                   Glc_i_frag[(k_frag + 1) % 2][2],
                   Glc_i_frag[(k_frag + 1) % 2][3],
                   Glc_i_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(Glc_r_frag[(k_frag + 1) % 2][4],
                   Glc_r_frag[(k_frag + 1) % 2][5],
                   Glc_r_frag[(k_frag + 1) % 2][6],
                   Glc_r_frag[(k_frag + 1) % 2][7],
                   Glc_r_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(Glc_i_frag[(k_frag + 1) % 2][4],
                   Glc_i_frag[(k_frag + 1) % 2][5],
                   Glc_i_frag[(k_frag + 1) % 2][6],
                   Glc_i_frag[(k_frag + 1) % 2][7],
                   Glc_i_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds64(Gcr_r_frag[(k_frag + 1) % 2][0],
                  Gcr_r_frag[(k_frag + 1) % 2][1],
                  Gcr_r_lds_addr + (k_frag + 1) % 8 * 64 * sizeof(float));
            lds64(Gcr_i_frag[(k_frag + 1) % 2][0],
                  Gcr_i_frag[(k_frag + 1) % 2][1],
                  Gcr_i_lds_addr + (k_frag + 1) % 8 * 64 * sizeof(float));
            lds64(Gcr_r_frag[(k_frag + 1) % 2][2],
                  Gcr_r_frag[(k_frag + 1) % 2][3],
                  Gcr_r_lds_addr + ((k_frag + 1) % 8 * 64 + 16) * sizeof(float));
            lds64(Gcr_i_frag[(k_frag + 1) % 2][2],
                  Gcr_i_frag[(k_frag + 1) % 2][3],
                  Gcr_i_lds_addr + ((k_frag + 1) % 8 * 64 + 16) * sizeof(float));

            // load next Glc&Gcr tile
            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(Glc_r_ldg_reg[i],
                             Glc_r_ldg_ptr + i * Glc_ldg_step,
                             (Glc_ldg_guard & (1u << i)) != 0);
                    ldg32_nc(Glc_i_ldg_reg[i],
                             Glc_i_ldg_ptr + i * Glc_ldg_step,
                             (Glc_ldg_guard & (1u << i)) != 0);
                }

                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ldg32_nc(Gcr_r_ldg_reg[i],
                             Gcr_r_ldg_ptr + i * 32 * sizeof(float),
                             (Gcr_ldg_guard & (1u << i)) != 0);
                    ldg32_nc(Gcr_i_ldg_reg[i],
                             Gcr_i_ldg_ptr + i * 32 * sizeof(float),
                             (Gcr_ldg_guard & (1u << i)) != 0);
                }
            }
            
            // 
            if (k_tile < k_tiles || k_frag < first_k_tile){
                // FFMA loop
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        T_r_frag[i][j] += U_r_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j] - Glc_i_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j]) - U_i_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j] + Glc_i_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j]);
                        T_i_frag[i][j] += U_r_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j] + Glc_i_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j]) + U_i_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j] - Glc_i_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j]);
                    }
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        // Load next center charge increment index
        if (c == next_c) {
            int charge_local_0;
            int charge_local_1;
            lds32(charge_local_0, chcC0_lds_addr + charge_id * sizeof(float));
            lds32(charge_local_1, chcC1_lds_addr + charge_id * sizeof(float));
            charge_id += 1;
            if (charge_id < changes) {
                lds32(next_c, idxcC_lds_addr + charge_id * sizeof(float));
            }
            __syncthreads();
            // Load needed U fragments
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    if (cl0[i] >= charge_local_0 && charge_local_0 >= cr0[j] && cl0[i] != -1 && cr0[j] != -1 && cl1[i] >= charge_local_1 && charge_local_1 >= cr1[j] && cl1[i] != -1 && cr1[j] != -1) {
                        U_r_frag[i][j] = U_r[((cl0[i] - cc0) * d + cl1[i] - cc1) * d * d * d * d 
                                            + ((cc0 - cr0[j]) * d + cc1 - cr1[j]) * d * d
                                            + (cl0[i] - charge_local_0) * d + cl1[i] - charge_local_1];
                        U_i_frag[i][j] = U_i[((cl0[i] - cc0) * d + cl1[i] - cc1) * d * d * d * d 
                                            + ((cc0 - cr0[j]) * d + cc1 - cr1[j]) * d * d
                                            + (cl0[i] - charge_local_0) * d + cl1[i] - charge_local_1];
                    }
                    else { U_r_frag[i][j] = U_i_frag[i][j] = 0; }
                }
            }
            __syncthreads();
        }
        // Increase c
        c += 1;

        if (k_frag < 7) {
            // load next Glc&Gcr fragment from shared memory to register
            lds128(Glc_r_frag[(k_frag + 1) % 2][0],
                   Glc_r_frag[(k_frag + 1) % 2][1],
                   Glc_r_frag[(k_frag + 1) % 2][2],
                   Glc_r_frag[(k_frag + 1) % 2][3],
                   Glc_r_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(Glc_i_frag[(k_frag + 1) % 2][0],
                   Glc_i_frag[(k_frag + 1) % 2][1],
                   Glc_i_frag[(k_frag + 1) % 2][2],
                   Glc_i_frag[(k_frag + 1) % 2][3],
                   Glc_i_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(Glc_r_frag[(k_frag + 1) % 2][4],
                   Glc_r_frag[(k_frag + 1) % 2][5],
                   Glc_r_frag[(k_frag + 1) % 2][6],
                   Glc_r_frag[(k_frag + 1) % 2][7],
                   Glc_r_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(Glc_i_frag[(k_frag + 1) % 2][4],
                   Glc_i_frag[(k_frag + 1) % 2][5],
                   Glc_i_frag[(k_frag + 1) % 2][6],
                   Glc_i_frag[(k_frag + 1) % 2][7],
                   Glc_i_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds64(Gcr_r_frag[(k_frag + 1) % 2][0],
                  Gcr_r_frag[(k_frag + 1) % 2][1],
                  Gcr_r_lds_addr + (k_frag + 1) % 8 * 64 * sizeof(float));
            lds64(Gcr_i_frag[(k_frag + 1) % 2][0],
                  Gcr_i_frag[(k_frag + 1) % 2][1],
                  Gcr_i_lds_addr + (k_frag + 1) % 8 * 64 * sizeof(float));
            lds64(Gcr_r_frag[(k_frag + 1) % 2][2],
                  Gcr_r_frag[(k_frag + 1) % 2][3],
                  Gcr_r_lds_addr + ((k_frag + 1) % 8 * 64 + 16) * sizeof(float));
            lds64(Gcr_i_frag[(k_frag + 1) % 2][2],
                  Gcr_i_frag[(k_frag + 1) % 2][3],
                  Gcr_i_lds_addr + ((k_frag + 1) % 8 * 64 + 16) * sizeof(float));
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                T_r_frag[i][j] += U_r_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j] - Glc_i_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j]) - U_i_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j] + Glc_i_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j]);
                T_i_frag[i][j] += U_r_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j] + Glc_i_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j]) + U_i_frag[i / 4][j / 2] * (Glc_r_frag[k_frag % 2][i] * Gcr_r_frag[k_frag % 2][j] - Glc_i_frag[k_frag % 2][i] * Gcr_i_frag[k_frag % 2][j]);
            }
        }
    }

    // T_tile write back, reuse Glc&Gcr tile shared memory buffer
    uint32_t T_r_sts_addr = smem_u32addr((float2 *)(smem + warp_id * 1024) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    uint32_t T_i_sts_addr = smem_u32addr((float2 *)(smem + 8192 + warp_id * 1024) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *T_r_lds_ptr = (float *)(smem + warp_id * 1024) + lane_id;
    const float *T_i_lds_ptr = (float *)(smem + 8192 + warp_id * 1024) + lane_id;

    m_idx = blockIdx.y * 128 + warp_id / 2 * 32 + lane_id / 16;
    n_idx = blockIdx.x * 64 + warp_id % 2 * 32 + lane_id % 16;

    float *T_r_stg_ptr = T_r + m_idx * n + n_idx;
    float *T_i_stg_ptr = T_i + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else { 
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;
        uint32_t m_guard;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            m_guard = m < m_idx + 16 * i ? 0 : m - (m_idx + 16 * i);
            m_guard = (m_guard + 1) / 2;

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts64(T_r_frag[i * 4 + p][j * 2],
                        T_r_frag[i * 4 + p][j * 2 + 1],
                        T_r_sts_addr + p * 8 * sizeof(float2));
                    sts64(T_i_frag[i * 4 + p][j * 2],
                        T_i_frag[i * 4 + p][j * 2 + 1],
                        T_i_sts_addr + p * 8 * sizeof(float2));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 8; ++p) {
                    stg32(T_r_lds_ptr[p * 32],
                        T_r_stg_ptr + (i * 16 + p * 2) * n + j * 16,
                        p < m_guard && j * 16 < n_guard);
                    stg32(T_i_lds_ptr[p * 32],
                        T_i_stg_ptr + (i * 16 + p * 2) * n + j * 16,
                        p < m_guard && j * 16 < n_guard);
                }
            }
        }
    }
}