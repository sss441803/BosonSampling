#pragma once

#include <helper.cu>

// Fill array with random floats
void random_init(float *data, size_t size);

// Fill array with random ints between a ranges
void random_init(int* array, size_t size, int low_limit, int high_limit);

// Obtain the indices of a sorted array where the entries increases. Used for charges.
__global__
void get_index(const int *charges, int *indices, const int size);

int* index_of_charge_increase(const int *charges, const int size);

// Checking if GPU computed results agree with the CPU results
bool check( const float *U,
            const float *A, const float *B,
            const float *LL, const float *LC, const float *LR,
            const float *C,
            int m, int n, int k);

// Turn shared memory pointer into uint32_t address.
__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr);

// Non-coheret load from global memory to register
__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard);

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard);

// Set global memory from register
__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard);

// Load from shared memory to register
__device__ __forceinline__
void lds32(float &reg, const uint32_t &addr);

__device__ __forceinline__
void lds32(int &reg, const uint32_t &addr);

__device__ __forceinline__
void lds64(float &reg0, float &reg1, const uint32_t &addr);

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr);

// Set shared memory from register
__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr);

__device__ __forceinline__
void sts64(const float &reg0, const float &reg1, const uint32_t &addr);

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr);