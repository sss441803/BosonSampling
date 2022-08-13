#include <cstdio>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "../include/sort.cuh"

// Obtain the indices of a sorted array where the entries increases. Used for charges.
__global__
void get_index(const int size, const int *charges, int *indices) {
    int charge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (charge_idx < size -1) {
        int charge = charges[charge_idx];
        int next_charge = charges[charge_idx + 1];
        if (charge_idx == 0) {
            indices[charge] = 0;
        }
        if (charge != next_charge) {
            indices[next_charge] = charge_idx + 1;
        }
    }
}

int* index_of_charge_increase(const int d, const int size, const int *charges) {
    int *inc;
    cudaMalloc(&inc, d * sizeof(int));
    cudaMemset(inc, -1, d * sizeof(int));
    dim3 grid((size + 63) / 64);
    get_index<<<grid, 64>>>(size, charges, inc);
    return inc;
}

// Initialize a sequence
__global__
void sequence(int *indices, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        indices[idx] = idx;
    }
}

SortedInfo sort(const int d, const int size, int *d_C) {
    // initialize array of indices
    int* d_id;
    cudaMalloc(&d_id, size * sizeof(int));
    // set the elements of H to 0, 1, 2, 3, ...
    dim3 grid((size + 63) / 64);
    sequence<<<grid, 64>>>(d_id, size);
    // changing raw pointer to thrust device pointer
    thrust::device_ptr<int> t_C(d_C);
    thrust::device_ptr<int> t_id(d_id);
    // Sorting charges. Reordering indices given the sort. (keys = charges)
    thrust::sort_by_key(thrust::device, t_C, t_C + size, t_id);
    // Obtaining indices of charge increases
    int* inc = index_of_charge_increase(d, size, d_C);
    // // verification
    // int* h_C;
    // cudaMallocHost(&h_C, size * sizeof(int));
    // cudaMemcpy(h_C, d_C, size * sizeof(int), cudaMemcpyDefault);
    // int last_charge;
    // int current_charge = 0;
    // for (int c = 0; c < size; ++c) {
    //     last_charge = current_charge;
    //     current_charge = h_C[c];
    //     if (current_charge < last_charge) {
    //         printf("At c = %i charge decreased from %i to %i.\n", c, last_charge, current_charge);
    //         break;
    //     }
    //     if (c == size - 1) {
    //         printf("Sort succeeded.\n");
    //     }
    // }

    SortedInfo output;
    output.inc = inc;
    output.id = d_id;
    return output;
}

RemapInfo index_remapping(const int size, const int d, const int *index, const int *inc) {

    // Finding the index offset needed for each charge value
    int Offset = 0;
    int incidx = -1;
    int c = 0;
    int Offsets[d] = { 0 };
    int incNew[d];
    std::fill_n (incNew, d, -1);
    while (c < d) {    
        for (; c < d && incidx == -1; ++c) {
            incidx = inc[c];
            //printf("c: %i, inc: %i.\n", c, incidx);
        }
        if (incidx == -1) { incidx = size; }
        int OffsetAdd = (8 - (incidx + Offset) % 8) % 8;
        incidx = -1;
        Offset += OffsetAdd;
        Offsets[c-1] = Offset;
        if (inc[c-1] != -1) { incNew[c-1] = inc[c-1] + Offset; }
        //printf("c: %i, new_inc: %i, iOffset: %i.\n", c-1, new_inc[c-1], iOffsets[c-1]);
    }
    // Dimension of the new array to store the data
    int sizeNew = (((size + Offsets[d-1]) / 8) + 1) * 8;
    // printf("m: %i, iOffset %i, mNew: %i.\n", m, iOffsets[d-1], mNew);

    // Create a new array to hold the data
    int *cNew, *indexNew;
    cudaMallocHost(&cNew, sizeNew * sizeof(int));
    cudaMallocHost(&indexNew, sizeNew * sizeof(int));
    // Fill in a new charge array
    c = 0;
    int old_c = c;
    incidx = 0;
    for (int i = 0; i < sizeNew; ++i) {
        if (i == incidx) {
            old_c = c;
            c++;
            incidx = -1;
            for (; c < d && incidx <= 0; ++c) {
                incidx = incNew[c];
                if (incidx == 0) { old_c = c; }
            }
            c--;
            //printf("old_c: %i, c: %i, incidx: %i iOffset: %i.\n", old_c, c, incidx, iOffsets[c]);
        }
        cNew[i] = old_c;
    }
    // Fill in index mapping array
    c = 0;
    old_c = c;
    incidx = 0;
    for (int i = 0; i < size; ++i) {
        if (i == incidx) {
            old_c = c;
            c++;
            incidx = -1;
            for (; c < d && incidx <= 0; ++c) {
                incidx = inc[c];
                if (incidx == 0) { old_c = c; }
            }
            c--;
            //printf("old_c: %i, c: %i, incidx: %i iOffset: %i.\n", old_c, c, incidx, iOffsets[c]);
        }
        indexNew[i + Offsets[old_c]] = index[i] + 1; // 0th entry is reserved for 0 values
    }

    RemapInfo remap_info;
    remap_info.size = sizeNew;
    remap_info.index = indexNew;
    remap_info.inc = incNew;
    remap_info.c = cNew;

    return remap_info;
}