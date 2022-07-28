#include <cstdio>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

// Obtain the indices of a sorted array where the entries increases. Used for charges.
__global__
void get_index(const int *charges, int *indices, const int size) {
    int charge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (charge_idx < size -1) {
        int charge = charges[charge_idx];
        int next_charge = charges[charge_idx + 1];
        if (charge != next_charge) {
            indices[next_charge] = charge_idx + 1;
        }
    }
}

int* index_of_charge_increase(const int *charges, const int size) {
    int *inc;
    cudaMalloc(&inc, size * sizeof(int));
    dim3 grid((size + 63) / 64);
    get_index<<<grid, 64>>>(charges, inc, size);
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

// Sorts charges and returns new indices
struct SortedInfo {
    int* inc;
    int* id;
};

SortedInfo sort(int *d_C, const int size) {
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
    int* inc = index_of_charge_increase(d_C, size);
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

// Fill array with random floats but each charge must occupy multiples of eight rows/columns
struct RemapInfo {
    int size;
    int *index;
    int *inc;
    int *c;
};

RemapInfo index_remapping(const int size, const int d, const int *index, const int *inc) {

    // Finding the index offset needed for each charge value
    int Offset = 0;
    int incIdx = 0;
    int c = 0;
    int Offsets[d] = { 0 };
    int incNew[d] = { 0 };
    int* cNew;
    cudaMallocHost(&cNew, (size + d*8) * sizeof(float));
    while (c <= d) {    
        for (; c <= d && incIdx == 0; ++c) {
            incIdx = inc[c];
            //printf("c: %i, inc: %i.\n", c, incIdx);
        }
        if (incIdx == 0) { incIdx = size; }
        int OffsetAdd = (8 - (incIdx + Offset) % 8) % 8;
        incIdx = 0;
        Offset += OffsetAdd;
        Offsets[c-1] = Offset;
        incNew[c-1] = inc[c-1] + Offset;
        for (int i = incNew[c - 2]; i < incNew[c - 1]; ++i) {
            cNew[i] = c - 2;
        }
        //printf("Offset: %i.\n", Offset);
    }
    // Dimension of the new array to store the data
    int sizeNew = (((size + Offsets[d-1]) / 8) + 1) * 8;
    //printf("size: %i, Offset %i, sizeNew: %i.\n", size, Offsets[d-1], sizeNew);
    for (int i = incNew[d]; i < sizeNew; ++i) {
        cNew[i] = d;
    }

    int* indexNew;
    cudaMallocHost(&indexNew, sizeNew * sizeof(float));
    c = 0;
    incIdx = 0;
    for (int i = 0; i < size; ++i) {
        if (i == incIdx) {
            c++;
            incIdx = 0;
            for (; c <= d + 1 && incIdx == 0; ++c) {
                incIdx = inc[c];
            }
            c--;
            //printf("c: %i, incIdx: %i Offset: %i.\n", c, incIdx, Offsets[c]);
        }
        //printf("index: %i\n", index[i]);
        indexNew[i + Offsets[c - 1]] = index[i] + 1; // 0th entry is reserved for 0 values
    }

    RemapInfo remap_info;
    remap_info.size = sizeNew;
    remap_info.index = indexNew;
    remap_info.inc = incNew;
    remap_info.c = cNew;

    return remap_info;
}