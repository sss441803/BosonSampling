#include <cstdio>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include <helper.h>

struct SortedInfo {
    int* inc;
    int* id;
};

SortedInfo sort(int *d_C, size_t size) {
    // initialize vectors of indices
    thrust::device_vector<int> id(size, 0);
    // set the elements of H to 0, 1, 2, 3, ...
    thrust::sequence(id.begin(), id.end());
    // changing raw pointer to thrust device pointer
    thrust::device_ptr<int> t_C(d_C);
    // Sorting charges. Reordering indices given the sort. (keys = charges)
    thrust::sort_by_key(thrust::device, t_C, t_C + size, id.begin());
    // Obtaining indices of charge increases
    int* inc = index_of_charge_increase(d_C, size);
    // changing thrust device vector to raw pointer
    int* d_id = thrust::raw_pointer_cast(id.data());
    // verification
    int* h_C;
    cudaMallocHost(&h_C, size * sizeof(int));
    cudaMemcpy(h_C, d_C, size * sizeof(int), cudaMemcpyDefault);
    int last_charge;
    int current_charge = 0;
    for (int c = 0; c < size; ++c) {
        last_charge = current_charge;
        current_charge = h_C[c];
        if (current_charge < last_charge) {
            printf("At c = %i charge decreased from %i to %i.\n", c, last_charge, current_charge);
            break;
        }
        if (c == size - 1) {
            printf("Sort succeeded.\n");
        }
    }

    SortedInfo output;
    output.inc = inc;
    output.id = d_id;
    return output;
}