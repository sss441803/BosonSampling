#pragma once
#include <sort.cu>

// Obtain the indices of a sorted array where the entries increases. Used for charges.
__global__
void get_index(const int *charges, int *indices, const int size);

int* index_of_charge_increase(const int *charges, const int size);

SortedInfo sort(int *d_C, const int size);

// Fill array with random floats but each charge must occupy multiples of eight rows/columns
RemapInfo index_remapping(const int size, const int d, const int *index, const int *inc);