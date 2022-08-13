#pragma once

// Sorts charges and returns new indices
struct SortedInfo {
    int* inc;
    int* id;
};

// Fill array with random floats but each charge must occupy multiples of eight rows/columns
struct RemapInfo {
    int size;
    int *index;
    int *inc;
    int *c;
};

SortedInfo sort(const int d, const int size, int *d_C);

// Fill array with random floats but each charge must occupy multiples of eight rows/columns
RemapInfo index_remapping(const int size, const int d, const int *index, const int *inc);