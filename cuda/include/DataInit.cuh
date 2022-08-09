#pragma once

// Fill array with random floats
void random_init(float *data, size_t size);

// Fill array with random ints between a ranges
void random_init(int* array, size_t size, int low_limit, int high_limit);

struct NewData {
    int m;
    int k;
    int n;
    float* data;
};

// Fill array with random floats but each left charge must occupy multiples of eight rows
NewData left_align_init_1d(const int m, const int d, const int *inc1);

// Fill array with random floats but each left charge must occupy multiples of eight rows
NewData left_align_init(const int m, const int k, const int d, const int *inc1);

// Fill array with random floats but each right charge must occupy multiples of eight columns
NewData right_align_init_1d(const int n, const int d, const int *inc2);

// Fill array with random floats but each right charge must occupy multiples of eight columns
NewData right_align_init(const int k, const int n, const int d, const int *inc2);

// Fill array with random floats but each charge must occupy multiples of eight rows/columns
NewData random_init_2D(const int m, const int n, const int d, const int *inc1, const int *inc2);