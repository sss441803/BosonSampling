#include <TwoDInit.cu>

// Fill array with random floats but each charge must occupy multiples of eight rows/columns
NewData random_init_2D(const int m, const int n, const int d, const int *inc1, const int *inc2);