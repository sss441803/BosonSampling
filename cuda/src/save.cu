#include<string>
#include"cnpy.h"

void save(std::string fname, float* data, unsigned int size1) {
    cnpy::npy_save(fname, &data[0], {size1}, "w");
}

void save(std::string fname, float* data, unsigned int size1, unsigned int size2) {
    cnpy::npy_save(fname, &data[0], {size1, size2}, "w");
}

void save(std::string fname, float* data, unsigned int size1, unsigned int size2, unsigned int size3) {
    cnpy::npy_save(fname, &data[0], {size1, size2, size3}, "w");
}

void save(std::string fname, int* data, unsigned int size1) {
    cnpy::npy_save(fname, &data[0], {size1}, "w");
}

void save(std::string fname, int* data, unsigned int size1, unsigned int size2) {
    cnpy::npy_save(fname, &data[0], {size1, size2}, "w");
}

void save(std::string fname, int* data, unsigned int size1, unsigned int size2, unsigned int size3) {
    cnpy::npy_save(fname, &data[0], {size1, size2, size3}, "w");
}