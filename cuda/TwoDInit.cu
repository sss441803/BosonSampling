#include <cstdio>

struct NewData {
    int m;
    int n;
    float* data;
};

// Fill array with random floats but each charge must occupy multiples of eight rows/columns
NewData random_init_2D(const int m, const int n, const int d, const int *inc1, const int *inc2) {

    // Finding the index offset needed for each charge value
    int iOffset = 0;
    int inc1idx = 0;
    int c1 = 0;
    int iOffsets[d] = { 0 };
    int new_inc1[d] = { 0 };
    while (c1 < d) {    
        for (; c1 < d && inc1idx == 0; ++c1) {
            inc1idx = inc1[c1];
            printf("c1: %i, inc: %i.\n", c1, inc1idx);
        }
        if (inc1idx == 0) { inc1idx = m; }
        int iOffsetAdd = (8 - (inc1idx + iOffset) % 8) % 8;
        inc1idx = 0;
        iOffset += iOffsetAdd;
        iOffsets[c1-1] = iOffset;
        new_inc1[c1-1] = inc1[c1-1] + iOffset;
        printf("iOffset: %i.\n", iOffset);
    }
    // Dimension of the new array to store the data
    int mNew = (((m + iOffsets[d-1]) / 8) + 1) * 8;
    printf("m: %i, iOffset %i, mNew: %i.\n", m, iOffsets[d-1], mNew);

    // Finding the index offset needed for each charge value
    int jOffset = 0;
    int inc2idx = 0;
    int c2 = 0;
    int jOffsets[d] = { 0 };
    int new_inc2[d] = { 0 };
    while (c2 < d) {
        for (; c2 < d && inc2idx == 0; ++c2) {
            inc2idx = inc2[c2];
            printf("c2: %i, inc: %i.\n", c2, inc2idx);
        }
        if (inc2idx == 0) { inc2idx = n; }
        int jOffsetAdd = (8 - (inc2idx + jOffset) % 8) % 8;
        inc2idx = 0;
        jOffset += jOffsetAdd;
        jOffsets[c2-1] = jOffset;
        new_inc1[c1-1] = inc1[c1-1] + iOffset;
        printf("jOffset: %i.\n", jOffset);
    }
    // Dimension of the new array to store the data
    int nNew = (((n + jOffsets[d-1]) / 8) + 1) * 8;
    printf("n: %i, jOffset %i, nNew: %i.\n", n, jOffsets[d-1], nNew);

    // Create a new array to hold the data
    float *data;
    cudaMallocHost(&data, mNew * nNew * sizeof(float));
    // Fill in a new array
    c1 = 0;
    inc1idx = 0;
    for (int i = 0; i < m; ++i) {
        if (i == inc1idx) {
            c1++;
            inc1idx = 0;
            for (; c1 <= d && inc1idx == 0; ++c1) {
                inc1idx = inc1[c1];
            }
            c1--;
            printf("c1: %i, inc1idx: %i iOffset: %i.\n", c1, inc1idx, iOffsets[c1]);
        }
        c2 = 0;
        inc2idx = 0;
        for (int j = 0; j < n; ++j) {
            if (j == inc2idx) {
                c2++;
                inc2idx = 0;
                for (; c2 <= d && inc2idx == 0; ++c2) {
                    inc2idx = inc2[c2];
                }
                c2--;
                //printf("c2: %i, inc2idx: %i jOffset: %i.\n", c2, inc2idx, jOffsets[c2]);
            }
            data[(i + iOffsets[c1 - 1]) * nNew + (j + jOffsets[c2 - 1])] = float(rand()) / RAND_MAX;
        }
    }
    NewData new_data;
    new_data.m = mNew;
    new_data.n = nNew;
    new_data.data = data;
    return new_data;
}