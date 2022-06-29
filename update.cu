#include <cuComplex.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>


typedef cuFloatComplex ctype;

#define cmplx(x,y) (make_cuFloatComplex(x,y))
#define rpart(x)   (cuCreal(x))
#define ipart(x)   (cuCimag(x))
#define cmul(x, y) (cuCmulf(x,y))
#define cadd(x, y) (cuCaddf(x,y))


// __device__ float atomicAddFloat(float* address, float val)
// {
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __float_as_longlong(val +
//                                __longlong_as_float(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_float(old);
// }


// __device__ void atomicAddComplex(cuFloatComplex* a, cuFloatComplex b){
//     //transform the addresses of real and imag. parts to float pointers
//     float *x = (float*)a;
//     float *y = x+1;
//     //use atomicAdd for float variables
//     atomicAddFloat(x, cuCreal(b));
//     atomicAddFloat(y, cuCimag(b));
//   }


__global__
void fill_complex(const float* src, ctype* dst, int len)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < len) {dst[x] = cmplx(src[x], 0);}
}
  
__global__
void update(const int tau, const int d, const int chi, const int len_l, const int len_c, const int len_r, const int* l_bond, const int* c_bond, const int* r_bond, const int* bc_l, const int* bc_c, const int* bc_r, ctype* C, const ctype* BS, const float* Lambda_l, const ctype* Gamma_lc, const float* Lambda_c, const ctype* Gamma_cr, const float* Lambda_r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x<len_l && y<len_r)
    {   
        int id_l = l_bond[x]; int id_r = r_bond[y];
        int ch_l = bc_l[id_l]; int ch_r = bc_r[id_r];
        int id_c, ch_c;
        ctype c_entry = cmplx(0.0,0.0);
        ctype coeff;
        for(int c=0; c<len_c; c++)
        {   
            id_c = c_bond[c];
            ch_c = bc_c[id_c];
            //printf("%i ", id_r);
            if (ch_c >= ch_r && ch_c <= ch_l)
            {
                //printf("in loop %i", ch_l-ch_c);
                coeff = cmul(cmul(BS[(ch_l-tau)*d*d + (tau-ch_r)*d + (ch_l-ch_c)], cmul(Gamma_lc[id_l*chi + id_c], Gamma_cr[id_c*chi + id_r])), cmplx(Lambda_l[id_l]*Lambda_l[id_c]*Lambda_l[id_r], 0));
                //printf("have coeff ");

                c_entry = cadd(c_entry, coeff);
                //printf("added ");
            }
            //printf("finished if ");
        }
        C[x*len_r + y] = c_entry;
    }
}


__global__ void update_new(const int tau, const int d, const int chi, const int len_l, const int len_c, const int len_r, const int* l_bond, const int* c_bond, const int* r_bond, const int* bc_l, const int* bc_c, const int* bc_r, ctype* C, const ctype* BS, const float* Lambda_l, const ctype* Gamma_lc, const float* Lambda_c, const ctype* Gamma_cr, const float* Lambda_r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x<len_l && y<len_r)
    {   
        int id_l = l_bond[x]; int id_r = r_bond[y];
        int ch_l = bc_l[id_l]; int ch_r = bc_r[id_r];
        int id_c, ch_c;
        ctype c_entry = cmplx(0.0,0.0);
        ctype coeff;
        for(int c=0; c<len_c; c++)
        {   
            id_c = 10;
            ch_c = 5;
            //printf("%i ", id_r);
            if (ch_c >= ch_r && ch_c <= ch_l)
            {
                //printf("in loop %i", ch_l-ch_c);
                coeff = cmul(cmul(BS[(ch_l-tau)*d*d + (tau-ch_r)*d + (ch_l-ch_c)], cmul(Gamma_lc[c*chi+id_l], Gamma_cr[c*chi + id_r])), cmplx(Lambda_l[x]*Lambda_l[c]*Lambda_l[y], 0));
                //printf("have coeff ");

                c_entry = cadd(c_entry, coeff);
                //printf("added ");
            }
            //printf("finished if ");
        }
        C[x*len_r + y] = c_entry;
    }
}

void fill_rand(int* array, int size, int low_limit, int high_limit)
{
    int range = high_limit - low_limit;
    for(int i=0;i<size;i++)
    {
        array[i] = rand()%range + low_limit;
    }
}

void fill_rand(float* array, int size, float limit)
{
    for(int i=0;i<size;i++)
    {
        array[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/limit));
    }
}



int main(int argc, char **argv)
{
    
    int d = 10;
    int tau = 5;
    int chi{std::stoi(argv[1])};
    int bonds{std::stoi(argv[2])};
    
    //int *l_bond, *c_bond, *r_bond, *d_l_bond, *d_c_bond, *d_r_bond, *bc_l, *bc_c, *bc_r, *d_bc_l, *d_bc_c, *d_bc_r;
    int *d_l_bond, *d_c_bond, *d_r_bond, *d_bc_l, *d_bc_c, *d_bc_r;
    float *d_Lambda_l, *d_Lambda_c, *d_Lambda_r, *d_BS_d, *d_C_d, *d_Gamma_lc_d, *d_Gamma_cr_d;
    ctype *d_BS, *d_C, *d_Gamma_lc,*d_Gamma_cr;
    
    int l_bond[bonds], c_bond[bonds], r_bond[bonds], bc_l[chi], bc_c[chi], bc_r[chi];
    //float Lambda_l[chi], Lambda_c[chi], Lambda_r[chi];
    //float* BS[d*d*d], C[bonds*bonds], Gamma_lc[chi*chi], Gamma_cr[chi*chi];

    float *Lambda_l = (float*) malloc(chi*sizeof(float));
    float *Lambda_c = (float*) malloc(chi*sizeof(float));
    float *Lambda_r = (float*) malloc(chi*sizeof(float));
    float *BS = (float*) malloc(d*d*d*sizeof(float));
    float *C = (float*) malloc(bonds*bonds*sizeof(float));
    float *Gamma_lc = (float*) malloc(chi*chi*sizeof(float));
    float *Gamma_cr = (float*) malloc(chi*chi*sizeof(float));
    
    fill_rand(l_bond, bonds, 0, chi);
    fill_rand(c_bond, bonds, 0, chi);
    fill_rand(r_bond, bonds, 0, chi);
    fill_rand(bc_l, chi, tau, d);
    fill_rand(bc_c, chi, 0, d);
    fill_rand(bc_r, chi, 0, tau+1);
    fill_rand(Lambda_l, chi, 1.0);
    fill_rand(Lambda_c, chi, 1.0);
    fill_rand(Lambda_r, chi, 1.0);
    fill_rand(BS, d*d*d, 1.0);
    fill_rand(C, bonds*bonds, 1.0);
    fill_rand(Gamma_lc, chi*chi, 1.0);
    fill_rand(Gamma_cr, chi*chi, 1.0);

    //Lambda_l[10] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    //Lambda_c[10] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    //Lambda_r[10] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    cudaMalloc(&d_l_bond, bonds*sizeof(int));
    cudaMalloc(&d_c_bond, bonds*sizeof(int));
    cudaMalloc(&d_r_bond, bonds*sizeof(int));
    cudaMalloc(&d_bc_l, chi*sizeof(int));
    cudaMalloc(&d_bc_c, chi*sizeof(int));
    cudaMalloc(&d_bc_r, chi*sizeof(int));
    cudaMalloc(&d_Lambda_l, chi*sizeof(float));
    cudaMalloc(&d_Lambda_c, chi*sizeof(float));
    cudaMalloc(&d_Lambda_r, chi*sizeof(float));
    cudaMalloc(&d_BS_d, d*d*d*sizeof(float));
    cudaMalloc(&d_C_d, bonds*bonds*sizeof(float));
    cudaMalloc(&d_Gamma_lc_d, chi*chi*sizeof(float));
    cudaMalloc(&d_Gamma_cr_d, chi*chi*sizeof(float));
    cudaMalloc(&d_BS, d*d*d*sizeof(ctype));
    cudaMalloc(&d_C, bonds*bonds*sizeof(ctype));
    cudaMalloc(&d_Gamma_lc, chi*chi*sizeof(ctype));
    cudaMalloc(&d_Gamma_cr, chi*chi*sizeof(ctype));

    cudaMemcpy(d_l_bond, l_bond, bonds*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_bond, c_bond, bonds*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_bond, r_bond, bonds*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bc_l, bc_l, chi*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bc_c, bc_r, chi*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bc_r, bc_c, chi*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lambda_l, Lambda_l, chi*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lambda_c, Lambda_l, chi*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lambda_r, Lambda_l, chi*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_BS_d, BS, d*d*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_d, C, bonds*bonds*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gamma_lc_d, Gamma_lc, chi*chi*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gamma_cr_d, Gamma_cr, chi*chi*sizeof(float), cudaMemcpyHostToDevice);
    
    fill_complex<<<(int)ceil(d*d*d/64), 64>>>(d_BS_d, d_BS, d*d*d);
    fill_complex<<<(int)ceil(bonds*bonds/64), 64>>>(d_C_d, d_C, bonds*bonds);
    fill_complex<<<(int)ceil(chi*chi/64), 64>>>(d_Gamma_lc_d, d_Gamma_lc, chi*chi);
    fill_complex<<<(int)ceil(chi*chi/64), 64>>>(d_Gamma_cr_d, d_Gamma_cr, chi*chi);
    //cudaMemcpy(d_BS, BS, d*d*d*sizeof(ctype), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_C, C, chi*chi*sizeof(ctype), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Gamma_lc, Gamma_lc, chi*chi*sizeof(ctype), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Gamma_cr, Gamma_cr, chi*chi*sizeof(ctype), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    //printf("Hello");
    if ( err != cudaSuccess )
    {
        printf("CUDA Error before sync: %s\n", cudaGetErrorString(err));       
    }
    // Possibly: exit(-1) if program cannot continue....
    
    dim3 threadsPerBlock(32, 32);
    //dim3 threadsPerBlock(1,1,1);
    int nblocks = (int)ceil(bonds/32);
    dim3 numBlocks(nblocks, nblocks);
    //dim3 numBlocks(1,1,1);

    update_new<<<numBlocks, threadsPerBlock>>>(tau, d, chi, bonds, bonds, bonds, d_l_bond, d_c_bond, d_r_bond, d_bc_l, d_bc_c, d_bc_r, d_C, d_BS, d_Lambda_l, d_Gamma_lc, d_Lambda_c, d_Gamma_cr, d_Lambda_r);

    


    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
        //printf("Hello");
    if ( err != cudaSuccess )
    {
        printf("CUDA Error after sync: %s\n", cudaGetErrorString(err));       

    // Possibly: exit(-1) if program cannot continue....
    }
    
    
    return 0;
}