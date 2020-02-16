/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <chrono>
#include <ratio>

class TimeIt {
public:
    using TimePoint       = std::chrono::high_resolution_clock::time_point;
    using Clock           = std::chrono::high_resolution_clock;
    TimePoint const start = Clock::now();

    double stop() const {
        TimePoint end = Clock::now();
        return TimeIt::diff(start, end);
    }

    static double diff(TimePoint start, TimePoint end){
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        return time_span.count();
    }

    TimeIt operator=(TimeIt p){
        return TimeIt(p);
    }

    TimeIt(const TimeIt& p):
        start(p.start)
    {}

    TimeIt() = default;
};


bool check(cudaError_t err, const char* msg){
    if (err != cudaSuccess)
    {
        fprintf(stderr, msg, cudaGetErrorString(err));
        fprintf(stderr, "(ec: %d) (error: %s)", err, cudaGetErrorString(err));
    }
    return true;
}


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 *
 * Performance: 39.4 x 10 x 10 x 50000000
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void cross(const float *A, const float *B, float *C, int numElements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        C[i] += A[i] * B[i];
    }
}

__global__ void simpleMultiply(float *a, float* b, float *c, int N)
{
    // blockDim.y == blockDim.x == TILE_DIM == 32
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < TILE_DIM; i++) {
        // the cache line may easily be evicted from the cache between iterations i and i+1. 
        sum += a[row*TILE_DIM+i] * b[i*N+col];
    }

    c[row*N+col] = sum;
}

__global__ void coalescedMultiply(float *a, float* b, float *c, int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    
    // __syncwarp() is sufficient after reading the tile of A into shared memory because only threads within the warp that write the data into shared memory read this data
    __syncwarp();
    
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* b[i*N+col];
    }
    
    c[row*N+col] = sum;
}


__global__ void sharedABMultiply(float *a, float* b, float *c, int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float bTile[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    
    // __syncthreads() call is required after reading the B tile because a warp reads data from shared memory that were written to shared memory by different warps.
    __syncthreads();
    
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}

template<typename T>
T* device_malloc(std::size_t n){
    T* ptr = nullptr;
    check(cudaMalloc((void **)&ptr, n), "Failed to allocate device memory");
    return ptr;
}

// Base Clock 1506 MHz
// Shading Units 1280
// TMUs 80              Texture Mapping Unit
// ROPs 48              Render Output Unit
// SM Count 10          32 CUDA cores
// L1 Cache 48 KB (per SM)
// L2 Cache 1536 KB
// Threads are scheduled in groups of 32 threads called warps
// Each SM features two warp schedulers and two instruction dispatch units, allowing two warps to be issued and executed concurrently.
// The dual warp scheduler selects two warps, and issues one instruction from each warp to a group of 16 cores, 16 load/store units, or 4 SFUs.

// cudaMalloc(), is guaranteed to be aligned to at least 256 bytes.
// Therefore, choosing sensible thread block sizes, such as multiples of the warp size (i.e., 32 on current GPUs)

// non-unit-stride global memory accesses should be avoided whenever possible

//  compute capability 7.0 each multiprocessor has
//  - 65,536 32-bit registers and can have a
//  - maximum of 2048 simultaneous threads resident (64 warps x 32 threads per warp)
// for a multiprocessor to have 100% occupancy, each thread can use at most 32 registers.

// The number of threads per block should be a multiple of 32 threads, because this provides optimal computing efficiency and facilitates coalescing. 


/**
 * Host main routine
 */
int
main(void)
{
    // Print the vector length to be used, and compute its size
    int numElements = 50000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = (float *)malloc(size); // Allocate the host input vector A
    float *h_B = (float *)malloc(size); // Allocate the host input vector B
    float *h_C = (float *)malloc(size); // Allocate the host output vector C

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vectors
    float *d_A = device_malloc<float>(size);
    float *d_B = device_malloc<float>(size);
    float *d_C = device_malloc<float>(size);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");

    check(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Failed to copy vector A from host to device");
    check(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Failed to copy vector B from host to device");

    FILE* data = fopen("perf.csv", "w");

    // Launch the Vector Add CUDA Kernel
    for(int i = 0; i < 1024; i += 8){
        int threadsPerBlock = i + 8;
        int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

        // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        // check this configuration is okay
        // grid, block, 0, stream
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

        if (check(cudaGetLastError(), "Failed to launch vectorAdd kernel")){
            float time = 0;
            cudaEvent_t start, stop;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);
            TimeIt chrono;
            for (int j = 0; j < 10; j++){

                for(int i = 0; i < 10; i++){
                    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
                }
            }

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            time = chrono.stop();

            float cuda_time = 0;
            cudaEventElapsedTime( &cuda_time, start, stop );
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            fprintf(data, "%d, %d, %f, %f\n", blocksPerGrid, threadsPerBlock, time / 10, cuda_time / 10);
        }
    }

    fclose(data);
    check(cudaGetLastError(), "Failed to launch vectorAdd kernel");

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    check(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    check(cudaFree(d_A), "Failed to free device vector A");
    check(cudaFree(d_B), "Failed to free device vector B");
    check(cudaFree(d_C), "Failed to free device vector C");


    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

