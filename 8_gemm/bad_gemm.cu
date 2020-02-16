#include <cstdio>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
    bool cpu;
} Matrix;


Matrix make_cpu(int w, int h){
    Matrix m;
    m.width = w;
    m.height = h;
    m.elements = static_cast<float*>(malloc(w * h * sizeof(float)));
    m.cpu = true;
    return m;
}

Matrix make_gpu(int w, int h){
    Matrix m;
    m.width = w;
    m.height = h;
    auto size = w * h;
    cudaMalloc(&m.elements, size);
    m.cpu = false;
    return m;
}

Matrix make_gpu_from(Matrix m){
    Matrix gpu = make_gpu(m.width, m.height);
    cudaMemcpy(gpu.elements, m.elements, m.width * m.height, cudaMemcpyHostToDevice);
    return gpu;
}

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];

    C.elements[row * C.width + col] = Cvalue;
}


int main(){
    std::size_t size = 2048;

    Matrix A = make_cpu(size, size);
    Matrix B = make_cpu(size, size);
    Matrix C = make_cpu(size, size);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time = 0;

    for (int i = 0; i < 10; ++i){
        cudaEventRecord(start, 0);
        for (int j = 0; j < 10; ++j){
            MatMul(A, B, C);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float cuda_time = 0;
        cudaEventElapsedTime(&cuda_time, start, stop);
        time += cuda_time;
    }

    printf("%f \n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
