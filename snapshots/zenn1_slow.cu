# include <stdio.h>

const int BLOCK = 16;
const int WIDTH = 8192;

void Host(double* a, double* b, double* c);
__global__ void Kernel1(double* a, double* b, double* c);
__global__ void Kernel2(double* a, double* b, double* c);
__global__ void Kernel3(double* a, double* b, double* c);

double h_a[WIDTH * WIDTH];
double h_b[WIDTH * WIDTH];
double h_c[WIDTH * WIDTH];

__global__ void Kernel1(double* A, double* B, double* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double tmp = 0.0;
    for (int k = 0; k < WIDTH; k++) {
        int row = k + y * WIDTH;
        int col = x + k * WIDTH;
        tmp += A[row] * B[col];
    }
    C[x + y * WIDTH] = tmp;
}

int main() {
    int i;

    // GPU上にメモリを確保
    double* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, sizeof(double) * WIDTH * WIDTH);
    cudaMalloc((void**)&d_b, sizeof(double) * WIDTH * WIDTH);
    cudaMalloc((void**)&d_c, sizeof(double) * WIDTH * WIDTH);
    cudaMemset(d_c, 0, sizeof(double) * WIDTH * WIDTH);

    for (i = 0; i < WIDTH * WIDTH; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    //変数をGPU上のメモリへコピー
    cudaMemcpy(d_a, h_a, sizeof(double) * WIDTH * WIDTH,
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(double) * WIDTH * WIDTH,
        cudaMemcpyHostToDevice);

    //ブロックとグリッドの定義とカーネルの起動
    dim3 grid(WIDTH / BLOCK, WIDTH / BLOCK, 1);
    dim3 threads(BLOCK, BLOCK, 1);
    Kernel1 << <grid, threads >> > (d_a, d_b, d_c);

    //計算結果を取得
    cudaMemcpy(h_c, d_c, sizeof(double) * WIDTH * WIDTH,
        cudaMemcpyDeviceToHost);
    printf("計算結果=%f\n", h_c[WIDTH * WIDTH - 1]);

    // GPU上のメモリを開放
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
