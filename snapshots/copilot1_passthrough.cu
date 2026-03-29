#include <iostream>
#include <cuda_runtime.h>

// カーネル関数の定義 (__global__ 修飾子を付ける)
__global__ void addKernel(int* data) {
    int idx = threadIdx.x;  // スレッドIDを取得
    data[idx] += 1;         // 各スレッドが1を加算
}

int main() {
    const int arraySize = 5;
    int hostData[arraySize] = { 0, 1, 2, 3, 4 };
    int* deviceData;

    // GPU上にメモリを確保 (cudaMalloc)
    cudaMalloc((void**)&deviceData, arraySize * sizeof(int));
    cudaMemcpy(deviceData, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // カーネルを起動 (<<<1, arraySize>>> はスレッド数を指定)
    addKernel << <1, arraySize >> > (deviceData);

    // 結果をCPUにコピー (cudaMemcpy)
    cudaMemcpy(hostData, deviceData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 結果を表示
    for (int i = 0; i < arraySize; i++) {
        std::cout << "Result[" << i << "]: " << hostData[i] << std::endl;
    }

    // GPUメモリを解放
    cudaFree(deviceData);

    return 0;
}

// CUDAプログラムの基本的な流れ：
//     データの準備: GPUに転送するためのデータを用意する。
//     メモリの確保: GPU上にメモリを確保する。
//     カーネル関数の定義: 並列実行される関数（カーネル）を定義する。
//     データの転送: データをCPUからGPUに転送する。
//     カーネルの実行: GPU上でカーネルを並列実行する。
//     結果の取得: GPUからCPUに結果を戻す。