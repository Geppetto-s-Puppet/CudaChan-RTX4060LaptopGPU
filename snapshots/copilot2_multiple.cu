#include <iostream>
#include <cuda_runtime.h>

// カーネル関数の定義
__global__ void multiplyByTwo(int* data) {
    int idx = threadIdx.x;
    data[idx] *= 2;  // 各スレッドが2倍にする
}

int main() {
    const int arraySize = 5;
    int hostData[arraySize] = { 1, 2, 3, 4, 5 };
    int* deviceData;

    // GPU上にメモリを確保
    cudaMalloc((void**)&deviceData, arraySize * sizeof(int));

    // データをホストからデバイスにコピー
    cudaMemcpy(deviceData, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // カーネルを実行
    multiplyByTwo << <1, arraySize >> > (deviceData);

    // 結果をホストにコピー
    cudaMemcpy(hostData, deviceData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 結果を表示
    for (int i = 0; i < arraySize; i++) {
        std::cout << "Result[" << i << "]: " << hostData[i] << std::endl;
    }

    // GPUメモリを解放
    cudaFree(deviceData);

    return 0;
}