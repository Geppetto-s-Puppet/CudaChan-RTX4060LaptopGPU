#include <iostream>
#include <cuda_runtime.h>

// カーネル関数の定義
__global__ void processData(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // グローバルなスレッドIDを計算
    data[idx] += idx;  // スレッドIDを使ってデータを操作
}

int main() {
    const int arraySize = 1024;
    int hostData[arraySize];
    for (int i = 0; i < arraySize; i++) {
        hostData[i] = i;
    }
    int* deviceData;

    // GPU上にメモリを確保
    cudaMalloc((void**)&deviceData, arraySize * sizeof(int));

    // データをホストからデバイスにコピー
    cudaMemcpy(deviceData, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 4ブロック x 256スレッドでカーネルを実行
    processData << <4, 256 >> > (deviceData);

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