#include <iostream>
#include <cuda_runtime.h>

// カーネル関数の定義
__global__ void calculateIndex(int* data) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;  // グローバルスレッドIDを計算
    data[globalIdx] = globalIdx;  // グローバルIDをそのまま格納
}

int main() {
    const int arraySize = 16;
    int hostData[arraySize];
    int* deviceData;

    // GPU上にメモリを確保
    cudaMalloc((void**)&deviceData, arraySize * sizeof(int));

    // 2ブロック x 8スレッドでカーネルを実行
    calculateIndex << <2, 8 >> > (deviceData);

    // 結果をホストにコピー
    cudaMemcpy(hostData, deviceData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 結果を表示
    for (int i = 0; i < arraySize; i++) {
        std::cout << "Result[" << i << "] = " << hostData[i] << std::endl;
    }

    // GPUメモリを解放
    cudaFree(deviceData);

    return 0;
}