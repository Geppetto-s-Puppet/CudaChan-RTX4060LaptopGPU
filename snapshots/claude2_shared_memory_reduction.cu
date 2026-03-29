#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// ============================================================
// Step2: Shared Memory
// 
// やること: 配列の合計値を並列で計算する（Reduction）
// 
// グローバルメモリだけでやると遅い理由:
//   全スレッドが何度もGPUのメインメモリにアクセスする
//   → バス帯域がボトルネックになる
//
// Shared Memoryを使うと:
//   ブロック内のスレッドでデータを共有できる高速メモリ
//   DX12でいうGroupSharedMemory / groupshared と全く同じ
// ============================================================

#define BLOCK_SIZE 8  // 1ブロックのスレッド数（説明用に小さくしてある）

__global__ void sumReduction(int* input, int* output, int n)
{
    // ★ Shared Memory の宣言
    // __shared__ をつけるだけ。ブロック内の全スレッドが同じ領域を見る
    // DX12の groupshared float s_data[256]; と全く同じ
    __shared__ int s_data[BLOCK_SIZE];

    // このスレッドが担当するグローバルインデックス
    int tid = threadIdx.x;                          // ブロック内のスレッド番号
    int i = blockIdx.x * blockDim.x + threadIdx.x; // グローバルインデックス

    // -------------------------------------------------------
    // Step1: グローバルメモリ → Shared Memoryにロード
    // 各スレッドが自分の担当要素を1つだけ持ってくる
    // -------------------------------------------------------
    if (i < n)
        s_data[tid] = input[i];
    else
        s_data[tid] = 0; // 範囲外は0埋め

    // ★ 全スレッドがここまで終わるまで待つ（同期）
    // DX12の GroupMemoryBarrierWithGroupSync() と全く同じ
    __syncthreads();

    // -------------------------------------------------------
    // Step2: Reduction ループ
    // 半分ずつたたみ込んで合計を計算する
    //
    // 例: [3, 1, 4, 1, 5, 9, 2, 6] を合計する場合
    //
    // stride=4:  [3+5, 1+9, 4+2, 1+6, 5, 9, 2, 6]
    //          = [8,   10,  6,   7,   5, 9, 2, 6]
    // stride=2:  [8+6, 10+7, 6, 7, ...]
    //          = [14,  17,   6, 7, ...]
    // stride=1:  [14+17, 17, ...]
    //          = [31, ...]   ← s_data[0]に合計が入る！
    // -------------------------------------------------------
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads(); // ★ 各ステップの後に必ず同期
    }

    // -------------------------------------------------------
    // Step3: ブロックの合計をグローバルメモリに書き出す
    // s_data[0]にブロック全体の合計が入っている
    // -------------------------------------------------------
    if (tid == 0)
    {
        output[blockIdx.x] = s_data[0];
    }
}

int main()
{
    const int N = 16;       // 合計する要素数
    const int GRID_SIZE = N / BLOCK_SIZE; // ブロック数 = 16/8 = 2

    // CPU側メモリ
    int h_input[N];
    int h_output[GRID_SIZE] = { 0 };

    // テストデータ: 1〜16を入れる（合計は136になるはず）
    printf("入力データ: ");
    for (int i = 0; i < N; i++)
    {
        h_input[i] = i + 1;
        printf("%d ", h_input[i]);
    }
    printf("\n期待する合計: %d\n\n", N * (N + 1) / 2); // = 136

    // GPU側メモリ確保
    int* d_input = nullptr;
    int* d_output = nullptr;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, GRID_SIZE * sizeof(int));

    // CPU → GPU転送
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // ★ カーネル起動
    // <<<GRID_SIZE, BLOCK_SIZE>>> = <<<2, 8>>>
    // つまり 2ブロック × 8スレッド = 16スレッド同時起動
    printf("カーネル起動: <<%d, %d>>>\n", GRID_SIZE, BLOCK_SIZE);
    printf("Block0: 要素[0..7]の合計を計算\n");
    printf("Block1: 要素[8..15]の合計を計算\n\n");

    sumReduction << <GRID_SIZE, BLOCK_SIZE >> > (d_input, d_output, N);
    cudaDeviceSynchronize();

    // GPU → CPU転送
    cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // 各ブロックの部分合計を表示
    int total = 0;
    for (int i = 0; i < GRID_SIZE; i++)
    {
        printf("Block%dの部分合計: %d\n", i, h_output[i]);
        total += h_output[i];
    }
    printf("\n最終合計: %d\n", total);

    // 後片付け
    cudaFree(d_input);
    cudaFree(d_output);
    cudaDeviceReset();
    return 0;
}
