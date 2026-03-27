#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

// ============================================================
// Step4: Softmax カーネル
//
// LLMで何に使うか:
//   Attention Score の正規化
//   softmax(Q×K^T / sqrt(d_k)) → Attention Weight
//   出力層のトークン確率計算
//
// Softmaxの定義:
//   softmax(x_i) = exp(x_i) / Σexp(x_j)
//
// 問題点:
//   x_i が大きいと exp(x_i) がオーバーフローする！
//   例: exp(1000) = inf
//
// 解決策: Numerically Stable Softmax
//   max値を引いてからexpを取る
//   softmax(x_i) = exp(x_i - max(x)) / Σexp(x_j - max(x))
//   → 値は変わらないが、オーバーフローしない
//
// 今回学ぶこと:
//   ① 3パスのナイーブ実装（max→sum→div を別々に）
//   ② 2パスのShared Memory実装（並列reduction活用）
// ============================================================

#define BLOCK_SIZE 256  // 1ブロックのスレッド数

// ============================================================
// ① ナイーブ実装（CPU的な発想）
// 1スレッドが1行を全部処理する
// 並列度が低い（行数分のスレッドしか使えない）
// ============================================================
__global__ void softmax_naive(
    const float* input,
    float* output,
    int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* x = input + row * cols;
    float* y = output + row * cols;

    // Pass1: max値を求める（数値安定性のため）
    float max_val = -FLT_MAX;
    for (int j = 0; j < cols; j++)
        max_val = fmaxf(max_val, x[j]);

    // Pass2: exp(x - max) の合計を求める
    float sum = 0.0f;
    for (int j = 0; j < cols; j++)
        sum += expf(x[j] - max_val);

    // Pass3: 正規化して出力
    for (int j = 0; j < cols; j++)
        y[j] = expf(x[j] - max_val) / sum;
}

// ============================================================
// ② Shared Memory実装（本番に近い形）
// 1ブロックが1行を処理する
// Shared Memoryで並列Reductionを使ってmax/sumを高速計算
// ============================================================
__global__ void softmax_shared(
    const float* input,
    float* output,
    int rows, int cols)
{
    // Shared Memory: max計算用 と sum計算用
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    int row = blockIdx.x;   // 1ブロック = 1行
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* x = input + row * cols;
    float* y = output + row * cols;

    // -------------------------------------------------------
    // Pass1: 自分の担当要素のmax値をShared Memoryに集める
    // -------------------------------------------------------
    float thread_max = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x)
        thread_max = fmaxf(thread_max, x[j]);
    s_max[tid] = thread_max;
    __syncthreads();

    // Parallel Reduction でブロック全体のmaxを求める
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    float max_val = s_max[0]; // ブロック全体のmax

    // -------------------------------------------------------
    // Pass2: exp(x - max) の合計をShared Memoryで並列計算
    // -------------------------------------------------------
    float thread_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x)
        thread_sum += expf(x[j] - max_val);
    s_sum[tid] = thread_sum;
    __syncthreads();

    // Parallel Reduction でブロック全体のsumを求める
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }
    float sum = s_sum[0]; // ブロック全体のsum

    // -------------------------------------------------------
    // Pass3: 正規化して出力
    // -------------------------------------------------------
    for (int j = tid; j < cols; j += blockDim.x)
        y[j] = expf(x[j] - max_val) / sum;
}

// CPU版Softmax（検証用）
void softmax_cpu(const float* x, float* y, int cols)
{
    float max_val = -FLT_MAX;
    for (int j = 0; j < cols; j++)
        max_val = fmaxf(max_val, x[j]);

    float sum = 0.0f;
    for (int j = 0; j < cols; j++)
        sum += expf(x[j] - max_val);

    for (int j = 0; j < cols; j++)
        y[j] = expf(x[j] - max_val) / sum;
}

int main()
{
    // LLMのAttentionでよくあるサイズ
    const int rows = 512;   // バッチ × シーケンス長
    const int cols = 512;   // Attention Scoreの次元（シーケンス長）

    printf("Softmax: (%d x %d) 行列\n\n", rows, cols);

    int total = rows * cols;

    // CPU側メモリ確保
    float* h_input = new float[total];
    float* h_out_naive = new float[total]();
    float* h_out_shared = new float[total]();
    float* h_out_cpu = new float[total]();

    // テストデータ生成（大きい値も含めてオーバーフロー耐性を確認）
    srand(42);
    for (int i = 0; i < total; i++)
        h_input[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f; // -5〜+5

    // 最初の行をCPUで計算（正解値）
    softmax_cpu(h_input, h_out_cpu, cols);

    // GPU側メモリ確保
    float* d_input, * d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));
    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);

    // タイマー
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- ① ナイーブ実装 ----
    printf("① ナイーブ実装 実行中...\n");
    dim3 naive_block(BLOCK_SIZE);
    dim3 naive_grid((rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        softmax_naive << <naive_grid, naive_block >> > (d_input, d_output, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    cudaMemcpy(h_out_naive, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (1000回合計)\n\n", ms_naive);

    // ---- ② Shared Memory実装 ----
    printf("② Shared Memory実装 実行中...\n");
    dim3 shared_block(BLOCK_SIZE);
    dim3 shared_grid(rows);  // 1ブロック = 1行

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        softmax_shared << <shared_grid, shared_block >> > (d_input, d_output, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_shared = 0;
    cudaEventElapsedTime(&ms_shared, start, stop);
    cudaMemcpy(h_out_shared, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (1000回合計)\n\n", ms_shared);

    // ---- 速度比較 ----
    printf("速度比較:\n");
    printf("  ナイーブ:       %.3f ms\n", ms_naive);
    printf("  Shared Memory: %.3f ms\n", ms_shared);
    printf("  速度向上: %.2fx\n\n", ms_naive / ms_shared);

    // ---- 精度検証 ----
    // 最初の行をCPU結果と比較
    float max_diff_naive = 0.0f;
    float max_diff_shared = 0.0f;
    float sum_check_naive = 0.0f;
    float sum_check_shared = 0.0f;

    for (int j = 0; j < cols; j++)
    {
        max_diff_naive = fmaxf(max_diff_naive, fabsf(h_out_naive[j] - h_out_cpu[j]));
        max_diff_shared = fmaxf(max_diff_shared, fabsf(h_out_shared[j] - h_out_cpu[j]));
        sum_check_naive += h_out_naive[j];
        sum_check_shared += h_out_shared[j];
    }

    printf("精度検証（1行目）:\n");
    printf("  ナイーブ  - CPU誤差: %e  合計: %.6f %s\n",
        max_diff_naive, sum_check_naive,
        sum_check_naive > 0.9999f ? "✓ OK" : "✗ NG");
    printf("  Shared    - CPU誤差: %e  合計: %.6f %s\n",
        max_diff_shared, sum_check_shared,
        sum_check_shared > 0.9999f ? "✓ OK" : "✗ NG");

    // ---- 出力値を少し表示 ----
    printf("\nSoftmax出力（1行目の先頭8要素）:\n");
    printf("  入力:  ");
    for (int j = 0; j < 8; j++) printf("%6.2f ", h_input[j]);
    printf("\n  出力:  ");
    for (int j = 0; j < 8; j++) printf("%6.4f ", h_out_shared[j]);
    printf("\n  ※全要素の合計 = 1.0 になるのがSoftmaxの特性\n");

    // 後片付け
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_out_naive;
    delete[] h_out_shared;
    delete[] h_out_cpu;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaDeviceReset();
    return 0;
}
