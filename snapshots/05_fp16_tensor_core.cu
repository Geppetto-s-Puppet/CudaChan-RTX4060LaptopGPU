#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda_fp16.h>       // FP16型 (__half)
#include <mma.h>             // Tensor Core (wmma API)

using namespace nvcuda;      // wmma:: を使うため

// ============================================================
// Step5: FP16 + Tensor Core
//
// LLMで何に使うか:
//   実際のLLM推論は全部FP16（またはBF16）で動いている
//   GPT/Llama等のモデルファイルもFP16で保存されている
//   Tensor Coreを使うとFP32より4〜8倍速い
//
// FP32 vs FP16:
//   FP32: 符号1bit + 指数8bit + 仮数23bit = 32bit
//   FP16: 符号1bit + 指数5bit + 仮数10bit = 16bit
//   → メモリ使用量が半分、計算が速い、少し精度が落ちる
//
// Tensor Coreとは:
//   行列積専用のハードウェアユニット
//   RTX 4060は4th Gen Tensor Coreを搭載
//   16×16×16の行列積を1命令で実行できる
//
// wmma (Warp Matrix Multiply Accumulate) API:
//   Tensor Coreをプログラマが直接叩けるAPI
//   1ワープ(32スレッド)が協調して16×16行列を処理する
// ============================================================

#define TILE_SIZE 16  // Tensor Coreの基本サイズ（16×16）

// ============================================================
// ① FP32版 行列積（比較用・Step3と同じ原理）
// ============================================================
__global__ void matmul_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        s_A[ty][tx] = (row < M && t * TILE_SIZE + tx < K)
            ? A[row * K + t * TILE_SIZE + tx] : 0.0f;
        s_B[ty][tx] = (col < N && t * TILE_SIZE + ty < K)
            ? B[(t * TILE_SIZE + ty) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ============================================================
// ② Tensor Core版 行列積（wmma API使用）
//
// ポイント:
//   1ワープ(32スレッド)で16×16×16の行列積を処理
//   入力: FP16、出力: FP32（精度を保つため）
//   これがLLMの実際の計算に一番近い形
// ============================================================
__global__ void matmul_tensor_core(
    const __half* A,   // FP16入力
    const __half* B,   // FP16入力
    float* C,          // FP32出力（accumulator）
    int M, int N, int K)
{
    // wmmaのフラグメント（Tensor Coreが扱う行列の断片）
    // fragment = 1ワープが協調して持つ行列データ
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>    frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>    frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                   frag_C;

    // このワープが担当するCのタイル位置
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warp_col = blockIdx.x * 16;

    // accumulatorを0で初期化
    wmma::fill_fragment(frag_C, 0.0f);

    // Kの次元を16ずつ進む
    for (int k = 0; k < K; k += 16)
    {
        if (warp_row < M && warp_col < N && k + 16 <= K)
        {
            // ★ グローバルメモリからフラグメントにロード
            // 32スレッドが協調して16×16タイルを読み込む
            wmma::load_matrix_sync(frag_A, A + warp_row * K + k, K);
            wmma::load_matrix_sync(frag_B, B + k * N + warp_col, N);

            // ★ Tensor Coreで行列積！（これが1命令で16×16×16を計算）
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
    }

    // 結果をグローバルメモリに書き出す
    if (warp_row < M && warp_col < N)
        wmma::store_matrix_sync(C + warp_row * N + warp_col, frag_C, N,
            wmma::mem_row_major);
}

// FP32行列をFP16に変換するカーネル
__global__ void convert_fp32_to_fp16(
    const float* src, __half* dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

int main()
{
    // 行列サイズ（Tensor CoreはTILE_SIZEの倍数が必要）
    const int M = 512;
    const int K = 512;
    const int N = 512;

    printf("行列積: A(%dx%d) × B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);
    printf("FP32サイズ: %.1f MB × 3行列 = %.1f MB\n",
        M * K * 4.0f / (1024 * 1024),
        M * K * 4.0f / (1024 * 1024) * 3);
    printf("FP16サイズ: %.1f MB × 2行列 = %.1f MB（半分！）\n\n",
        M * K * 2.0f / (1024 * 1024),
        M * K * 2.0f / (1024 * 1024) * 2);

    int total = M * N;

    // CPU側メモリ
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_fp32 = new float[total]();
    float* h_C_tensor = new float[total]();

    // 初期化
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // GPU: FP32メモリ
    float* d_A_fp32, * d_B_fp32, * d_C_fp32;
    cudaMalloc(&d_A_fp32, M * K * sizeof(float));
    cudaMalloc(&d_B_fp32, K * N * sizeof(float));
    cudaMalloc(&d_C_fp32, total * sizeof(float));
    cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // GPU: FP16メモリ（FP32から変換）
    __half* d_A_fp16, * d_B_fp16;
    float* d_C_tensor;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(__half));
    cudaMalloc(&d_C_tensor, total * sizeof(float));

    // FP32 → FP16変換カーネルを実行
    int conv_block = 256;
    convert_fp32_to_fp16 << <(M * K + conv_block - 1) / conv_block, conv_block >> >
        (d_A_fp32, d_A_fp16, M * K);
    convert_fp32_to_fp16 << <(K * N + conv_block - 1) / conv_block, conv_block >> >
        (d_B_fp32, d_B_fp16, K * N);
    cudaDeviceSynchronize();

    // タイマー
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- ① FP32版 ----
    printf("① FP32 Tiled MatMul 実行中...\n");
    dim3 fp32_block(TILE_SIZE, TILE_SIZE);
    dim3 fp32_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        matmul_fp32 << <fp32_grid, fp32_block >> > (d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_fp32 = 0;
    cudaEventElapsedTime(&ms_fp32, start, stop);
    cudaMemcpy(h_C_fp32, d_C_fp32, total * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (100回合計)\n\n", ms_fp32);

    // ---- ② Tensor Core版 ----
    printf("② FP16 Tensor Core MatMul 実行中...\n");
    // 1ワープ=32スレッド、y方向に並べる
    dim3 tc_block(16, 32);
    dim3 tc_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE * 2);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        matmul_tensor_core << <tc_grid, tc_block >> > (d_A_fp16, d_B_fp16, d_C_tensor, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_tensor = 0;
    cudaEventElapsedTime(&ms_tensor, start, stop);
    cudaMemcpy(h_C_tensor, d_C_tensor, total * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (100回合計)\n\n", ms_tensor);

    // ---- 速度比較 ----
    printf("速度比較:\n");
    printf("  FP32 Tiled:         %.3f ms\n", ms_fp32);
    printf("  FP16 Tensor Core:   %.3f ms\n", ms_tensor);
    printf("  速度向上: %.2fx\n\n", ms_fp32 / ms_tensor);

    // ---- 精度検証 ----
    // FP16は精度が落ちるので誤差が大きめになる（それが正常）
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < total; i++)
    {
        float diff = fabsf(h_C_fp32[i] - h_C_tensor[i]);
        max_diff = fmaxf(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= total;

    printf("FP32 vs FP16 誤差:\n");
    printf("  最大誤差: %.4f\n", max_diff);
    printf("  平均誤差: %.4f\n", avg_diff);
    printf("  ※FP16は仮数部が少ないため誤差は出るが実用上問題なし\n\n");

    // メモリ節約量の表示
    printf("メモリ使用量比較:\n");
    printf("  FP32: %d MB\n", (int)((M * K + K * N) * sizeof(float) / (1024 * 1024)));
    printf("  FP16: %d MB（%.0f%%削減）\n",
        (int)((M * K + K * N) * sizeof(__half) / (1024 * 1024)),
        50.0f);

    // 後片付け
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_A; delete[] h_B; delete[] h_C_fp32; delete[] h_C_tensor;
    cudaFree(d_A_fp32); cudaFree(d_B_fp32); cudaFree(d_C_fp32);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16); cudaFree(d_C_tensor);
    cudaDeviceReset();
    return 0;
}