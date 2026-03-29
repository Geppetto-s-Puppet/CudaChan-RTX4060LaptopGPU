#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// ============================================================
// Step3: 行列積（Matrix Multiplication）
//
// LLMで何に使うか:
//   Q×K^T  （Attention Scoreの計算）
//   Score×V（Attention出力の計算）
//   x×W    （Linear層、FFNの計算）
//   → モデルの計算時間の約90%がここ
//
// 今回学ぶこと:
//   ① ナイーブ実装（Shared Memoryなし）
//   ② Tiled実装  （Shared Memoryあり、本番に近い形）
//   の2つを書いて速度差を体感する
// ============================================================

#define TILE_SIZE 16  // Shared Memoryのタイルサイズ（16×16）

// ============================================================
// ① ナイーブ実装
// 全スレッドが毎回グローバルメモリにアクセスする
// 遅い理由: M×N×K回のグローバルメモリ読み込みが発生する
// ============================================================
__global__ void matmul_naive(
    const float* A,  // M×K 行列
    const float* B,  // K×N 行列
    float* C,        // M×N 行列（出力）
    int M, int N, int K)
{
    // このスレッドが担当するCの(row, col)を計算
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // C[row][col] = A[row][:] · B[:][col]（内積）
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
    {
        sum += A[row * K + k] * B[k * N + col];
        // ↑ 毎ループでグローバルメモリに2回アクセス → 遅い
    }
    C[row * N + col] = sum;
}

// ============================================================
// ② Tiled実装（Shared Memory版）
//
// アイデア:
//   行列をTILE_SIZE×TILE_SIZEの小さなタイルに分割して
//   Shared Memoryにまとめてロードしてから計算する
//
//   グローバルメモリアクセス回数が 1/TILE_SIZE に減る！
// ============================================================
__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K)
{
    // Shared Memory（タイル1枚分）
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    // Kの次元をTILE_SIZEずつ進む
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // ★ タイルをShared Memoryにロード
        // 各スレッドが1要素だけ担当してロードする
        if (row < M && (t * TILE_SIZE + tx) < K)
            s_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < K)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        // ★ 全スレッドのロードが終わるまで待つ
        __syncthreads();

        // ★ Shared Memoryから読んで計算（高速！）
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // 次のタイルをロードする前に計算が終わるまで待つ
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// 行列をランダム値で初期化
void init_matrix(float* mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (float)(rand() % 10) / 10.0f; // 0.0〜0.9
}

// 行列の一部を表示
void print_matrix(const char* name, float* mat, int rows, int cols)
{
    printf("%s (%dx%d) の左上4x4:\n", name, rows, cols);
    for (int i = 0; i < 4 && i < rows; i++)
    {
        for (int j = 0; j < 4 && j < cols; j++)
            printf("%6.2f ", mat[i * cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main()
{
    // 行列サイズ（LLMでいえばシーケンス長×次元数）
    const int M = 64;  // Aの行数
    const int K = 64;  // Aの列数 = Bの行数
    const int N = 64;  // Bの列数

    printf("行列積: A(%dx%d) × B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    // CPU側メモリ確保
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_naive = new float[M * N]();
    float* h_C_tiled = new float[M * N]();

    // 初期化
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // GPU側メモリ確保
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // CPU → GPU転送
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // グリッドサイズ計算
    dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 16×16 = 256スレッド/ブロック
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,  // X方向のブロック数
        (M + TILE_SIZE - 1) / TILE_SIZE   // Y方向のブロック数
    );

    // ---- ① ナイーブ実装 ----
    printf("① ナイーブ実装 実行中...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) // 100回ループして時間計測
        matmul_naive << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    cudaMemcpy(h_C_naive, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (100回合計)\n\n", ms_naive);

    // ---- ② Tiled実装 ----
    printf("② Tiled実装（Shared Memory）実行中...\n");

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        matmul_tiled << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_tiled = 0;
    cudaEventElapsedTime(&ms_tiled, start, stop);
    cudaMemcpy(h_C_tiled, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (100回合計)\n\n", ms_tiled);

    // ---- 結果比較 ----
    printf("速度比較:\n");
    printf("  ナイーブ: %.3f ms\n", ms_naive);
    printf("  Tiled:   %.3f ms\n", ms_tiled);
    printf("  速度向上: %.2fx\n\n", ms_naive / ms_tiled);

    // 結果が同じか検証
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float diff = fabsf(h_C_naive[i] - h_C_tiled[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("結果の最大誤差: %e %s\n\n",
        max_diff, max_diff < 1e-3f ? "✓ OK" : "✗ NG");

    print_matrix("C (naive)", h_C_naive, M, N);
    print_matrix("C (tiled)", h_C_tiled, M, N);

    // 後片付け
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_A; delete[] h_B; delete[] h_C_naive; delete[] h_C_tiled;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaDeviceReset();
    return 0;
}
