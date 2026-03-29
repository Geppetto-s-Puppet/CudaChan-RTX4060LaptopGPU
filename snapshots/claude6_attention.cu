#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

// ============================================================
// Step6: Scaled Dot-Product Attention
//
// LLMの心臓部。ここまで学んだ全部を組み合わせる。
//
// 計算式:
//   Attention(Q, K, V) = softmax(Q×K^T / sqrt(d_k)) × V
//
// 各行列の意味:
//   Q (Query)  : 「何を探しているか」   shape: [seq_len, d_k]
//   K (Key)    : 「何が入っているか」   shape: [seq_len, d_k]
//   V (Value)  : 「実際の情報」         shape: [seq_len, d_v]
//   Score      : Q×K^T                  shape: [seq_len, seq_len]
//   Weight     : softmax(Score/√d_k)    shape: [seq_len, seq_len]
//   Output     : Weight×V               shape: [seq_len, d_v]
//
// 今回の実装:
//   ① ナイーブ実装（3つのカーネルを順番に呼ぶ）
//   ② Fused実装  （1カーネルで全部やる、中間バッファ削減）
// ============================================================

#define TILE_SIZE  16
#define SEQ_LEN    64    // シーケンス長（トークン数）
#define D_MODEL    64    // 次元数

// ============================================================
// 共通カーネル群（Step3,4で作ったものの再利用）
// ============================================================

// 行列積 C = A × B
__global__ void matmul(
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

// スケーリング: Score = Score / sqrt(d_k)
__global__ void scale(float* mat, int total, float factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) mat[i] *= factor;
}

// Softmax（行ごと）
__global__ void softmax(float* mat, int rows, int cols)
{
    __shared__ float s_max[TILE_SIZE * TILE_SIZE];
    __shared__ float s_sum[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float* x = mat + row * cols;

    float thread_max = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x)
        thread_max = fmaxf(thread_max, x[j]);
    s_max[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    float max_val = s_max[0];

    float thread_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x)
        thread_sum += expf(x[j] - max_val);
    s_sum[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }
    float sum = s_sum[0];

    for (int j = tid; j < cols; j += blockDim.x)
        x[j] = expf(x[j] - max_val) / sum;
}

// ============================================================
// ② Fused Attention カーネル
//
// ① のナイーブ実装は中間バッファ（Score行列）をGPUメモリに
// 書き出してからまた読み込む → メモリ帯域の無駄
//
// Fused版は1カーネルで Q×K^T → scale → softmax まで完結
// 中間結果をShared Memoryに保持したまま処理する
// FlashAttentionの基本アイデアがこれ
// ============================================================
__global__ void fused_attention(
    const float* Q,      // [seq_len, d_k]
    const float* K,      // [seq_len, d_k]
    const float* V,      // [seq_len, d_v]
    float* Output,       // [seq_len, d_v]
    int seq_len, int d_k, int d_v)
{
    // 1ブロック = 1クエリトークン を担当
    int q_idx = blockIdx.x;  // 何番目のトークンか
    int tid = threadIdx.x;

    if (q_idx >= seq_len) return;

    // Shared Memory
    extern __shared__ float smem[];
    float* s_score = smem;                  // [seq_len]  Attention Score
    float* s_reduce = smem + seq_len;        // [blockDim.x] reduction用

    const float scale_factor = 1.0f / sqrtf((float)d_k);

    // -------------------------------------------------------
    // Step1: Q[q_idx] × K^T を計算 → s_score[]に格納
    // -------------------------------------------------------
    for (int k_idx = tid; k_idx < seq_len; k_idx += blockDim.x)
    {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++)
            dot += Q[q_idx * d_k + d] * K[k_idx * d_k + d];
        s_score[k_idx] = dot * scale_factor;  // スケーリングも同時に
    }
    __syncthreads();

    // -------------------------------------------------------
    // Step2: Softmax（Shared Memory上で完結）
    // -------------------------------------------------------
    // max値を求める
    float thread_max = -FLT_MAX;
    for (int k_idx = tid; k_idx < seq_len; k_idx += blockDim.x)
        thread_max = fmaxf(thread_max, s_score[k_idx]);
    s_reduce[tid] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        __syncthreads();
    }
    float max_val = s_reduce[0];

    // exp & sum
    float thread_sum = 0.0f;
    for (int k_idx = tid; k_idx < seq_len; k_idx += blockDim.x)
    {
        s_score[k_idx] = expf(s_score[k_idx] - max_val);
        thread_sum += s_score[k_idx];
    }
    s_reduce[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }
    float sum = s_reduce[0];

    // 正規化
    for (int k_idx = tid; k_idx < seq_len; k_idx += blockDim.x)
        s_score[k_idx] /= sum;
    __syncthreads();

    // -------------------------------------------------------
    // Step3: Attention Weight × V → Output
    // -------------------------------------------------------
    for (int d = tid; d < d_v; d += blockDim.x)
    {
        float out = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++)
            out += s_score[k_idx] * V[k_idx * d_v + d];
        Output[q_idx * d_v + d] = out;
    }
}

int main()
{
    printf("========================================\n");
    printf("  Scaled Dot-Product Attention\n");
    printf("  seq_len=%d, d_model=%d\n", SEQ_LEN, D_MODEL);
    printf("========================================\n\n");

    int QK_size = SEQ_LEN * D_MODEL;
    int S_size = SEQ_LEN * SEQ_LEN;

    // CPU側メモリ
    float* h_Q = new float[QK_size];
    float* h_K = new float[QK_size];
    float* h_V = new float[QK_size];
    float* h_out_naive = new float[QK_size]();
    float* h_out_fused = new float[QK_size]();

    // 初期化（小さい値にしてSoftmaxを安定させる）
    srand(42);
    for (int i = 0; i < QK_size; i++)
    {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // GPU側メモリ
    float* d_Q, * d_K, * d_V;
    float* d_Score, * d_Weight, * d_Output;
    cudaMalloc(&d_Q, QK_size * sizeof(float));
    cudaMalloc(&d_K, QK_size * sizeof(float));
    cudaMalloc(&d_V, QK_size * sizeof(float));
    cudaMalloc(&d_Score, S_size * sizeof(float));  // ナイーブ用中間バッファ
    cudaMalloc(&d_Weight, S_size * sizeof(float));  // ナイーブ用中間バッファ
    cudaMalloc(&d_Output, QK_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q, QK_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, QK_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, QK_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 tile_block(TILE_SIZE, TILE_SIZE);
    dim3 tile_grid(
        (SEQ_LEN + TILE_SIZE - 1) / TILE_SIZE,
        (SEQ_LEN + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------------------------------------------
    // ① ナイーブ実装（3カーネルを順番に呼ぶ）
    // -------------------------------------------------------
    printf("① ナイーブ実装（3カーネル分離）\n");
    printf("   Q×K^T → Scale → Softmax → ×V\n\n");

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
    {
        // Step1: Score = Q × K^T
        matmul << <tile_grid, tile_block >> > (d_Q, d_K, d_Score, SEQ_LEN, SEQ_LEN, D_MODEL);

        // Step2: Score /= sqrt(d_k)
        scale << <(S_size + 255) / 256, 256 >> > (
            d_Score, S_size, 1.0f / sqrtf((float)D_MODEL));

        // Step3: Weight = softmax(Score)
        softmax << <SEQ_LEN, TILE_SIZE* TILE_SIZE >> > (d_Score, SEQ_LEN, SEQ_LEN);

        // Step4: Output = Weight × V
        matmul << <tile_grid, tile_block >> > (d_Score, d_V, d_Output, SEQ_LEN, SEQ_LEN, D_MODEL);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    cudaMemcpy(h_out_naive, d_Output, QK_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (1000回合計)\n\n", ms_naive);

    // -------------------------------------------------------
    // ② Fused実装（1カーネルで完結）
    // -------------------------------------------------------
    printf("② Fused実装（1カーネル）\n");
    printf("   Q×K^T + Scale + Softmax + ×V を1カーネルで\n\n");

    // Shared Memory サイズ計算
    int smem_size = (SEQ_LEN + 256) * sizeof(float);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
    {
        fused_attention << <SEQ_LEN, 256, smem_size >> > (
            d_Q, d_K, d_V, d_Output,
            SEQ_LEN, D_MODEL, D_MODEL);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_fused = 0;
    cudaEventElapsedTime(&ms_fused, start, stop);
    cudaMemcpy(h_out_fused, d_Output, QK_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  時間: %.3f ms (1000回合計)\n\n", ms_fused);

    // ---- 速度比較 ----
    printf("速度比較:\n");
    printf("  ナイーブ（4カーネル）: %.3f ms\n", ms_naive);
    printf("  Fused  （1カーネル）: %.3f ms\n", ms_fused);
    printf("  速度向上: %.2fx\n\n", ms_naive / ms_fused);

    // ---- 精度検証 ----
    float max_diff = 0.0f;
    for (int i = 0; i < QK_size; i++)
        max_diff = fmaxf(max_diff, fabsf(h_out_naive[i] - h_out_fused[i]));
    printf("ナイーブ vs Fused 最大誤差: %e %s\n\n",
        max_diff, max_diff < 1e-4f ? "✓ OK" : "要確認");

    // ---- 出力表示 ----
    printf("Attention出力（先頭トークンの先頭8次元）:\n");
    printf("  ナイーブ: ");
    for (int i = 0; i < 8; i++) printf("%7.4f ", h_out_naive[i]);
    printf("\n  Fused:   ");
    for (int i = 0; i < 8; i++) printf("%7.4f ", h_out_fused[i]);
    printf("\n\n");

    printf("ここまで来たらLLMのAttentionカーネルが書けます！\n");
    printf("次: Multi-Head Attention → Transformer Block へ\n");

    // 後片付け
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_Q; delete[] h_K; delete[] h_V;
    delete[] h_out_naive; delete[] h_out_fused;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Score); cudaFree(d_Weight); cudaFree(d_Output);
    cudaDeviceReset();
    return 0;
}
