#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

// ============================================================
// Step7: Transformer Block（GPT-2の1層と同じ構造）
//
// 構造:
//   Input
//     ↓
//   Multi-Head Attention  （Step6のAttentionをH個並列）
//     ↓
//   Add & LayerNorm        （残差接続 + 正規化）
//     ↓
//   FFN（Linear→GELU→Linear）
//     ↓
//   Add & LayerNorm
//     ↓
//   Output
//
// パラメータ（GPT-2 smallスケール）:
//   seq_len  = 64  （トークン数）
//   d_model  = 64  （埋め込み次元）
//   n_heads  = 4   （Attentionヘッド数）
//   d_head   = 16  （d_model / n_heads = 64/4）
//   d_ff     = 256 （FFNの中間次元、d_model×4が慣例）
// ============================================================

#define SEQ_LEN   64
#define D_MODEL   64
#define N_HEADS   4
#define D_HEAD    (D_MODEL / N_HEADS)   // 16
#define D_FF      (D_MODEL * 4)         // 256
#define TILE_SIZE 16

// ============================================================
// カーネル群
// ============================================================

// 行列積 C = A × B  [M×K] × [K×N] = [M×N]
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

// 行列加算（残差接続）: out += residual
__global__ void add_residual(float* out, const float* residual, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] += residual[i];
}

// ============================================================
// LayerNorm
// 各トークン（行）を独立に正規化する
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// LLMでは全層で使われる最重要正規化手法
// ============================================================
__global__ void layernorm(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int rows, int cols, float eps)
{
    __shared__ float s_buf[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* x = input + row * cols;
    float* y = output + row * cols;

    // mean を計算
    float thread_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x)
        thread_sum += x[j];
    s_buf[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) s_buf[tid] += s_buf[tid + stride];
        __syncthreads();
    }
    float mean = s_buf[0] / cols;

    // variance を計算
    float thread_var = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x)
    {
        float diff = x[j] - mean;
        thread_var += diff * diff;
    }
    s_buf[tid] = thread_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) s_buf[tid] += s_buf[tid + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(s_buf[0] / cols + eps);

    // 正規化 + affine変換
    for (int j = tid; j < cols; j += blockDim.x)
        y[j] = (x[j] - mean) * inv_std * gamma[j] + beta[j];
}

// ============================================================
// GELU活性化関数
// ReLUより滑らかでGPTシリーズが採用
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
// ============================================================
__global__ void gelu(float* x, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = x[i];
    float c = 0.7978845608f; // sqrt(2/pi)
    x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

// ============================================================
// Multi-Head Attention
// d_modelをn_headsに分割して並列でAttentionを計算
// 各ヘッドが異なる「注目パターン」を学習できる
// ============================================================
__global__ void multi_head_attention(
    const float* Q_all,   // [seq_len, d_model] = 全ヘッド分まとめたQ
    const float* K_all,   // [seq_len, d_model]
    const float* V_all,   // [seq_len, d_model]
    float* Output,        // [seq_len, d_model]
    int seq_len, int n_heads, int d_head)
{
    // blockIdx.x = クエリトークンのインデックス
    // blockIdx.y = ヘッドのインデックス
    int q_idx = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int d_model = n_heads * d_head;

    if (q_idx >= seq_len || head >= n_heads) return;

    // このヘッドが担当するオフセット
    int head_offset = head * d_head;

    extern __shared__ float smem[];
    float* s_score = smem;               // [seq_len]
    float* s_reduce = smem + seq_len;     // [blockDim.x]

    float scale = 1.0f / sqrtf((float)d_head);

    // Step1: Q[q_idx, head] × K[:, head]^T → Score
    for (int k_idx = tid; k_idx < seq_len; k_idx += blockDim.x)
    {
        float dot = 0.0f;
        for (int d = 0; d < d_head; d++)
            dot += Q_all[q_idx * d_model + head_offset + d]
            * K_all[k_idx * d_model + head_offset + d];
        s_score[k_idx] = dot * scale;
    }
    __syncthreads();

    // Step2: Softmax
    float tmax = -FLT_MAX;
    for (int k = tid; k < seq_len; k += blockDim.x)
        tmax = fmaxf(tmax, s_score[k]);
    s_reduce[tid] = tmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + s]);
        __syncthreads();
    }
    float max_val = s_reduce[0];

    float tsum = 0.0f;
    for (int k = tid; k < seq_len; k += blockDim.x)
    {
        s_score[k] = expf(s_score[k] - max_val);
        tsum += s_score[k];
    }
    s_reduce[tid] = tsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    for (int k = tid; k < seq_len; k += blockDim.x)
        s_score[k] /= s_reduce[0];
    __syncthreads();

    // Step3: Weight × V[head] → Output[q_idx, head]
    for (int d = tid; d < d_head; d += blockDim.x)
    {
        float out = 0.0f;
        for (int k = 0; k < seq_len; k++)
            out += s_score[k] * V_all[k * d_model + head_offset + d];
        Output[q_idx * d_model + head_offset + d] = out;
    }
}

// ============================================================
// ユーティリティ
// ============================================================
void init_random(float* p, int n, float scale = 0.02f)
{
    for (int i = 0; i < n; i++)
        p[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
}

void init_ones(float* p, int n) { for (int i = 0; i < n; i++) p[i] = 1.0f; }
void init_zeros(float* p, int n) { for (int i = 0; i < n; i++) p[i] = 0.0f; }

void gpu_alloc_copy(float** d, const float* h, int n)
{
    cudaMalloc(d, n * sizeof(float));
    if (h) cudaMemcpy(*d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    else   cudaMemset(*d, 0, n * sizeof(float));
}

int main()
{
    printf("========================================\n");
    printf("  Transformer Block（GPT-2 small相当）\n");
    printf("  seq=%d  d_model=%d  heads=%d  d_ff=%d\n",
        SEQ_LEN, D_MODEL, N_HEADS, D_FF);
    printf("========================================\n\n");

    // ---- ホスト側メモリ（重み・入力） ----
    int X_sz = SEQ_LEN * D_MODEL;
    int W_qkv = D_MODEL * D_MODEL;   // Q/K/V投影行列
    int W_ff1 = D_MODEL * D_FF;      // FFN 1層目
    int W_ff2 = D_FF * D_MODEL;   // FFN 2層目

    float* h_X = new float[X_sz];    // 入力トークン埋め込み

    // 重み行列（本来は学習済みモデルからロードする）
    float* h_Wq = new float[W_qkv];
    float* h_Wk = new float[W_qkv];
    float* h_Wv = new float[W_qkv];
    float* h_Wo = new float[W_qkv];   // Attention出力投影
    float* h_W1 = new float[W_ff1];
    float* h_W2 = new float[W_ff2];

    // LayerNorm パラメータ（gamma=1, beta=0で初期化）
    float* h_gamma1 = new float[D_MODEL];
    float* h_beta1 = new float[D_MODEL];
    float* h_gamma2 = new float[D_MODEL];
    float* h_beta2 = new float[D_MODEL];

    srand(42);
    init_random(h_X, X_sz);
    init_random(h_Wq, W_qkv);
    init_random(h_Wk, W_qkv);
    init_random(h_Wv, W_qkv);
    init_random(h_Wo, W_qkv);
    init_random(h_W1, W_ff1);
    init_random(h_W2, W_ff2);
    init_ones(h_gamma1, D_MODEL); init_zeros(h_beta1, D_MODEL);
    init_ones(h_gamma2, D_MODEL); init_zeros(h_beta2, D_MODEL);

    // ---- GPU側メモリ ----
    float* d_X, * d_Wq, * d_Wk, * d_Wv, * d_Wo, * d_W1, * d_W2;
    float* d_gamma1, * d_beta1, * d_gamma2, * d_beta2;
    float* d_Q, * d_K, * d_V;          // QKV投影後
    float* d_Attn_out;                // Attention出力
    float* d_Attn_proj;               // Attention出力投影後
    float* d_LN1_out;                 // LayerNorm1後
    float* d_FF1_out;                 // FFN 1層後
    float* d_FF2_out;                 // FFN 2層後
    float* d_LN2_out;                 // LayerNorm2後（ブロック出力）
    float* d_residual;                // 残差接続用コピー

    gpu_alloc_copy(&d_X, h_X, X_sz);
    gpu_alloc_copy(&d_Wq, h_Wq, W_qkv);
    gpu_alloc_copy(&d_Wk, h_Wk, W_qkv);
    gpu_alloc_copy(&d_Wv, h_Wv, W_qkv);
    gpu_alloc_copy(&d_Wo, h_Wo, W_qkv);
    gpu_alloc_copy(&d_W1, h_W1, W_ff1);
    gpu_alloc_copy(&d_W2, h_W2, W_ff2);
    gpu_alloc_copy(&d_gamma1, h_gamma1, D_MODEL);
    gpu_alloc_copy(&d_beta1, h_beta1, D_MODEL);
    gpu_alloc_copy(&d_gamma2, h_gamma2, D_MODEL);
    gpu_alloc_copy(&d_beta2, h_beta2, D_MODEL);

    gpu_alloc_copy(&d_Q, nullptr, X_sz);
    gpu_alloc_copy(&d_K, nullptr, X_sz);
    gpu_alloc_copy(&d_V, nullptr, X_sz);
    gpu_alloc_copy(&d_Attn_out, nullptr, X_sz);
    gpu_alloc_copy(&d_Attn_proj, nullptr, X_sz);
    gpu_alloc_copy(&d_LN1_out, nullptr, X_sz);
    gpu_alloc_copy(&d_FF1_out, nullptr, SEQ_LEN * D_FF);
    gpu_alloc_copy(&d_FF2_out, nullptr, X_sz);
    gpu_alloc_copy(&d_LN2_out, nullptr, X_sz);
    gpu_alloc_copy(&d_residual, nullptr, X_sz);

    // グリッドサイズ
    dim3 tile_block(TILE_SIZE, TILE_SIZE);
    auto tgrid = [](int M, int N) {
        return dim3((N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE);
        };

    // タイマー
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ============================================================
    // Transformer Block フォワードパス
    // ============================================================
    printf("フォワードパス実行中...\n\n");
    cudaEventRecord(start);

    for (int iter = 0; iter < 1000; iter++)
    {
        // --------------------------------------------------------
        // 1. QKV 投影
        //    Q = X × Wq,  K = X × Wk,  V = X × Wv
        //    各トークンの埋め込みをQ/K/V空間に変換する
        // --------------------------------------------------------
        matmul << <tgrid(SEQ_LEN, D_MODEL), tile_block >> > (d_X, d_Wq, d_Q, SEQ_LEN, D_MODEL, D_MODEL);
        matmul << <tgrid(SEQ_LEN, D_MODEL), tile_block >> > (d_X, d_Wk, d_K, SEQ_LEN, D_MODEL, D_MODEL);
        matmul << <tgrid(SEQ_LEN, D_MODEL), tile_block >> > (d_X, d_Wv, d_V, SEQ_LEN, D_MODEL, D_MODEL);

        // --------------------------------------------------------
        // 2. Multi-Head Attention
        //    4ヘッドが並列に異なる注目パターンを計算
        // --------------------------------------------------------
        int smem = (SEQ_LEN + 256) * sizeof(float);
        dim3 mha_grid(SEQ_LEN, N_HEADS);
        multi_head_attention << <mha_grid, 256, smem >> > (
            d_Q, d_K, d_V, d_Attn_out,
            SEQ_LEN, N_HEADS, D_HEAD);

        // --------------------------------------------------------
        // 3. Attention出力投影 + 残差接続 + LayerNorm
        //    out_proj = Attn_out × Wo
        //    x = LayerNorm(x + out_proj)   ← 残差接続！
        // --------------------------------------------------------
        matmul << <tgrid(SEQ_LEN, D_MODEL), tile_block >> > (
            d_Attn_out, d_Wo, d_Attn_proj, SEQ_LEN, D_MODEL, D_MODEL);

        cudaMemcpy(d_residual, d_X, X_sz * sizeof(float), cudaMemcpyDeviceToDevice);
        add_residual << <(X_sz + 255) / 256, 256 >> > (d_Attn_proj, d_residual, X_sz);

        layernorm << <SEQ_LEN, 256 >> > (
            d_Attn_proj, d_LN1_out,
            d_gamma1, d_beta1,
            SEQ_LEN, D_MODEL, 1e-5f);

        // --------------------------------------------------------
        // 4. FFN（Feed-Forward Network）
        //    ff1 = GELU(LN1_out × W1)   [seq, d_model] → [seq, d_ff]
        //    ff2 = ff1 × W2             [seq, d_ff]    → [seq, d_model]
        //    x = LayerNorm(x + ff2)     残差接続
        // --------------------------------------------------------
        matmul << <tgrid(SEQ_LEN, D_FF), tile_block >> > (d_LN1_out, d_W1, d_FF1_out, SEQ_LEN, D_FF, D_MODEL);
        gelu << <(SEQ_LEN * D_FF + 255) / 256, 256 >> > (d_FF1_out, SEQ_LEN * D_FF);
        matmul << <tgrid(SEQ_LEN, D_MODEL), tile_block >> > (d_FF1_out, d_W2, d_FF2_out, SEQ_LEN, D_MODEL, D_FF);

        add_residual << <(X_sz + 255) / 256, 256 >> > (d_FF2_out, d_LN1_out, X_sz);

        layernorm << <SEQ_LEN, 256 >> > (
            d_FF2_out, d_LN2_out,
            d_gamma2, d_beta2,
            SEQ_LEN, D_MODEL, 1e-5f);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // ---- 結果表示 ----
    float* h_output = new float[X_sz];
    cudaMemcpy(h_output, d_LN2_out, X_sz * sizeof(float), cudaMemcpyDeviceToHost);

    printf("実行時間: %.3f ms (1000回合計)\n", ms);
    printf("1回あたり: %.4f ms\n\n", ms / 1000.0f);

    printf("出力（先頭3トークン × 先頭8次元）:\n");
    for (int t = 0; t < 3; t++)
    {
        printf("  token[%d]: ", t);
        for (int d = 0; d < 8; d++)
            printf("%7.4f ", h_output[t * D_MODEL + d]);
        printf("\n");
    }

    printf("\n");
    printf("処理ステップまとめ:\n");
    printf("  1. QKV投影     X × Wq/Wk/Wv\n");
    printf("  2. MHA         %dヘッド並列Attention\n", N_HEADS);
    printf("  3. 残差+LN1    x = LayerNorm(x + Attn)\n");
    printf("  4. FFN         Linear(%d→%d)→GELU→Linear(%d→%d)\n",
        D_MODEL, D_FF, D_FF, D_MODEL);
    printf("  5. 残差+LN2    x = LayerNorm(x + FFN)\n");
    printf("\nこれを%d層積み上げたのがGPT-2 small！\n", 12);

    // 後片付け
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_X; delete[] h_Wq; delete[] h_Wk; delete[] h_Wv;
    delete[] h_Wo; delete[] h_W1; delete[] h_W2;
    delete[] h_gamma1; delete[] h_beta1;
    delete[] h_gamma2; delete[] h_beta2;
    delete[] h_output;
    cudaFree(d_X);   cudaFree(d_Wq);  cudaFree(d_Wk);  cudaFree(d_Wv);
    cudaFree(d_Wo);  cudaFree(d_W1);  cudaFree(d_W2);
    cudaFree(d_Q);   cudaFree(d_K);   cudaFree(d_V);
    cudaFree(d_Attn_out); cudaFree(d_Attn_proj);
    cudaFree(d_LN1_out);  cudaFree(d_FF1_out);
    cudaFree(d_FF2_out);  cudaFree(d_LN2_out);
    cudaFree(d_residual);
    cudaFree(d_gamma1); cudaFree(d_beta1);
    cudaFree(d_gamma2); cudaFree(d_beta2);
    cudaDeviceReset();
    return 0;
}

//## 今回の全体像
//```
//X[64×64]
//│
//├─── × Wq → Q ─┐
//├─── × Wk → K ─┤→ Multi - Head Attention（4ヘッド）→ Attn_out
//└─── × Wv → V ─┘         ↓ × Wo
//+ 残差(X)
//LayerNorm → LN1_out
//↓ × W1
//GELU
//↓ × W2
//+ 残差(LN1_out)
//LayerNorm → Output