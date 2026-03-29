// ============================================================
// GPT-2 推論エンジン（CUDA + Tensor Core + cuBLAS）
//
// GPT-2 small アーキテクチャ:
//   vocab=50257  ctx=1024  embd=768  heads=12  layers=12
//
// Tensor Core使用箇所:
//   cublasGemmEx / cublasGemmStridedBatchedEx
//   with CUBLAS_COMPUTE_32F_FAST_16F
//   → 全GEMM（QKV投影/Attention/FFN/LMHead）でTensor Core起動
//
// 実行: Cuda-Chan.exe gpt2_weights.bin gpt2_tokenizer.bin
// ============================================================

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#pragma comment(lib, "cublas.lib")

// ============================================================
// GPT-2 small 定数
// ============================================================
#define N_VOCAB  50257
#define N_CTX    1024
#define N_EMBD   768
#define N_HEAD   12
#define N_LAYER  12
#define D_HEAD   64      // N_EMBD / N_HEAD
#define D_FF     3072    // N_EMBD * 4
#define MAX_GEN  256

#define CUDA_CHECK(x) do { \
    cudaError_t e=(x); \
    if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t s=(x); \
    if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error %d (%s:%d)\n",s,__FILE__,__LINE__);exit(1);} \
} while(0)

// ============================================================
// CUDA カーネル群
// ============================================================

// FP32 → FP16 変換
__global__ void k_f32_to_f16(const float* s, __half* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2half(s[i]);
}

// LayerNorm: y = (x - mean) / sqrt(var + 1e-5) * gamma + beta
__global__ void k_layernorm(
    const float* x, float* y,
    const float* gamma, const float* beta,
    int rows, int cols)
{
    __shared__ float s[256];
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;

    // mean
    float sum = 0;
    for (int j = tid; j < cols; j += 256) sum += xr[j];
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st)s[tid] += s[tid + st]; __syncthreads(); }
    float mean = s[0] / cols;

    // var
    float var = 0;
    for (int j = tid; j < cols; j += 256) { float d = xr[j] - mean; var += d * d; }
    s[tid] = var; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st)s[tid] += s[tid + st]; __syncthreads(); }
    float inv = rsqrtf(s[0] / cols + 1e-5f);

    for (int j = tid; j < cols; j += 256)
        yr[j] = (xr[j] - mean) * inv * gamma[j] + beta[j];
}

// GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))
__global__ void k_gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}

// 要素加算 a += b
__global__ void k_add(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// バイアス加算 x[row, col] += bias[col]
__global__ void k_add_bias(float* x, const float* b, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) x[i] += b[i % cols];
}

// Embedding + Positional Encoding → FP32
// wte[token_id, :] + wpe[pos, :] をそれぞれ FP16 で持つ
__global__ void k_embed(
    const int* tokens,
    const __half* wte,  // [N_VOCAB, N_EMBD]
    const __half* wpe,  // [N_CTX,   N_EMBD]
    float* out,         // [seq, N_EMBD]
    int seq_len)
{
    int pos = blockIdx.x, d = threadIdx.x;
    if (pos >= seq_len || d >= N_EMBD) return;
    int tok = tokens[pos];
    out[pos * N_EMBD + d] = __half2float(wte[tok * N_EMBD + d])
        + __half2float(wpe[pos * N_EMBD + d]);
}

// QKV を分割してヘッド次元でまとめ直す
// qkv[seq, 3*N_EMBD] → Q,K,V それぞれ [N_HEAD, seq, D_HEAD]
// Tensor Core batched GEMM のための連続化
__global__ void k_qkv_split(
    const float* qkv,  // [seq, 3*N_EMBD]
    __half* Q,         // [N_HEAD, seq, D_HEAD]
    __half* K,
    __half* V,
    int seq_len)
{
    int h = blockIdx.x, s = blockIdx.y, d = threadIdx.x;
    if (h >= N_HEAD || s >= seq_len || d >= D_HEAD) return;
    int oi = h * seq_len * D_HEAD + s * D_HEAD + d;
    int ii = s * 3 * N_EMBD + h * D_HEAD + d;
    Q[oi] = __float2half(qkv[ii]);
    K[oi] = __float2half(qkv[ii + N_EMBD]);
    V[oi] = __float2half(qkv[ii + 2 * N_EMBD]);
}

// Causal Softmax: scores [N_HEAD, seq, seq] にマスク + Softmax
// 未来トークン (k > q) を -inf にする → GPTの自己回帰性
__global__ void k_softmax_causal(float* S, int seq_len) {
    __shared__ float s[256];
    int h = blockIdx.x, q = blockIdx.y, tid = threadIdx.x;
    float* row = S + (h * seq_len + q) * seq_len;

    // Causal mask
    for (int k = tid; k < seq_len; k += 256)
        if (k > q) row[k] = -1e10f;
    __syncthreads();

    // Max
    float tmax = -FLT_MAX;
    for (int k = tid; k <= q; k += 256) tmax = fmaxf(tmax, row[k]);
    s[tid] = tmax; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st)s[tid] = fmaxf(s[tid], s[tid + st]); __syncthreads(); }

    // Exp + sum
    float tsum = 0;
    for (int k = tid; k < seq_len; k += 256) { row[k] = expf(row[k] - s[0]); tsum += row[k]; }
    s[tid] = tsum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st)s[tid] += s[tid + st]; __syncthreads(); }

    for (int k = tid; k < seq_len; k += 256) row[k] /= s[0];
}

// Attention出力をヘッドから結合
// ctx_t [N_HEAD, seq, D_HEAD] → out [seq, N_EMBD]
__global__ void k_ctx_merge(
    const float* ctx_t,
    float* out,
    int seq_len)
{
    int h = blockIdx.x, s = blockIdx.y, d = threadIdx.x;
    if (h >= N_HEAD || s >= seq_len || d >= D_HEAD) return;
    out[s * N_EMBD + h * D_HEAD + d] = ctx_t[h * seq_len * D_HEAD + s * D_HEAD + d];
}

// ============================================================
// モデル重み
// ============================================================
struct GPT2W {
    __half* wte;   // [N_VOCAB, N_EMBD]  トークン埋め込み
    __half* wpe;   // [N_CTX,   N_EMBD]  位置埋め込み

    // 各レイヤー
    float* ln1w[N_LAYER], * ln1b[N_LAYER];   // LayerNorm1
    __half* qkv_w[N_LAYER];  // [N_EMBD, 3*N_EMBD]  QKV投影
    float* qkv_b[N_LAYER];  // [3*N_EMBD]
    __half* cp_w[N_LAYER];   // [N_EMBD, N_EMBD]  Attn出力投影
    float* cp_b[N_LAYER];   // [N_EMBD]
    float* ln2w[N_LAYER], * ln2b[N_LAYER];   // LayerNorm2
    __half* fc_w[N_LAYER];   // [N_EMBD, D_FF]    FFN 1層目
    float* fc_b[N_LAYER];   // [D_FF]
    __half* pp_w[N_LAYER];   // [D_FF,   N_EMBD]  FFN 2層目
    float* pp_b[N_LAYER];   // [N_EMBD]

    float* lnfw, * lnfb;     // 最終 LayerNorm
};

// ============================================================
// 推論バッファ
// ============================================================
struct GPT2B {
    float* x;          // [N_CTX, N_EMBD]  現在の隠れ状態
    float* ln_out;     // LayerNorm出力
    float* qkv_out;    // [N_CTX, 3*N_EMBD]
    __half* Q;          // [N_HEAD, N_CTX, D_HEAD]
    __half* K;
    __half* V;
    float* scores;     // [N_HEAD, N_CTX, N_CTX]  Attention Score
    __half* scores16;   // FP16版（Tensor Core入力用）
    float* ctx_t;      // [N_HEAD, N_CTX, D_HEAD]  Attention出力
    float* ctx;        // [N_CTX, N_EMBD]  ヘッド結合後
    float* proj_out;   // Attn投影出力
    float* fc_out;     // [N_CTX, D_FF]
    float* ff_out;     // [N_CTX, N_EMBD]
    float* lnf_out;    // 最終LN出力
    float* logits;     // [N_VOCAB]  次トークンの確率分布

    // cuBLAS FP16入力用バッファ
    __half* x16;        // [N_CTX, N_EMBD]
    __half* ctx16;
    __half* fc16;       // [N_CTX, D_FF]
    __half* last16;     // [N_EMBD]  最後のトークンのみ

    int* d_tok;      // [N_CTX]  GPU上のトークン列
};

// ============================================================
// 重みロードヘルパー
// ============================================================
static __half* load_f16(FILE* f, int n) {
    float* h = new float[n]; fread(h, 4, n, f);
    float* d32; cudaMalloc(&d32, n * 4);
    __half* d16; cudaMalloc(&d16, n * 2);
    cudaMemcpy(d32, h, n * 4, cudaMemcpyHostToDevice);
    k_f32_to_f16 << <(n + 255) / 256, 256 >> > (d32, d16, n);
    cudaFree(d32); delete[] h; return d16;
}
static float* load_f32(FILE* f, int n) {
    float* h = new float[n]; fread(h, 4, n, f);
    float* d; cudaMalloc(&d, n * 4);
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    delete[] h; return d;
}

bool load_weights(GPT2W& w, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return false; }
    unsigned hdr[6]; fread(hdr, 4, 6, f);
    if (hdr[0] != 0x47505432) { fprintf(stderr, "Bad magic\n"); return false; }
    printf("GPT-2 small: vocab=%u ctx=%u embd=%u heads=%u layers=%u\n",
        hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]);

    w.wte = load_f16(f, N_VOCAB * N_EMBD);
    w.wpe = load_f16(f, N_CTX * N_EMBD);

    for (int l = 0; l < N_LAYER; l++) {
        w.ln1w[l] = load_f32(f, N_EMBD);
        w.ln1b[l] = load_f32(f, N_EMBD);
        w.qkv_w[l] = load_f16(f, N_EMBD * 3 * N_EMBD);
        w.qkv_b[l] = load_f32(f, 3 * N_EMBD);
        w.cp_w[l] = load_f16(f, N_EMBD * N_EMBD);
        w.cp_b[l] = load_f32(f, N_EMBD);
        w.ln2w[l] = load_f32(f, N_EMBD);
        w.ln2b[l] = load_f32(f, N_EMBD);
        w.fc_w[l] = load_f16(f, N_EMBD * D_FF);
        w.fc_b[l] = load_f32(f, D_FF);
        w.pp_w[l] = load_f16(f, D_FF * N_EMBD);
        w.pp_b[l] = load_f32(f, N_EMBD);
        if (l % 4 == 3) printf("  layer %d/%d loaded\n", l + 1, N_LAYER);
    }
    w.lnfw = load_f32(f, N_EMBD);
    w.lnfb = load_f32(f, N_EMBD);
    fclose(f);
    printf("重みロード完了（GPU上: FP16）\n");
    return true;
}

void alloc_buffers(GPT2B& b) {
    auto f32 = [](float** p, int n) {cudaMalloc(p, n * 4); cudaMemset(*p, 0, n * 4); };
    auto f16 = [](__half** p, int n) {cudaMalloc(p, n * 2); cudaMemset(*p, 0, n * 2); };

    f32(&b.x, N_CTX * N_EMBD);
    f32(&b.ln_out, N_CTX * N_EMBD);
    f32(&b.qkv_out, N_CTX * 3 * N_EMBD);
    f16(&b.Q, N_HEAD * N_CTX * D_HEAD);
    f16(&b.K, N_HEAD * N_CTX * D_HEAD);
    f16(&b.V, N_HEAD * N_CTX * D_HEAD);
    f32(&b.scores, N_HEAD * N_CTX * N_CTX);
    f16(&b.scores16, N_HEAD * N_CTX * N_CTX);
    f32(&b.ctx_t, N_HEAD * N_CTX * D_HEAD);
    f32(&b.ctx, N_CTX * N_EMBD);
    f32(&b.proj_out, N_CTX * N_EMBD);
    f32(&b.fc_out, N_CTX * D_FF);
    f32(&b.ff_out, N_CTX * N_EMBD);
    f32(&b.lnf_out, N_CTX * N_EMBD);
    f32(&b.logits, N_VOCAB);
    f16(&b.x16, N_CTX * N_EMBD);
    f16(&b.ctx16, N_CTX * N_EMBD);
    f16(&b.fc16, N_CTX * D_FF);
    f16(&b.last16, N_EMBD);
    cudaMalloc(&b.d_tok, N_CTX * sizeof(int));
}

// ============================================================
// cuBLAS GEMM ヘルパー（Tensor Core使用）
// ============================================================

static void gemm(cublasHandle_t h,
    const __half* A, const __half* B, float* C,
    int M, int N, int K,
    float alpha = 1.f, float beta = 0.f)
{
    CUBLAS_CHECK(cublasGemmEx(h,
        CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
        &beta, C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void gemm_bt(cublasHandle_t h,
    const __half* A, const __half* B, float* C,
    int M, int N, int K)
{
    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(h,
        CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
        &beta, C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void batched_qk(cublasHandle_t h,
    const __half* Q, const __half* K, float* S,
    int seq_len, float scale)
{
    float beta = 0.f;
    long long sQK = (long long)seq_len * D_HEAD;
    long long sS = (long long)seq_len * seq_len;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, D_HEAD,
        &scale,
        K, CUDA_R_16F, D_HEAD, sQK,
        Q, CUDA_R_16F, D_HEAD, sQK,
        &beta,
        S, CUDA_R_32F, seq_len, sS,
        N_HEAD,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void batched_av(cublasHandle_t h,
    const __half* attn, const __half* V, float* ctx,
    int seq_len)
{
    float alpha = 1.f, beta = 0.f;
    long long sA = (long long)seq_len * seq_len;
    long long sV = (long long)seq_len * D_HEAD;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D_HEAD, seq_len, seq_len,
        &alpha,
        V, CUDA_R_16F, D_HEAD, sV,
        attn, CUDA_R_16F, seq_len, sA,
        &beta,
        ctx, CUDA_R_32F, D_HEAD, sV,
        N_HEAD,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ============================================================
// BPEトークナイザ（Tokenizer構造体を先に定義）
// ============================================================
struct Tokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id2tok;

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open tokenizer: %s\n", path); return false; }
        unsigned n; fread(&n, 4, 1, f);
        id2tok.resize(n);
        vocab.reserve(n);
        for (unsigned i = 0; i < n; i++) {
            unsigned len; fread(&len, 4, 1, f);
            std::string s(len, '\0');
            fread(&s[0], 1, len, f);
            id2tok[i] = s;
            vocab[s] = i;
        }
        fclose(f);
        printf("Tokenizer loaded: %u tokens\n", n);
        return true;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> out;
        size_t i = 0;
        while (i < text.size()) {
            bool found = false;
            for (int len = (int)std::min(text.size() - i, (size_t)20); len > 0; len--) {
                auto it = vocab.find(text.substr(i, len));
                if (it != vocab.end()) {
                    out.push_back(it->second);
                    i += len;
                    found = true;
                    break;
                }
            }
            if (!found) { out.push_back(vocab.count("?") ? vocab["?"] : 0); i++; }
        }
        return out;
    }

    std::string decode(int id) {
        if (id < 0 || id >= (int)id2tok.size()) return "?";
        return id2tok[id];
    }
};

// ============================================================
// GPT-2 フォワードパス（1トークン生成）
// ============================================================
int forward(cublasHandle_t cb, GPT2W& w, GPT2B& b, const std::vector<int>& tokens, Tokenizer& tok) {
    int seq = (int)tokens.size();
    if (seq > N_CTX) { fprintf(stderr, "seq too long\n"); return -1; }

    cudaMemcpy(b.d_tok, tokens.data(), seq * sizeof(int), cudaMemcpyHostToDevice);

    k_embed << <seq, N_EMBD >> > (b.d_tok, w.wte, w.wpe, b.x, seq);

    float scale = 1.0f / sqrtf((float)D_HEAD);

    for (int l = 0; l < N_LAYER; l++) {
        k_layernorm << <seq, 256 >> > (b.x, b.ln_out, w.ln1w[l], w.ln1b[l], seq, N_EMBD);
        k_f32_to_f16 << <(seq * N_EMBD + 255) / 256, 256 >> > (b.ln_out, b.x16, seq * N_EMBD);
        gemm(cb, b.x16, w.qkv_w[l], b.qkv_out, seq, 3 * N_EMBD, N_EMBD);
        k_add_bias << <(seq * 3 * N_EMBD + 255) / 256, 256 >> > (b.qkv_out, w.qkv_b[l], seq, 3 * N_EMBD);

        dim3 gQKV(N_HEAD, seq), bQKV(D_HEAD);
        k_qkv_split << <gQKV, bQKV >> > (b.qkv_out, b.Q, b.K, b.V, seq);

        batched_qk(cb, b.Q, b.K, b.scores, seq, scale);
        k_softmax_causal << <dim3(N_HEAD, seq), 256 >> > (b.scores, seq);
        k_f32_to_f16 << <(N_HEAD * seq * seq + 255) / 256, 256 >> > (b.scores, b.scores16, N_HEAD * seq * seq);
        batched_av(cb, b.scores16, b.V, b.ctx_t, seq);

        k_ctx_merge << <dim3(N_HEAD, seq), D_HEAD >> > (b.ctx_t, b.ctx, seq);

        k_f32_to_f16 << <(seq * N_EMBD + 255) / 256, 256 >> > (b.ctx, b.ctx16, seq * N_EMBD);
        gemm(cb, b.ctx16, w.cp_w[l], b.proj_out, seq, N_EMBD, N_EMBD);
        k_add_bias << <(seq * N_EMBD + 255) / 256, 256 >> > (b.proj_out, w.cp_b[l], seq, N_EMBD);
        k_add << <(seq * N_EMBD + 255) / 256, 256 >> > (b.x, b.proj_out, seq * N_EMBD);

        k_layernorm << <seq, 256 >> > (b.x, b.ln_out, w.ln2w[l], w.ln2b[l], seq, N_EMBD);
        k_f32_to_f16 << <(seq * N_EMBD + 255) / 256, 256 >> > (b.ln_out, b.x16, seq * N_EMBD);
        gemm(cb, b.x16, w.fc_w[l], b.fc_out, seq, D_FF, N_EMBD);
        k_add_bias << <(seq * D_FF + 255) / 256, 256 >> > (b.fc_out, w.fc_b[l], seq, D_FF);

        k_gelu << <(seq * D_FF + 255) / 256, 256 >> > (b.fc_out, seq * D_FF);

        k_f32_to_f16 << <(seq * D_FF + 255) / 256, 256 >> > (b.fc_out, b.fc16, seq * D_FF);
        gemm(cb, b.fc16, w.pp_w[l], b.ff_out, seq, N_EMBD, D_FF);
        k_add_bias << <(seq * N_EMBD + 255) / 256, 256 >> > (b.ff_out, w.pp_b[l], seq, N_EMBD);
        k_add << <(seq * N_EMBD + 255) / 256, 256 >> > (b.x, b.ff_out, seq * N_EMBD);
    }

    k_layernorm << <1, 256 >> > (b.x + (seq - 1) * N_EMBD, b.lnf_out, w.lnfw, w.lnfb, 1, N_EMBD);
    k_f32_to_f16 << <(N_EMBD + 255) / 256, 256 >> > (b.lnf_out, b.last16, N_EMBD);
    gemm_bt(cb, b.last16, w.wte, b.logits, 1, N_VOCAB, N_EMBD);

    cudaDeviceSynchronize();

    std::vector<float> logits_h(N_VOCAB);
    cudaMemcpy(logits_h.data(), b.logits, N_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);

    // ===== logits 診断（最初の3ステップのみ） =====
    if (tokens.size() <= 3) {
        float minval = *std::min_element(logits_h.begin(), logits_h.end());
        float maxval = *std::max_element(logits_h.begin(), logits_h.end());
        printf("\n[DEBUG] Logits stats (step %zu):\n", tokens.size());
        printf("  Min: %f, Max: %f\n", minval, maxval);

        // Top 5 token を手動で抽出
        std::vector<int> top5_ids;
        for (int i = 0; i < N_VOCAB; i++) {
            top5_ids.push_back(i);
        }
        std::partial_sort(top5_ids.begin(), top5_ids.begin() + 5, top5_ids.end(),
            [&logits_h](int a, int b) { return logits_h[a] > logits_h[b]; });

        printf("  Top 5 tokens:\n");
        for (int i = 0; i < 5; i++) {
            int id = top5_ids[i];
            printf("    %d: logit=%f, decoded='%s'\n", id, logits_h[id], tok.decode(id).c_str());
        }
    }
    // ===== 診断終了 =====

    return (int)(std::max_element(logits_h.begin(), logits_h.end()) - logits_h.begin());
}

// ============================================================
// メイン
// ============================================================
int main(int argc, char* argv[]) {
    const char* weights_path = "gpt2_weights.bin";
    const char* tokenizer_path = "gpt2_tokenizer.bin";

    if (argc >= 3) {
        weights_path = argv[1];
        tokenizer_path = argv[2];
    }

    cublasHandle_t cb;
    CUBLAS_CHECK(cublasCreate(&cb));
    CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_TENSOR_OP_MATH));

    GPT2W w; GPT2B b;
    if (!load_weights(w, weights_path)) return 1;
    alloc_buffers(b);

    Tokenizer tok;
    if (!tok.load(tokenizer_path)) return 1;

    printf("\n=== C++ Tokenizer Diagnosis ===\n");
    printf("First 20 tokens from id2tok:\n");
    for (int i = 0; i < 20; i++) {
        std::string s = tok.id2tok[i];
        printf("  %5d: '", i);
        for (char c : s) {
            if (c < 32 || c > 126) printf("\\x%02x", (unsigned char)c);
            else if (c == '\'') printf("\\'");
            else printf("%c", c);
        }
        printf("'\n");
    }
    printf("\nEncode test:\n");
    std::string test = "hello";
    auto test_tokens = tok.encode(test);
    printf("  Input: 'hello'\n  Tokens: ");
    for (int t : test_tokens) printf("%d ", t);
    printf("\n");

    printf("\n=== GPT-2 Inference (Tensor Core / cuBLAS) ===\n");
    printf("Prompt: ");
    std::string prompt;
    std::getline(std::cin, prompt);
    if (prompt.empty()) prompt = "Hello, world!";

    std::vector<int> tokens = tok.encode(prompt);
    printf("Tokens: %zu\n", tokens.size());

    printf("\n%s", prompt.c_str());
    fflush(stdout);

    for (int step = 0; step < MAX_GEN; step++) {
        if ((int)tokens.size() >= N_CTX) break;
        int next = forward(cb, w, b, tokens, tok);

        if (next < 0) break;
        tokens.push_back(next);
        printf("%s", tok.decode(next).c_str());
        fflush(stdout);
        if (next == 50256) break;
    }
    printf("\n\n生成完了 (%d tokens)\n", (int)tokens.size());

    cublasDestroy(cb);
    return 0;
}
