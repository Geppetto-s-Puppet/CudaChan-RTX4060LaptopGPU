// ============================================================
// GPT-2 推論エンジン（CUDA + Tensor Core + cuBLAS）
// 継続会話・ログ保存・top-k サンプリング対応版
//
// GPT-2 small アーキテクチャ:
//   vocab=50257  ctx=1024  embd=768  heads=12  layers=12
//
// 実行: Cuda-Chan.exe [gpt2_weights.bin gpt2_tokenizer.bin]
// 終了: "exit" または "quit" を入力
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
#include <random>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

#pragma comment(lib, "cublas.lib")

// ============================================================
// ★ ユーザー設定 ★  ← ここを自由に編集してね
// ============================================================

// デバッグ情報をコンソールに表示するか
//   true  → logits の min/max や Top-5 トークンを毎ステップ表示
//   false → クリーンな会話出力のみ（通常はこっち）
static const bool DEBUG_MODE = false;

// AIの名前・性格・口調（システムプロンプト）
// GPT-2 は chat fine-tune されていないので完璧ではないが、
// フォーマットを守ることで会話らしい出力になる
static const std::string AI_NAME = "Cuda-Chan";
static const std::string SYSTEM_PROMPT =
"The following is a conversation with " + AI_NAME + ".\n"
+ AI_NAME + " is a helpful AI assistant.\n"
+ AI_NAME + " always stays on topic and answers directly.\n\n"
+ AI_NAME + ": Hello! I am " + AI_NAME + ", your AI assistant. How can I help you?\n\n";

// サンプリング設定
static const int   TOP_K = 40;    // 候補トークン数（10〜100 推奨）
static const float TEMPERATURE = 0.6f; // 温度（0.5=保守的 / 0.85=バランス / 1.2=創造的）

// 生成設定
static const int         MAX_GEN = 200;                    // 1ターンの最大生成トークン数
static const std::string LOG_FILE = "conversation_log.txt"; // 会話ログ保存先ファイル名

// ============================================================
// GPT-2 small 定数（変更不要）
// ============================================================
#define N_VOCAB  50257
#define N_CTX    1024
#define N_EMBD   768
#define N_HEAD   12
#define N_LAYER  12
#define D_HEAD   64      // N_EMBD / N_HEAD
#define D_FF     3072    // N_EMBD * 4

#define CUDA_CHECK(x) do { \
    cudaError_t e=(x); \
    if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t s=(x); \
    if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error %d (%s:%d)\n",s,__FILE__,__LINE__);exit(1);} \
} while(0)

// ============================================================
// ログ保存ヘルパー（追記モード）
// ============================================================
static void append_log(const std::string& text) {
    std::ofstream f(LOG_FILE, std::ios::app);
    if (f.is_open()) f << text;
}

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
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
    float mean = s[0] / cols;

    // var
    float var = 0;
    for (int j = tid; j < cols; j += 256) { float d = xr[j] - mean; var += d * d; }
    s[tid] = var; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
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

// QKV 分割: qkv[seq, 3*N_EMBD] → Q,K,V [N_HEAD, seq, D_HEAD]
__global__ void k_qkv_split(
    const float* qkv,
    __half* Q, __half* K, __half* V,
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

// Causal Softmax（未来トークンをマスク）
__global__ void k_softmax_causal(float* S, int seq_len) {
    __shared__ float s[256];
    int h = blockIdx.x, q = blockIdx.y, tid = threadIdx.x;
    float* row = S + (h * seq_len + q) * seq_len;

    for (int k = tid; k < seq_len; k += 256)
        if (k > q) row[k] = -1e10f;
    __syncthreads();

    float tmax = -FLT_MAX;
    for (int k = tid; k <= q; k += 256) tmax = fmaxf(tmax, row[k]);
    s[tid] = tmax; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] = fmaxf(s[tid], s[tid + st]); __syncthreads(); }

    float tsum = 0;
    for (int k = tid; k < seq_len; k += 256) { row[k] = expf(row[k] - s[0]); tsum += row[k]; }
    s[tid] = tsum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }

    for (int k = tid; k < seq_len; k += 256) row[k] /= s[0];
}

// Attention ヘッド結合: ctx_t[N_HEAD, seq, D_HEAD] → out[seq, N_EMBD]
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
    __half* wte;   // [N_VOCAB, N_EMBD]
    __half* wpe;   // [N_CTX,   N_EMBD]

    float* ln1w[N_LAYER], * ln1b[N_LAYER];
    __half* qkv_w[N_LAYER];  float* qkv_b[N_LAYER];
    __half* cp_w[N_LAYER];   float* cp_b[N_LAYER];
    float* ln2w[N_LAYER], * ln2b[N_LAYER];
    __half* fc_w[N_LAYER];   float* fc_b[N_LAYER];
    __half* pp_w[N_LAYER];   float* pp_b[N_LAYER];

    float* lnfw, * lnfb;
};

// ============================================================
// 推論バッファ
// ============================================================
struct GPT2B {
    float* x, * ln_out, * qkv_out;
    __half* Q, * K, * V;
    float* scores;
    __half* scores16;
    float* ctx_t, * ctx, * proj_out;
    float* fc_out, * ff_out, * lnf_out;
    float* logits;
    __half* x16, * ctx16, * fc16, * last16;
    int* d_tok;
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
    auto f32 = [](float** p, int n) { cudaMalloc(p, n * 4); cudaMemset(*p, 0, n * 4); };
    auto f16 = [](__half** p, int n) { cudaMalloc(p, n * 2); cudaMemset(*p, 0, n * 2); };

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
// BPEトークナイザ
// ============================================================
struct Tokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id2tok;

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open tokenizer: %s\n", path); return false; }
        unsigned n;
        if (fread(&n, 4, 1, f) != 1) { fclose(f); return false; }
        if (n > N_VOCAB + 1000) { fprintf(stderr, "Bad token count: %u\n", n); fclose(f); return false; }

        id2tok.resize(n);
        vocab.reserve(n);

        for (unsigned i = 0; i < n; i++) {
            unsigned len;
            if (fread(&len, 4, 1, f) != 1 || len > 1000) {
                fprintf(stderr, "Bad token at %u\n", i); fclose(f); return false;
            }
            std::string s(len, '\0');
            if (fread(&s[0], 1, len, f) != len) {
                fprintf(stderr, "Read fail at token %u\n", i); fclose(f); return false;
            }
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

    // 生テキストとして返す（コンソール出力用）
    const std::string& decode(int id) const {
        static const std::string unk = "?";
        if (id < 0 || id >= (int)id2tok.size()) return unk;
        return id2tok[id];
    }

    // デバッグ用（非表示文字をエスケープ表示）
    std::string decode_escaped(int id) const {
        if (id < 0 || id >= (int)id2tok.size()) return "?";
        std::string out;
        for (unsigned char c : id2tok[id]) {
            if (c < 32 || c > 126) {
                char buf[8]; snprintf(buf, 8, "\\x%02x", c);
                out += buf;
            }
            else {
                out += (char)c;
            }
        }
        return out;
    }
};

// ============================================================
// Top-k サンプリング
// ============================================================
static std::mt19937 g_rng(std::random_device{}());

static int sample_topk(const std::vector<float>& logits) {
    int k = std::min(TOP_K, N_VOCAB);

    // 温度スケール + Top-k 抽出
    std::vector<std::pair<float, int>> scored(N_VOCAB);
    for (int i = 0; i < N_VOCAB; i++)
        scored[i] = { logits[i] / TEMPERATURE, i };

    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

    // Softmax（top-k のみ）
    float maxv = scored[0].first, sumexp = 0.f;
    for (int i = 0; i < k; i++) {
        scored[i].first = expf(scored[i].first - maxv);
        sumexp += scored[i].first;
    }
    for (int i = 0; i < k; i++) scored[i].first /= sumexp;

    // 累積分布サンプリング
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(g_rng), cumsum = 0.f;
    for (int i = 0; i < k; i++) {
        cumsum += scored[i].first;
        if (r <= cumsum) return scored[i].second;
    }
    return scored[0].second;
}

// ============================================================
// GPT-2 フォワードパス（1トークン生成）
// ============================================================
int forward(cublasHandle_t cb, GPT2W& w, GPT2B& b,
    const std::vector<int>& tokens, const Tokenizer& tok)
{
    int seq = (int)tokens.size();
    if (seq > N_CTX) { fprintf(stderr, "seq too long\n"); return -1; }

    for (int i = 0; i < seq; i++) {
        if (tokens[i] < 0 || tokens[i] >= N_VOCAB) {
            fprintf(stderr, "Invalid token %d at pos %d\n", tokens[i], i);
            return -1;
        }
    }

    cudaMemcpy(b.d_tok, tokens.data(), seq * sizeof(int), cudaMemcpyHostToDevice);
    k_embed << <seq, N_EMBD >> > (b.d_tok, w.wte, w.wpe, b.x, seq);

    const float scale = 1.0f / sqrtf((float)D_HEAD);

    for (int l = 0; l < N_LAYER; l++) {
        // Self-Attention
        k_layernorm << <seq, 256 >> > (b.x, b.ln_out, w.ln1w[l], w.ln1b[l], seq, N_EMBD);
        k_f32_to_f16 << <(seq * N_EMBD + 255) / 256, 256 >> > (b.ln_out, b.x16, seq * N_EMBD);
        gemm(cb, b.x16, w.qkv_w[l], b.qkv_out, seq, 3 * N_EMBD, N_EMBD);
        k_add_bias << <(seq * 3 * N_EMBD + 255) / 256, 256 >> > (b.qkv_out, w.qkv_b[l], seq, 3 * N_EMBD);

        k_qkv_split << <dim3(N_HEAD, seq), D_HEAD >> > (b.qkv_out, b.Q, b.K, b.V, seq);

        batched_qk(cb, b.Q, b.K, b.scores, seq, scale);
        k_softmax_causal << <dim3(N_HEAD, seq), 256 >> > (b.scores, seq);
        k_f32_to_f16 << <(N_HEAD * seq * seq + 255) / 256, 256 >> > (b.scores, b.scores16, N_HEAD * seq * seq);
        batched_av(cb, b.scores16, b.V, b.ctx_t, seq);

        k_ctx_merge << <dim3(N_HEAD, seq), D_HEAD >> > (b.ctx_t, b.ctx, seq);

        k_f32_to_f16 << <(seq * N_EMBD + 255) / 256, 256 >> > (b.ctx, b.ctx16, seq * N_EMBD);
        gemm(cb, b.ctx16, w.cp_w[l], b.proj_out, seq, N_EMBD, N_EMBD);
        k_add_bias << <(seq * N_EMBD + 255) / 256, 256 >> > (b.proj_out, w.cp_b[l], seq, N_EMBD);
        k_add << <(seq * N_EMBD + 255) / 256, 256 >> > (b.x, b.proj_out, seq * N_EMBD);

        // Feed-Forward Network
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

    // LM Head
    k_layernorm << <1, 256 >> > (b.x + (seq - 1) * N_EMBD, b.lnf_out, w.lnfw, w.lnfb, 1, N_EMBD);
    k_f32_to_f16 << <(N_EMBD + 255) / 256, 256 >> > (b.lnf_out, b.last16, N_EMBD);
    gemm_bt(cb, b.last16, w.wte, b.logits, 1, N_VOCAB, N_EMBD);
    cudaDeviceSynchronize();

    std::vector<float> logits_h(N_VOCAB);
    cudaMemcpy(logits_h.data(), b.logits, N_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);

    // デバッグ表示（DEBUG_MODE == true のときのみ）
    if (DEBUG_MODE) {
        float minv = *std::min_element(logits_h.begin(), logits_h.end());
        float maxv = *std::max_element(logits_h.begin(), logits_h.end());
        printf("\n[DEBUG] step=%zu  logits min=%.3f max=%.3f\n", tokens.size(), minv, maxv);

        std::vector<int> idx(N_VOCAB);
        for (int i = 0; i < N_VOCAB; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
            [&logits_h](int a, int b) { return logits_h[a] > logits_h[b]; });
        printf("  Top5:");
        for (int i = 0; i < 5; i++)
            printf(" [%d '%s' %.3f]", idx[i], tok.decode_escaped(idx[i]).c_str(), logits_h[idx[i]]);
        printf("\n");
    }

    return sample_topk(logits_h);
}

// ============================================================
// メイン（継続会話ループ）
// ============================================================
int main(int argc, char* argv[]) {
    // Windows: コンソールを UTF-8 に設定
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    const char* weights_path = "gpt2_weights.bin";
    const char* tokenizer_path = "gpt2_tokenizer.bin";
    if (argc >= 3) { weights_path = argv[1]; tokenizer_path = argv[2]; }

    // cuBLAS 初期化
    cublasHandle_t cb;
    CUBLAS_CHECK(cublasCreate(&cb));
    CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_TENSOR_OP_MATH));

    // 重みロード・バッファ確保
    GPT2W w; GPT2B b;
    if (!load_weights(w, weights_path)) return 1;
    alloc_buffers(b);

    // トークナイザロード
    Tokenizer tok;
    if (!tok.load(tokenizer_path)) return 1;

    // デバッグ時のみトークナイザ診断を表示
    if (DEBUG_MODE) {
        printf("\n[DEBUG] === Tokenizer Diagnosis ===\n");
        printf("First 20 tokens:\n");
        for (int i = 0; i < 20; i++)
            printf("  %5d: '%s'\n", i, tok.decode_escaped(i).c_str());
        auto t = tok.encode("hello");
        printf("encode('hello') → %d → '%s'\n", t[0], tok.decode(t[0]).c_str());
    }

    // ログファイルにセッション開始を記録
    {
        std::time_t now = std::time(nullptr);
        char tbuf[64];
        strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        append_log("\n========================================\n");
        append_log(std::string("Session: ") + tbuf + "\n");
        append_log("[System] " + SYSTEM_PROMPT + "\n");
    }

    // ヘッダ表示
    printf("\n================================================\n");
    printf("  %s  (GPT-2 Tensor Core 推論エンジン)\n", AI_NAME.c_str());
    printf("  'exit' / 'quit' で終了\n");
    printf("  ログ保存先: %s\n", LOG_FILE.c_str());
    printf("================================================\n\n");

    // システムプロンプトのトークン列（コンテキスト切り詰め時に常に先頭に保持）
    const std::vector<int> sys_tokens = tok.encode(SYSTEM_PROMPT);
    const int sys_len = (int)sys_tokens.size();
    const int max_input_len = N_CTX - MAX_GEN - 10;

    // 会話履歴（文字列として管理）
    std::string history = SYSTEM_PROMPT;

    // ストリーミング出力用のストップパターン
    // モデルがユーザーやAIの発言を自分で生成し始めたら止める
    const std::vector<std::string> STOP_PATTERNS = {
        "\nYou:",
        std::string("\n") + AI_NAME + ":"
    };
    const int LOOKAHEAD = 20; // ストップパターンを先読みするバッファ長

    // ============================================================
    // 会話ループ
    // ============================================================
    while (true) {
        printf("You: ");
        fflush(stdout);

        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            printf("\n(EOF検出。終了します)\n");
            break;
        }

        // 終了コマンド
        if (user_input == "exit" || user_input == "quit") {
            printf("会話を終了します。ログ: %s\n", LOG_FILE.c_str());
            break;
        }
        if (user_input.empty()) continue;

        // ログに記録
        append_log("You: " + user_input + "\n");

        // プロンプト組み立て
        std::string prompt = history + "You: " + user_input + "\n" + AI_NAME + ":";
        std::vector<int> tokens = tok.encode(prompt);

        // コンテキスト長超過時は古い会話を削除（システムプロンプトは保持）
        if ((int)tokens.size() > max_input_len) {
            int keep = max_input_len - sys_len;
            std::vector<int> trimmed = sys_tokens;
            if (keep > 0) {
                trimmed.insert(trimmed.end(), tokens.end() - keep, tokens.end());
            }
            else {
                trimmed = std::vector<int>(tokens.end() - max_input_len, tokens.end());
            }
            tokens = trimmed;
            if (DEBUG_MODE)
                printf("[DEBUG] Context trimmed to %d tokens\n", (int)tokens.size());
        }

        // ============================================================
        // トークン生成（ストリーミング出力 + ルックアヘッドストップ）
        // ============================================================
        printf("%s:", AI_NAME.c_str());
        fflush(stdout);

        std::string response;   // 生成済み（確定済み）テキスト
        std::string pending;    // 未表示のルックアヘッドバッファ
        bool stopped = false;

        for (int step = 0; step < MAX_GEN && !stopped; step++) {
            if ((int)tokens.size() >= N_CTX) break;

            int next = forward(cb, w, b, tokens, tok);
            if (next < 0 || next == 50256) { stopped = true; break; }

            tokens.push_back(next);
            pending += tok.decode(next);

            // ストップパターンチェック
            size_t stop_pos = std::string::npos;
            for (const auto& pat : STOP_PATTERNS) {
                size_t p = pending.find(pat);
                if (p != std::string::npos) {
                    if (stop_pos == std::string::npos || p < stop_pos)
                        stop_pos = p;
                }
            }

            if (stop_pos != std::string::npos) {
                // ストップ位置までを表示して終了
                std::string safe = pending.substr(0, stop_pos);
                printf("%s", safe.c_str());
                fflush(stdout);
                response += safe;
                stopped = true;
                break;
            }

            // ルックアヘッド分を残してそれ以前を出力
            if ((int)pending.size() > LOOKAHEAD) {
                std::string safe = pending.substr(0, pending.size() - LOOKAHEAD);
                printf("%s", safe.c_str());
                fflush(stdout);
                response += safe;
                pending = pending.substr(pending.size() - LOOKAHEAD);
            }
        }

        // 残りのバッファを全部出力
        if (!stopped && !pending.empty()) {
            printf("%s", pending.c_str());
            fflush(stdout);
            response += pending;
        }

        printf("\n\n");

        // 会話履歴を更新（次ターンのコンテキストに含める）
        history += "You: " + user_input + "\n"
            + AI_NAME + ": " + response + "\n\n";

        // ログに記録
        append_log(AI_NAME + ": " + response + "\n\n");
    }

    cublasDestroy(cb);
    return 0;
}
