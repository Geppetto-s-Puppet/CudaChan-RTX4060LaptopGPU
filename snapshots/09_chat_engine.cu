#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <string>
#include <vector>
#include <iostream>

// ============================================================
// Step9: チャットエンジン（会話履歴保持）
//
// 追加要素:
//   - メインループでcinを受け付ける
//   - 会話履歴をトークン列として保持
//   - [User]: / [GPT]: のターン管理
//   - コンテキストウィンドウが溢れたら古い履歴を削除
// ============================================================

#define VOCAB_SIZE  65
#define D_MODEL     64
#define N_HEADS     4
#define D_HEAD      (D_MODEL / N_HEADS)
#define D_FF        (D_MODEL * 4)
#define N_LAYERS    4
#define MAX_SEQ     64
#define TILE_SIZE   16
#define MAX_GEN     80   // 1ターンの最大生成トークン数
#define EOS_TOKEN   0    // 改行を会話終端として使う

// ============================================================
// 語彙
// ============================================================
static const char* VOCAB =
"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

int char_to_token(char c) {
    for (int i = 0; i < VOCAB_SIZE; i++)
        if (VOCAB[i] == c) return i;
    return 1; // スペースにフォールバック
}
char token_to_char(int t) {
    if (t < 0 || t >= VOCAB_SIZE) return '?';
    return VOCAB[t];
}

// ============================================================
// カーネル群（Step8から流用）
// ============================================================

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
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        s_A[ty][tx] = (row < M && t * TILE_SIZE + tx < K) ? A[row * K + t * TILE_SIZE + tx] : 0.0f;
        s_B[ty][tx] = (col < N && t * TILE_SIZE + ty < K) ? B[(t * TILE_SIZE + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) sum += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

__global__ void add_inplace(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void layernorm(
    const float* x, float* y,
    const float* gamma, const float* beta,
    int rows, int cols, float eps)
{
    __shared__ float s[256];
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) sum += xr[j];
    s[tid] = sum; __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (tid < st)s[tid] += s[tid + st]; __syncthreads(); }
    float mean = s[0] / cols;
    float var = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) { float d = xr[j] - mean; var += d * d; }
    s[tid] = var; __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (tid < st)s[tid] += s[tid + st]; __syncthreads(); }
    float inv = rsqrtf(s[0] / cols + eps);
    for (int j = tid; j < cols; j += blockDim.x) yr[j] = (xr[j] - mean) * inv * gamma[j] + beta[j];
}

__global__ void gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], c = 0.7978845608f;
    x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

__global__ void causal_mha(
    const float* Q, const float* K, const float* V,
    float* Out, int seq_len, int n_heads, int d_head)
{
    int q_idx = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int dm = n_heads * d_head;
    if (q_idx >= seq_len) return;
    int hoff = head * d_head;
    extern __shared__ float smem[];
    float* s_score = smem;
    float* s_reduce = smem + seq_len;
    float scale = 1.0f / sqrtf((float)d_head);

    for (int k = tid; k < seq_len; k += blockDim.x) {
        if (k > q_idx) { s_score[k] = -1e9f; continue; }
        float dot = 0.0f;
        for (int d = 0; d < d_head; d++)
            dot += Q[q_idx * dm + hoff + d] * K[k * dm + hoff + d];
        s_score[k] = dot * scale;
    }
    __syncthreads();

    float tmax = -FLT_MAX;
    for (int k = tid; k <= q_idx; k += blockDim.x) tmax = fmaxf(tmax, s_score[k]);
    s_reduce[tid] = tmax; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s)s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + s]); __syncthreads(); }
    float mv = s_reduce[0];

    float ts = 0.0f;
    for (int k = tid; k < seq_len; k += blockDim.x) { s_score[k] = expf(s_score[k] - mv); ts += s_score[k]; }
    s_reduce[tid] = ts; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s)s_reduce[tid] += s_reduce[tid + s]; __syncthreads(); }
    for (int k = tid; k < seq_len; k += blockDim.x) s_score[k] /= s_reduce[0];
    __syncthreads();

    for (int d = tid; d < d_head; d += blockDim.x) {
        float o = 0.0f;
        for (int k = 0; k <= q_idx; k++) o += s_score[k] * V[k * dm + hoff + d];
        Out[q_idx * dm + hoff + d] = o;
    }
}

__global__ void embed_and_pos(
    const int* tokens, const float* tok_emb, const float* pos_emb,
    float* out, int seq_len, int d_model)
{
    int pos = blockIdx.x, tid = threadIdx.x;
    if (pos >= seq_len) return;
    int tok = tokens[pos];
    for (int d = tid; d < d_model; d += blockDim.x)
        out[pos * d_model + d] = tok_emb[tok * d_model + d] + pos_emb[pos * d_model + d];
}

__global__ void lm_head(
    const float* hidden, const float* weight,
    float* logits, int vocab_size, int d_model)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vocab_size) return;
    float s = 0.0f;
    for (int d = 0; d < d_model; d++) s += hidden[d] * weight[v * d_model + d];
    logits[v] = s;
}

// ============================================================
// モデル構造体
// ============================================================
struct GPTWeights {
    float* tok_emb;
    float* pos_emb;
    float* Wq[N_LAYERS], * Wk[N_LAYERS], * Wv[N_LAYERS], * Wo[N_LAYERS];
    float* W1[N_LAYERS], * W2[N_LAYERS];
    float* gamma1[N_LAYERS], * beta1[N_LAYERS];
    float* gamma2[N_LAYERS], * beta2[N_LAYERS];
    float* gamma_f, * beta_f;
};

struct GPTBuffers {
    float* x, * residual, * Q, * K, * V;
    float* attn_out, * attn_proj, * ln1_out;
    float* ff1_out, * ff2_out, * ln2_out;
    float* logits;
    int* d_tokens;
};

void alloc_weights(GPTWeights& w) {
    cudaMalloc(&w.tok_emb, VOCAB_SIZE * D_MODEL * sizeof(float));
    cudaMalloc(&w.pos_emb, MAX_SEQ * D_MODEL * sizeof(float));
    for (int l = 0; l < N_LAYERS; l++) {
        cudaMalloc(&w.Wq[l], D_MODEL * D_MODEL * sizeof(float));
        cudaMalloc(&w.Wk[l], D_MODEL * D_MODEL * sizeof(float));
        cudaMalloc(&w.Wv[l], D_MODEL * D_MODEL * sizeof(float));
        cudaMalloc(&w.Wo[l], D_MODEL * D_MODEL * sizeof(float));
        cudaMalloc(&w.W1[l], D_MODEL * D_FF * sizeof(float));
        cudaMalloc(&w.W2[l], D_FF * D_MODEL * sizeof(float));
        cudaMalloc(&w.gamma1[l], D_MODEL * sizeof(float));
        cudaMalloc(&w.beta1[l], D_MODEL * sizeof(float));
        cudaMalloc(&w.gamma2[l], D_MODEL * sizeof(float));
        cudaMalloc(&w.beta2[l], D_MODEL * sizeof(float));
    }
    cudaMalloc(&w.gamma_f, D_MODEL * sizeof(float));
    cudaMalloc(&w.beta_f, D_MODEL * sizeof(float));
}

void init_weights_random(GPTWeights& w) {
    auto fill = [](float* d, int n, float sc) {
        float* h = new float[n];
        for (int i = 0; i < n; i++) h[i] = ((float)rand() / RAND_MAX - 0.5f) * sc;
        cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h;
        };
    auto ones = [](float* d, int n) {
        float* h = new float[n];
        for (int i = 0; i < n; i++) h[i] = 1.0f;
        cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h;
        };
    fill(w.tok_emb, VOCAB_SIZE * D_MODEL, 0.02f);
    fill(w.pos_emb, MAX_SEQ * D_MODEL, 0.01f);
    for (int l = 0; l < N_LAYERS; l++) {
        fill(w.Wq[l], D_MODEL * D_MODEL, 0.02f);
        fill(w.Wk[l], D_MODEL * D_MODEL, 0.02f);
        fill(w.Wv[l], D_MODEL * D_MODEL, 0.02f);
        fill(w.Wo[l], D_MODEL * D_MODEL, 0.02f);
        fill(w.W1[l], D_MODEL * D_FF, 0.02f);
        fill(w.W2[l], D_FF * D_MODEL, 0.02f);
        ones(w.gamma1[l], D_MODEL); cudaMemset(w.beta1[l], 0, D_MODEL * sizeof(float));
        ones(w.gamma2[l], D_MODEL); cudaMemset(w.beta2[l], 0, D_MODEL * sizeof(float));
    }
    ones(w.gamma_f, D_MODEL);
    cudaMemset(w.beta_f, 0, D_MODEL * sizeof(float));
}

void alloc_buffers(GPTBuffers& b) {
    cudaMalloc(&b.x, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.residual, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.Q, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.K, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.V, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.attn_out, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.attn_proj, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.ln1_out, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.ff1_out, MAX_SEQ * D_FF * sizeof(float));
    cudaMalloc(&b.ff2_out, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.ln2_out, MAX_SEQ * D_MODEL * sizeof(float));
    cudaMalloc(&b.logits, VOCAB_SIZE * sizeof(float));
    cudaMalloc(&b.d_tokens, MAX_SEQ * sizeof(int));
}

// ============================================================
// フォワードパス
// ============================================================
void gpt_forward(GPTWeights& w, GPTBuffers& b, const int* tokens, int seq_len) {
    int X_sz = seq_len * D_MODEL;
    cudaMemcpy(b.d_tokens, tokens, seq_len * sizeof(int), cudaMemcpyHostToDevice);
    embed_and_pos << <seq_len, 256 >> > (b.d_tokens, w.tok_emb, w.pos_emb, b.x, seq_len, D_MODEL);

    auto tg = [&](int M, int N) {
        return dim3((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        };
    dim3 tb(TILE_SIZE, TILE_SIZE);

    for (int l = 0; l < N_LAYERS; l++) {
        matmul << <tg(seq_len, D_MODEL), tb >> > (b.x, w.Wq[l], b.Q, seq_len, D_MODEL, D_MODEL);
        matmul << <tg(seq_len, D_MODEL), tb >> > (b.x, w.Wk[l], b.K, seq_len, D_MODEL, D_MODEL);
        matmul << <tg(seq_len, D_MODEL), tb >> > (b.x, w.Wv[l], b.V, seq_len, D_MODEL, D_MODEL);

        int smem = (seq_len + 256) * sizeof(float);
        causal_mha << <dim3(seq_len, N_HEADS), 256, smem >> > (
            b.Q, b.K, b.V, b.attn_out, seq_len, N_HEADS, D_HEAD);

        matmul << <tg(seq_len, D_MODEL), tb >> > (
            b.attn_out, w.Wo[l], b.attn_proj, seq_len, D_MODEL, D_MODEL);

        cudaMemcpy(b.residual, b.x, X_sz * sizeof(float), cudaMemcpyDeviceToDevice);
        add_inplace << <(X_sz + 255) / 256, 256 >> > (b.attn_proj, b.residual, X_sz);
        layernorm << <seq_len, 256 >> > (b.attn_proj, b.ln1_out,
            w.gamma1[l], w.beta1[l], seq_len, D_MODEL, 1e-5f);

        matmul << <tg(seq_len, D_FF), tb >> > (b.ln1_out, w.W1[l], b.ff1_out, seq_len, D_FF, D_MODEL);
        gelu << <(seq_len * D_FF + 255) / 256, 256 >> > (b.ff1_out, seq_len * D_FF);
        matmul << <tg(seq_len, D_MODEL), tb >> > (b.ff1_out, w.W2[l], b.ff2_out, seq_len, D_MODEL, D_FF);

        add_inplace << <(X_sz + 255) / 256, 256 >> > (b.ff2_out, b.ln1_out, X_sz);
        layernorm << <seq_len, 256 >> > (b.ff2_out, b.ln2_out,
            w.gamma2[l], w.beta2[l], seq_len, D_MODEL, 1e-5f);

        cudaMemcpy(b.x, b.ln2_out, X_sz * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    layernorm << <seq_len, 256 >> > (b.x, b.ln2_out, w.gamma_f, w.beta_f, seq_len, D_MODEL, 1e-5f);

    float* last = b.ln2_out + (seq_len - 1) * D_MODEL;
    lm_head << <(VOCAB_SIZE + 255) / 256, 256 >> > (last, w.tok_emb, b.logits, VOCAB_SIZE, D_MODEL);
}

// ============================================================
// サンプリング
// ============================================================
int sample_temperature(float* logits, float temperature) {
    float max_l = -FLT_MAX;
    for (int i = 0; i < VOCAB_SIZE; i++) max_l = fmaxf(max_l, logits[i]);
    float sum = 0.0f;
    float probs[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = expf((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; i++) probs[i] /= sum;
    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return VOCAB_SIZE - 1;
}

// ============================================================
// 会話履歴マネージャー
// ============================================================
struct ChatHistory {
    std::vector<int> tokens;  // 全会話のトークン列

    // 文字列をトークン化して末尾に追加
    void append(const std::string& text) {
        for (char c : text)
            tokens.push_back(char_to_token(c));
    }

    // トークン数を返す
    int size() const { return (int)tokens.size(); }

    // コンテキストウィンドウに収まるよう古い履歴を削除
    void trim_to_context() {
        while ((int)tokens.size() > MAX_SEQ - 10) {
            // 先頭から1トークン削除（スライディングウィンドウ）
            tokens.erase(tokens.begin());
        }
    }

    // 現在のトークン列を配列にコピー
    int get_tokens(int* buf, int max_len) const {
        int n = std::min((int)tokens.size(), max_len);
        for (int i = 0; i < n; i++) buf[i] = tokens[i];
        return n;
    }
};

// ============================================================
// メインチャットループ
// ============================================================
int main() {
    printf("========================================\n");
    printf("  Mini GPT チャットエンジン\n");
    printf("  vocab=%d  d=%d  heads=%d  layers=%d\n",
        VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS);
    printf("========================================\n\n");

    srand((unsigned)time(nullptr));

    // モデル初期化
    printf("モデルを初期化中...\n");
    GPTWeights weights;
    GPTBuffers buffers;
    alloc_weights(weights);
    alloc_buffers(buffers);
    init_weights_random(weights);
    printf("完了！\n\n");

    printf("チャット開始（'quit' で終了）\n");
    printf("※重みがランダムなため返答は意味不明ですが\n");
    printf("  会話履歴は正しく保持・参照されています\n");
    printf("----------------------------------------\n\n");

    ChatHistory history;
    float h_logits[VOCAB_SIZE];
    int   token_buf[MAX_SEQ];
    int   turn = 0;

    // ============================================================
    // メインループ
    // ============================================================
    while (true) {
        // ---- ユーザー入力 ----
        printf("[You]: ");
        fflush(stdout);

        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input == "quit" || user_input == "exit")
            break;
        if (user_input.empty())
            continue;

        turn++;

        // 会話履歴にユーザー発言を追加
        // フォーマット: "\n[You]: {input}\n[GPT]: "
        if (turn == 1)
            history.append("[You]: " + user_input + "\n[GPT]: ");
        else
            history.append("\n[You]: " + user_input + "\n[GPT]: ");

        // コンテキストに収まるようトリム
        history.trim_to_context();

        // ---- GPT応答生成 ----
        printf("[GPT]: ");
        fflush(stdout);

        std::string response = "";
        int generated = 0;

        while (generated < MAX_GEN) {
            // 現在のコンテキストでフォワードパス
            int seq_len = history.get_tokens(token_buf, MAX_SEQ);
            if (seq_len == 0) break;

            gpt_forward(weights, buffers, token_buf, seq_len);

            // ロジット取得
            cudaMemcpy(h_logits, buffers.logits,
                VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            // サンプリング（temperature=0.8）
            int next_tok = sample_temperature(h_logits, 0.8f);
            char next_char = token_to_char(next_tok);

            // 改行が2回連続したら応答終了
            if (next_char == '\n' && !response.empty() &&
                response.back() == '\n')
                break;

            // 一定文字数で自動終了
            printf("%c", next_char);
            fflush(stdout);
            response += next_char;
            generated++;

            // 生成トークンを履歴に追加
            history.tokens.push_back(next_tok);
            history.trim_to_context();

            // 改行で文章が一区切りついたら一定確率で終了
            if (next_char == '\n' && generated > 10) {
                if ((float)rand() / RAND_MAX < 0.6f) break;
            }
        }

        printf("\n\n");

        // 統計表示（デバッグ用）
        printf("  [context: %d tokens / %d]\n\n",
            history.size(), MAX_SEQ);
    }

    printf("\nチャット終了。\n");

    // 後片付け
    cudaFree(weights.tok_emb); cudaFree(weights.pos_emb);
    for (int l = 0; l < N_LAYERS; l++) {
        cudaFree(weights.Wq[l]); cudaFree(weights.Wk[l]);
        cudaFree(weights.Wv[l]); cudaFree(weights.Wo[l]);
        cudaFree(weights.W1[l]); cudaFree(weights.W2[l]);
        cudaFree(weights.gamma1[l]); cudaFree(weights.beta1[l]);
        cudaFree(weights.gamma2[l]); cudaFree(weights.beta2[l]);
    }
    cudaFree(weights.gamma_f); cudaFree(weights.beta_f);
    cudaFree(buffers.x);       cudaFree(buffers.residual);
    cudaFree(buffers.Q);       cudaFree(buffers.K); cudaFree(buffers.V);
    cudaFree(buffers.attn_out); cudaFree(buffers.attn_proj);
    cudaFree(buffers.ln1_out); cudaFree(buffers.ff1_out);
    cudaFree(buffers.ff2_out); cudaFree(buffers.ln2_out);
    cudaFree(buffers.logits);  cudaFree(buffers.d_tokens);
    cudaDeviceReset();
    return 0;
}