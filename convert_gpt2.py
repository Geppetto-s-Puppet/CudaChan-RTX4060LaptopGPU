#!/usr/bin/env python3
"""
HuggingFaceからGPT-2をダウンロード・正確なサイズで保存（修正版）
"""

import os
import struct
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

OUTPUT_DIR = "."
TOKENIZER_FILE = "gpt2_tokenizer.bin"
WEIGHTS_FILE = "gpt2_weights.bin"

N_VOCAB = 50257
N_CTX = 1024
N_EMBD = 768
N_HEAD = 12
N_LAYER = 12
D_FF = N_EMBD * 4  # 3072

print("=" * 70)
print("GPT-2 Download & Convert (Fixed)")
print("=" * 70)

print("\n[1] Downloading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(f"✓ Tokenizer: {len(tokenizer)} tokens")

print(f"[2] Saving {TOKENIZER_FILE}...")
with open(os.path.join(OUTPUT_DIR, TOKENIZER_FILE), 'wb') as f:
    f.write(struct.pack('<I', len(tokenizer)))
    for token_id in range(len(tokenizer)):
        token_str = tokenizer.decode([token_id])
        token_bytes = token_str.encode('utf-8')
        f.write(struct.pack('<I', len(token_bytes)))
        f.write(token_bytes)
        if (token_id + 1) % 10000 == 0:
            print(f"  {token_id + 1} / {len(tokenizer)}")
print(f"✓ Tokenizer saved: {os.path.getsize(os.path.join(OUTPUT_DIR, TOKENIZER_FILE)) / 1024 / 1024:.2f} MB")

print(f"\n[3] Downloading model...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
print("✓ Model loaded")

# ============================================================
# Conv1D の実際の weight shape 確認
# Conv1D(nf, nx) → weight.shape = (nx, nf)
# ============================================================
block0 = model.transformer.h[0]
print(f"\n[Weight shape verification]")
print(f"  c_attn.weight : {tuple(block0.attn.c_attn.weight.shape)}")  # [768, 2304]
print(f"  c_proj.weight : {tuple(block0.attn.c_proj.weight.shape)}")  # [768, 768]
print(f"  c_fc.weight   : {tuple(block0.mlp.c_fc.weight.shape)}")     # [768, 3072]
print(f"  c_proj(mlp)   : {tuple(block0.mlp.c_proj.weight.shape)}")   # [3072, 768]

def write_array(f, arr, name=""):
    arr_f32 = arr.astype(np.float32)
    f.write(arr_f32.tobytes())
    if name:
        print(f"  {name}: {arr_f32.shape} → {arr_f32.nbytes / 1024 / 1024:.2f} MB")
    return arr_f32.nbytes

print(f"\n[4] Saving {WEIGHTS_FILE}...")
total_bytes = 0

with open(os.path.join(OUTPUT_DIR, WEIGHTS_FILE), 'wb') as f:
    # ヘッダ
    magic = 0x47505432
    header = struct.pack('<IIIIII', magic, N_VOCAB, N_CTX, N_EMBD, N_HEAD, N_LAYER)
    f.write(header)
    total_bytes += len(header)

    # wte: [N_VOCAB, N_EMBD]
    total_bytes += write_array(f, model.transformer.wte.weight.detach().cpu().numpy(), "wte")

    # wpe: [N_CTX, N_EMBD]
    total_bytes += write_array(f, model.transformer.wpe.weight.detach().cpu().numpy(), "wpe")

    for layer_idx in range(N_LAYER):
        block = model.transformer.h[layer_idx]

        # LayerNorm 1: [N_EMBD]
        total_bytes += write_array(f, block.ln_1.weight.detach().cpu().numpy())
        total_bytes += write_array(f, block.ln_1.bias.detach().cpu().numpy())

        # QKV projection
        # Conv1D(3*N_EMBD, N_EMBD) → weight.shape = [N_EMBD, 3*N_EMBD]
        # C++ gemm(x[seq,N_EMBD], W[N_EMBD, 3*N_EMBD]) → そのまま保存
        qkv_w = block.attn.c_attn.weight.detach().cpu().numpy()  # [N_EMBD, 3*N_EMBD]
        qkv_b = block.attn.c_attn.bias.detach().cpu().numpy()    # [3*N_EMBD]
        total_bytes += write_array(f, qkv_w)   # ← .T 不要
        total_bytes += write_array(f, qkv_b)

        # Attention output projection
        # Conv1D(N_EMBD, N_EMBD) → weight.shape = [N_EMBD, N_EMBD]
        cp_w = block.attn.c_proj.weight.detach().cpu().numpy()  # [N_EMBD, N_EMBD]
        cp_b = block.attn.c_proj.bias.detach().cpu().numpy()
        total_bytes += write_array(f, cp_w)    # ← .T 不要
        total_bytes += write_array(f, cp_b)

        # LayerNorm 2
        total_bytes += write_array(f, block.ln_2.weight.detach().cpu().numpy())
        total_bytes += write_array(f, block.ln_2.bias.detach().cpu().numpy())

        # FFN 1st layer
        # Conv1D(D_FF, N_EMBD) → weight.shape = [N_EMBD, D_FF]
        # C++ gemm(x[seq,N_EMBD], W[N_EMBD, D_FF]) → そのまま保存
        fc_w = block.mlp.c_fc.weight.detach().cpu().numpy()  # [N_EMBD, D_FF]
        fc_b = block.mlp.c_fc.bias.detach().cpu().numpy()
        total_bytes += write_array(f, fc_w)    # ← .T 不要
        total_bytes += write_array(f, fc_b)

        # FFN 2nd layer
        # Conv1D(N_EMBD, D_FF) → weight.shape = [D_FF, N_EMBD]
        # C++ gemm(x[seq,D_FF], W[D_FF, N_EMBD]) → そのまま保存
        pp_w = block.mlp.c_proj.weight.detach().cpu().numpy()  # [D_FF, N_EMBD]
        pp_b = block.mlp.c_proj.bias.detach().cpu().numpy()
        total_bytes += write_array(f, pp_w)    # ← .T 不要
        total_bytes += write_array(f, pp_b)

        if (layer_idx + 1) % 4 == 0:
            print(f"  Layer {layer_idx + 1}/{N_LAYER} complete")

    # Final LayerNorm
    total_bytes += write_array(f, model.transformer.ln_f.weight.detach().cpu().numpy(), "lnfw")
    total_bytes += write_array(f, model.transformer.ln_f.bias.detach().cpu().numpy(), "lnfb")

print(f"\n✓ Weights saved: {os.path.getsize(os.path.join(OUTPUT_DIR, WEIGHTS_FILE)) / 1024 / 1024:.2f} MB")
print(f"  Expected total: ~548 MB (FP32)")

# ============================================================
# PyTorch で logits を確認（デバッグ用）
# ============================================================
print("\n[5] PyTorch sanity check...")
import torch
from transformers import GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained('gpt2')
test_input = tok.encode("Hello", return_tensors="pt")
with torch.no_grad():
    out = model(test_input)
logits = out.logits[0, -1]
top5 = torch.topk(logits, 5)
print("  Top 5 next tokens after 'Hello':")
for val, idx in zip(top5.values, top5.indices):
    print(f"    {idx.item():6d}: {val.item():.4f}  '{tok.decode([idx.item()])}'")

print("\n" + "=" * 70)
print("✓ Complete! Re-run C++ inference with new weights.")
print("=" * 70)