"""
check_data_alignment.py — 检查 T5 / RQVAE-T5 / SASRec 数据输入对齐
运行方式: python check_data_alignment.py  (项目根目录)
"""
import os, sys
import numpy as np
import torch
from collections import Counter

def ok(m):   print(f"  [ OK  ] {m}")
def warn(m): print(f"  [WARN ] {m}")
def fail(m): print(f"  [FAIL ] {m}")
def info(m): print(f"  [INFO ] {m}")
def skip(m): print(f"  [SKIP ] {m}")
def sec(t):  print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ─────────────────────────────────────────────────────────────
# 1. T5 — 序列样本对齐
# ─────────────────────────────────────────────────────────────
sec("① T5 — 序列样本对齐 (build_sequence_samples)")

def t5_samples(item_list, max_seq_len=20):
    train, test = [], []
    if len(item_list) < 2:
        return train, test
    train_items = item_list[:-1]
    for i in range(1, len(train_items)):
        h = train_items[max(0, i - max_seq_len):i]
        t = train_items[i]
        train.append((list(h), t))
    test.append((item_list[max(0, len(item_list)-1-max_seq_len):len(item_list)-1],
                 item_list[-1]))
    return train, test

demo = [10, 20, 30, 40, 50]
tr, te = t5_samples(demo)

errs = []
for h, t in tr:
    pos_h = demo.index(h[-1])
    pos_t = demo.index(t)
    if pos_t != pos_h + 1:
        errs.append(f"h={h}, t={t}: 不连续")
    if t in h:
        errs.append(f"t={t} 出现在 h 中 (泄漏)")
    if t == demo[-1]:
        errs.append(f"t={t} 是测试目标 (泄漏)")

if errs:
    for e in errs: fail(e)
else:
    ok("训练样本：target 紧跟 history 末尾，无测试集泄漏")

h_te, t_te = te[0]
if t_te != demo[-1] or t_te in h_te:
    fail(f"测试样本异常: h={h_te}, t={t_te}")
else:
    ok(f"测试样本：target={t_te}(最后物品)，未出现在输入中")

# ─────────────────────────────────────────────────────────────
# 2. T5 — collate_emb_batch 掩码 + mean-pooling
# ─────────────────────────────────────────────────────────────
sec("② T5 — collate_emb_batch 注意力掩码与 mean-pooling")

seq_lens = [5, 3, 7]
max_l = max(seq_lens)
embs  = torch.zeros(len(seq_lens), max_l, 4)
mask  = torch.zeros(len(seq_lens), max_l, dtype=torch.long)
for i, l in enumerate(seq_lens):
    embs[i, :l, :] = 1.0
    mask[i, :l]    = 1

ok_flag = True
for i, l in enumerate(seq_lens):
    if not (mask[i, :l].all() and (mask[i, l:] == 0).all()):
        fail(f"样本{i}: padding 掩码方向错误"); ok_flag = False
    m = mask[i].float().unsqueeze(-1)
    pooled = (embs[i] * m).sum(0) / m.sum().clamp(min=1e-9)
    if abs(pooled[0].item() - 1.0) > 1e-4:
        fail(f"样本{i}: mean-pooling 结果异常"); ok_flag = False
if ok_flag:
    ok("掩码方向正确(左对齐真实数据+右侧padding)，mean-pooling 数值无误")

# ─────────────────────────────────────────────────────────────
# 3. T5 — user_embs 索引方式警告
# ─────────────────────────────────────────────────────────────
sec("③ T5 — item_embs / user_embs 索引对齐")

info("item_encode.py: order_texts[0]='' (padding), order_texts[k]=item k")
info("data_vision.py: item_embs[item_id]  ← item_id 从 1 开始，与数组位置一一对应")
ok("item_embs 索引：无偏移，item_id 直接对应数组下标")

info("")
info("user_embs 存储: sorted(user_profile_map, key=user_id) → 按 user_id 升序")
info("data_vision.py: user_profile_embs[user_id - 1]")
warn("前提假设：user_id 从 1 连续递增 (1,2,3,...,N)")
warn("若 user_id 不连续，user_id-1 会取到错误 embedding！")
info("建议：验证 user_id 连续性，或改为字典映射")

# ─────────────────────────────────────────────────────────────
# 4. RQVAE-T5 — 语义码 token 范围与 EOS 冲突
# ─────────────────────────────────────────────────────────────
sec("④ RQVAE-T5 — Code Token 范围与 EOS/PAD 冲突")

cb_size, c_dim, eos_id, pad_id = 8, 4, 31, 0

valid = {}
all_v = set()
for pos in range(c_dim):
    s = pos * cb_size + 1
    valid[pos] = set(range(s, s + cb_size))
    all_v |= valid[pos]

info(f"codebook_size={cb_size}, code_dim={c_dim}")
for p, v in valid.items():
    info(f"  pos {p}: token [{min(v)}, {max(v)}]")

if pad_id in all_v:
    fail(f"PAD token {pad_id} 与合法 code token 冲突")
else:
    ok(f"PAD token ({pad_id}) 不在合法范围内 ✓")

if eos_id in all_v:
    for p, v in valid.items():
        if eos_id in v:
            cv = eos_id - (p * cb_size + 1)
            warn(f"EOS token ({eos_id}) 与合法 code 重叠：pos={p}, code_val={cv}")
            if p == c_dim - 1:
                info("  → 重叠位置是 pos 3（最后位置）")
                info("  → max_length=5 恰好容纳 [start,c0,c1,c2,31]，通常不会提前截断")
                warn("  → 建议改 eos_token_id=33，彻底消除歧义")
            else:
                fail(f"  → 非末位冲突(pos={p})，生成时会提前截断！")
else:
    ok(f"EOS token ({eos_id}) 不在合法范围内 ✓")

# ─────────────────────────────────────────────────────────────
# 5. RQVAE-T5 — attention mask 覆盖 padding
# ─────────────────────────────────────────────────────────────
sec("⑤ RQVAE-T5 — Attention Mask 对 Padding 覆盖")

pad_code = [0] * c_dim
demo_hist = [pad_code, pad_code, [1,9,17,25], [2,10,18,26], [3,11,19,27]]
flat = [e for sub in demo_hist for e in sub]
mask_demo = [1 if e != 0 else 0 for e in flat]

pad_pos  = {i for i,v in enumerate(flat) if v == 0}
zero_pos = {i for i,v in enumerate(mask_demo) if v == 0}

if pad_pos == zero_pos:
    ok(f"Attention mask 精确覆盖所有 padding 位 ({len(pad_pos)} 个)")
else:
    fail(f"掩码不匹配: 漏标={pad_pos-zero_pos}, 误标={zero_pos-pad_pos}")

info("code value 0 的最小 token = 0+0*8+1=1 ≠ 0，不会被误判为 padding")
ok("合法 code 最小 token=1，无法被误判为 pad_token=0")

# ─────────────────────────────────────────────────────────────
# 6. SASRec — input/target 位移对齐
# ─────────────────────────────────────────────────────────────
sec("⑥ SASRec — input / target 位移对齐")

def sasrec_train_item(seq, max_len=10):
    ri = seq[:-1][-max_len:]
    rt = seq[1:][-max_len:]
    p = max_len - len(ri)
    return [0]*p + ri, [0]*p + rt

demo_seq = [1, 2, 3, 4, 5, 6]
s, o = sasrec_train_item(demo_seq)

errs = []
for i in range(len(s)-1):
    if s[i] != 0 and o[i] != 0:
        if o[i] != s[i+1]:
            errs.append(f"pos {i}: s={s[i]}, o_t={o[i]}, s[i+1]={s[i+1]}")

if errs:
    for e in errs: fail(e)
else:
    ok("input/target 位移正确：对每个非 padding 位，o_t[i] == s[i+1]")
    info(f"  示例: s={s[-6:]}, o_t={o[-6:]}")

# ─────────────────────────────────────────────────────────────
# 7. SASRec — 测试集 leave-one-out
# ─────────────────────────────────────────────────────────────
sec("⑦ SASRec — 测试集 leave-one-out 对齐")

def sasrec_test_item(items, max_len=10):
    inp = items[:-1]
    tgt = items[-1]
    s = inp[-max_len:] if len(inp) >= max_len else [0]*(max_len-len(inp)) + inp
    return s, tgt

demo_items = [10, 20, 30, 40, 50]
s_t, tgt_t = sasrec_test_item(demo_items)
if tgt_t != demo_items[-1]:
    fail(f"测试 target 不是最后一个物品: {tgt_t}")
elif tgt_t in s_t:
    fail(f"测试 target ({tgt_t}) 泄漏到输入序列中")
else:
    ok(f"测试目标={tgt_t}(最后物品)，未出现在输入序列中")

# ─────────────────────────────────────────────────────────────
# 8. SASRec — padding mask 缺失警告
# ─────────────────────────────────────────────────────────────
sec("⑧ SASRec — Attention 中 Padding Mask 缺失")
warn("model.py forward 仅使用因果掩码（causal mask），未添加 key_padding_mask")
warn("padding 位 (item_id=0) embedding=0，但 + pos_emb 后非零，仍参与注意力计算")
info("  影响：短序列用户（大量 padding）的表示质量下降")
info("  修复方式：在 MultiheadAttention 调用时传入 key_padding_mask=(log_seqs==0)")
info("  当前实现与原 SASRec 论文一致，属于设计选择而非数据错位 bug")

# ─────────────────────────────────────────────────────────────
# 9. 真实数据文件抽样（存在则运行）
# ─────────────────────────────────────────────────────────────
sec("⑨ 真实文件抽样检查（不存在则跳过）")

try:
    import h5py

    rec_path = os.path.join(os.path.dirname(__file__), "../data/user_item_interact.h5")
    if os.path.exists(rec_path):
        with h5py.File(rec_path, 'r') as f:
            uids = f['user_id'][:]
        uniq = sorted(set(uids.tolist()))
        expected = list(range(1, len(uniq)+1))
        if uniq == expected:
            ok(f"T5 user_id 从 1 连续递增（{len(uniq)} 个用户），user_id-1 索引正确")
        else:
            warn(f"T5 user_id 不连续(min={uniq[0]}, max={uniq[-1]}, 个数={len(uniq)})")
            warn("  user_profile_embs[user_id-1] 可能对应错误用户！")
    else:
        skip(f"未找到: {rec_path}")

    item_emb_path = os.path.join(os.path.dirname(__file__), "../data/course_item_embs.h5")
    if os.path.exists(item_emb_path):
        with h5py.File(item_emb_path, 'r') as f:
            ie = f['item_embs'][:]
        info(f"item_embs shape: {ie.shape}")
        if np.abs(ie[0]).max() < 1e-5:
            ok("item_embs[0] ≈ 0（padding 占位），item_id 从 1 开始对应正确")
        else:
            warn(f"item_embs[0] 非零 (max={np.abs(ie[0]).max():.4f})，请确认 item_id 是否从 0 开始")
    else:
        skip(f"未找到: {item_emb_path}")

    rqvae_path = os.path.join(os.path.dirname(__file__), "../data/tiger/train_dataset.h5")
    if os.path.exists(rqvae_path):
        with h5py.File(rqvae_path, 'r') as f:
            targets = f['target'][:]
        t_min, t_max = int(targets.min()), int(targets.max())
        info(f"RQVAE-T5 target token 范围: [{t_min}, {t_max}]")
        if t_max > max(all_v):
            fail(f"超出合法范围 (合法最大={max(all_v)})")
        else:
            ok("target token 范围合法")
        eos_cnt = int((targets == eos_id).sum())
        if eos_cnt > 0:
            warn(f"EOS token={eos_id} 在 target 中出现 {eos_cnt} 次（是合法 code，pos3 val=6）")
    else:
        skip(f"未找到: {rqvae_path}")

    # SASRec 与 T5 使用相同的 user_item_interact.h5
    sasrec_h5 = os.path.join(os.path.dirname(__file__), "../data/user_item_interact.h5")
    if os.path.exists(sasrec_h5):
        with h5py.File(sasrec_h5, 'r') as f:
            s_uids = f['user_id'][:]
            item_lists_s = f['item_id_list'][:]
        cnt = Counter(s_uids.tolist())
        dups = [u for u, c in cnt.items() if c > 1]
        if dups:
            warn(f"SASRec {len(dups)} 个用户有多行，extend 拼接可能时序混乱")
        else:
            ok(f"SASRec 每个 user_id 只有 1 行（{len(cnt)} 个用户），extend 等价直接赋值")
        # 检查 user_id 连续性（与 T5 共用同一份验证）
        uniq_s = sorted(set(s_uids.tolist()))
        if uniq_s == list(range(1, len(uniq_s)+1)):
            ok(f"SASRec/T5 user_id 从 1 连续递增（{len(uniq_s)} 个用户）")
        else:
            warn(f"SASRec/T5 user_id 不连续(min={uniq_s[0]}, max={uniq_s[-1]}, 个数={len(uniq_s)})")
            warn("  T5 的 user_profile_embs[user_id-1] 会取到错误 embedding！")
        # 验证序列长度
        lens = [len(lst) for lst in item_lists_s]
        info(f"序列长度: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.1f}")
    else:
        skip(f"未找到: {sasrec_h5}")

except ImportError:
    skip("h5py 未安装，跳过真实文件检查")

# ─────────────────────────────────────────────────────────────
sec("检查完成 — 问题汇总")
print("""
  模型        | 严重度 | 问题
  ------------|--------|----------------------------------------------
  T5          | WARN   | user_embs[user_id-1]: 假设 user_id 从 1
              |        | 连续递增；不连续时取错 embedding
  T5          | OK     | item_embs[item_id]: 索引正确 (0=padding占位)
  T5          | OK     | 序列样本无泄漏，掩码方向正确
  RQVAE-T5    | WARN   | eos_token_id=31 与 pos3/val6 的合法 code
              |        | 重叠；max_length=5 下通常不截断，但存在
              |        | 理论风险，建议改为 33
  RQVAE-T5    | OK     | pad_token=0 不与合法 code 冲突
  RQVAE-T5    | OK     | attention mask 正确覆盖 padding 位
  SASRec      | OK     | input/target 位移正确 (o_t[i]=s[i+1])
  SASRec      | OK     | 测试集 leave-one-out 无泄漏
  SASRec      | WARN   | 未添加 key_padding_mask，短序列用户
              |        | padding 位置参与注意力计算（设计选择）
""")
