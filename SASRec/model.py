import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, item_num, params):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.dev = params['device']
        self.d = params['d']

        # 1. M ∈ R^{|I|×d}
        self.item_emb = nn.Embedding(item_num + 1, self.d, padding_idx=0)
        # 2. P ∈ R^{n×d}
        self.pos_emb = nn.Embedding(params['max_len'], self.d)

        # 3. Self-Attention 权重矩阵
        self.W_Q = nn.Linear(self.d, self.d)
        self.W_K = nn.Linear(self.d, self.d)
        self.W_V = nn.Linear(self.d, self.d)

        # Pre-Norm结构 + 多层堆叠
        self.attention_layernorms = nn.ModuleList(
            [nn.LayerNorm(self.d, eps=1e-8) for _ in range(params['num_blocks'])])
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(self.d, params['num_heads'], params['dropout'], batch_first=True) for _ in
             range(params['num_blocks'])])
        self.forward_layernorms = nn.ModuleList(
            [nn.LayerNorm(self.d, eps=1e-8) for _ in range(params['num_blocks'])])

        # 4. Feed Forward Network (F_i = ReLU(S_iW1 + b1)W2 + b2)
        self.forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d, 4 * self.d),
                nn.ReLU(),
                nn.Dropout(params['dropout']),
                nn.Linear(4 * self.d, self.d),
                nn.Dropout(params['dropout'])
            ) for _ in range(params['num_blocks'])
        ])

        self.last_layernorm = nn.LayerNorm(self.d, eps=1e-8)

    def forward(self, log_seqs):
        """
        前向流程：
        1. 输入嵌入：Ẽ_i = M_{s_i} + P_i
        2. 因果Mask的Self-Attention：S_i = Σ_{j≤i} α_{i,j}V_j
        3. Feed Forward：F_i = ReLU(S_iW1 + b1)W2 + b2
        4. 残差+LayerNorm：Output = x + Dropout(LayerNorm(f(x)))
        """
        # 1. Ẽ_i = M_{s_i} + P_i
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(log_seqs.shape[1], device=self.dev).unsqueeze(0).expand_as(log_seqs)  # 位置索引
        seqs += self.pos_emb(positions)

        # 2. 生成Q/K/V
        Q = self.W_Q(seqs)
        K = self.W_K(seqs)
        V = self.W_V(seqs)

        # 3. Causality Mask (j>i时α_{i,j}=0)（因果掩码，不需要滑动构建训练样本）
        tl = log_seqs.shape[1]
        attention_mask = torch.triu(torch.ones((tl, tl), dtype=torch.bool, device=self.dev), diagonal=1)

        # 4. 多层Self-Attention + FFN堆叠
        for i in range(len(self.attention_layers)):
            '''
            Self-Attention Block  g(x) = x + Dropout(g(LayerNorm(x)))
            1. LayerNorm(x) x = seqs = F^(b-1), LayerNorm(x)
            2. g(LayerNorm(x)) = SA(LayerNorm(x)) (S^(b) = SA(F^(b-1)))
            3. x + Dropout(g(LayerNorm(x)))
                F^(b-1) + SA(LayerNorm(F^(b-1))) = S^(b)
            '''
            mha_input = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                mha_input, mha_input, mha_input,
                attn_mask=attention_mask)
            seqs = seqs + mha_outputs

            '''
            FFN Block
            1: LayerNorm(S^(b))
            2: g(LayerNorm(x)) = FFN(LayerNorm(x)) F_i^(b) = FFN(S_i^(b))
            3: x + Dropout(g(LayerNorm(x)))   # S^(b) + FFN(LayerNorm(S^(b))) = F^(b)
            '''
            ffn_input = self.forward_layernorms[i](seqs)
            ffn_outputs = self.forward_layers[i](ffn_input)
            seqs = seqs + ffn_outputs

        return self.last_layernorm(seqs)

    def predict(self, log_seqs):
        """
        1. 取最后一步状态h_t = F_t^{(b)}
        2. 计算所有物品得分：r_{i,t} = h_t · M_i
        """
        final_feats = self.forward(log_seqs)  # [B, L, d]
        final_feat = final_feats[:, -1, :]  # h_t: [B, d] (最后一步状态)

        # r_{i,t} = h_t · M_i
        logits = final_feat.matmul(self.item_emb.weight.t())
        return logits