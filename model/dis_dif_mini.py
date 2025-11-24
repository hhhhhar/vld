import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class DiscreteDiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,           # 动作维度 (例如 7: 6关节 + 1夹爪)
        num_action_bins: int,      # 每个动作维度的离散份数 (例如 256)
        state_dim: int,            # 你的自定义特征提取器输出的维度
        hidden_dim: int = 512,     # Transformer 内部维度
        num_layers: int = 6,       # Transformer 层数
        nhead: int = 8,            # 注意力头数
        max_seq_len: int = 100,    # 最大序列长度
        dropout: float = 0.1
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_bins = num_action_bins
        
        # MASK token 的 ID 是 num_action_bins (例如 0-255 是动作, 256 是 MASK)
        self.mask_token_id = num_action_bins 
        self.vocab_size = num_action_bins + 1 # +1 for MASK

        # 1. State Projector (将你的自定义特征映射到 Transformer 维度)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # 2. Action Embeddings (包括 MASK token)
        self.action_embed = nn.Embedding(self.vocab_size, hidden_dim)

        # 3. Positional Encoding (用于区分动作序列的前后顺序)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        # 4. Transformer Encoder (双向注意力，因为是一次性预测所有 token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Prediction Head (输出每个位置的 Token 概率)
        self.predict_head = nn.Linear(hidden_dim, num_action_bins) # 不预测 MASK，只预测动作

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_mask_schedule(self, t, total_tokens, method="cosine"):
        """计算在这个时间步 t 需要 mask 多少个 token"""
        if method == "cosine":
            return torch.cos(t * math.pi * 0.5)
        else:
            # Linear
            return 1 - t

    def forward(self, state_features, action_tokens):
        """
        训练时的前向传播
        Args:
            state_features: (Batch, state_dim) 来自你的视觉/文本提取器
            action_tokens: (Batch, action_dim) 真实的离散动作索引 (0 ~ num_bins-1)
        Returns:
            loss: 标量 CrossEntropyLoss
        """
        B, L = action_tokens.shape
        device = action_tokens.device

        # --- 1. 随机 Masking 过程 (Forward Diffusion) ---
        # 随机采样一个时间比率 t ~ [0, 1]
        r = torch.rand(B, device=device)
        
        # 计算每个样本需要 mask 多少个 token
        # 使用简单的余弦调度: t=0 (mask多) -> t=1 (mask少) *注意这里的方向定义可能与论文略有不同，逻辑一致即可*
        # 这里我们就用简单的: 随机 mask 掉 k 个
        num_to_mask = (r * L).long().clamp(min=1, max=L) # 至少 mask 1 个

        # 创建 Mask
        # 为每个 token 生成一个随机分数，取分数最小的前 k 个进行 mask
        rand_scores = torch.rand(B, L, device=device)
        # argsort 两次得到 rank
        mask_indices = rand_scores.argsort(dim=1).argsort(dim=1) < num_to_mask.unsqueeze(1)
        
        # 构造输入序列：被选中的位置换成 MASK_ID，没选中的保持原样
        masked_input_ids = action_tokens.clone()
        masked_input_ids[mask_indices] = self.mask_token_id

        # --- 2. 模型编码 ---
        # 嵌入动作
        action_embeds = self.action_embed(masked_input_ids) # (B, L, H)
        # 添加位置编码
        action_embeds = action_embeds + self.pos_embed[:, :L, :]

        # 嵌入状态 (State)
        state_embeds = self.state_proj(state_features).unsqueeze(1) # (B, 1, H)

        # 拼接: [State, Action_Sequence]
        # 注意：这里我们把 State 放在最前面作为一个 Context Token
        transformer_input = torch.cat([state_embeds, action_embeds], dim=1) # (B, L+1, H)

        # 运行 Transformer
        output = self.transformer(transformer_input)

        # 取出对应 Action 部分的输出 (去掉第一个 State token)
        action_output = output[:, 1:, :] # (B, L, H)

        # --- 3. 计算 Loss ---
        logits = self.predict_head(action_output) # (B, L, num_bins)

        # 只计算被 Mask 掉的位置的 Loss
        loss = F.cross_entropy(logits.permute(0, 2, 1), action_tokens, reduction='none')
        
        # 应用 mask: 只优化被 mask 的部分
        masked_loss = (loss * mask_indices.float()).sum() / (mask_indices.float().sum() + 1e-6)

        return masked_loss

    @torch.no_grad()
    def sample(self, state_features, num_steps=10):
        """
        推理时的采样 (Reverse Diffusion / Iterative Decoding)
        Args:
            state_features: (Batch, state_dim)
            num_steps: 迭代去噪步数 (例如 10-12 步)
        Returns:
            pred_action_tokens: (Batch, action_dim)
        """
        B = state_features.shape[0]
        L = self.action_dim
        device = state_features.device

        # 1. 初始化：全都是 [MASK]
        current_tokens = torch.full((B, L), self.mask_token_id, dtype=torch.long, device=device)
        
        # 预计算 State Embedding (只需计算一次)
        state_embeds = self.state_proj(state_features).unsqueeze(1) # (B, 1, H)

        for step in range(num_steps):
            # --- A. 预测当前所有 Token 的概率 ---
            action_embeds = self.action_embed(current_tokens) + self.pos_embed[:, :L, :]
            transformer_input = torch.cat([state_embeds, action_embeds], dim=1)
            
            output = self.transformer(transformer_input)
            action_output = output[:, 1:, :] # (B, L, H)
            logits = self.predict_head(action_output) # (B, L, num_bins)
            
            probs = F.softmax(logits, dim=-1) # (B, L, num_bins)
            
            # 直接取最大概率的 Token 作为当前预测
            # (这里也可以用 Gumbel Softmax 或 Top-K 采样增加多样性)
            pred_scores, pred_ids = probs.max(dim=-1) 

            # --- B. Re-masking 策略 (核心) ---
            # 计算这一步应该保留多少个 token (从少到多，cosine schedule)
            ratio = math.cos((step + 1) / num_steps * math.pi * 0.5)
            # num_to_mask 是这一步结束后还需要保持 Mask 状态的数量
            num_to_mask = int(L * ratio)

            if num_to_mask == 0:
                current_tokens = pred_ids
                break

            # 根据置信度 (pred_scores) 决定哪些是“确信”的，哪些需要重新 Mask
            # 我们希望 Mask 掉分数最低的那些
            # 为了 Mask 掉分数低的，我们将分数取反排序，或者直接 argsort
            
            # 对置信度加入一点点随机噪声，防止陷入死循环
            noise = torch.rand_like(pred_scores) * 1e-6
            noisy_scores = pred_scores + noise
            
            # 找到置信度最低的 num_to_mask 个位置
            # topk(largest=False) 返回最小的
            _, mask_indices = torch.topk(noisy_scores, k=num_to_mask, dim=1, largest=False)
            
            # 更新 token 序列：
            # 1. 先全部设为刚才预测的结果
            current_tokens = pred_ids
            # 2. 把置信度低的重新盖上 [MASK]
            # create batch indices
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_to_mask)
            current_tokens[batch_indices, mask_indices] = self.mask_token_id

        return current_tokens
    
def test_discrete_diffusion_policy():
    print("\n========== 开始测试 Discrete Diffusion Policy ==========")
    
    # --- 1. 配置测试参数 (使用较小的参数以便快速运行) ---
    batch_size = 4
    action_dim = 7       # 假设是 7 自由度机械臂
    num_bins = 100       # 动作离散为 100 份
    state_dim = 32       # 假设视觉特征只有 32 维
    hidden_dim = 64      # Transformer 内部维度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    # --- 2. 实例化模型 ---
    try:
        policy = DiscreteDiffusionPolicy(
            action_dim=action_dim,
            num_action_bins=num_bins,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=2,    # 测试用，层数少一点
            nhead=4
        ).to(device)
        print(f"[Pass] 模型实例化成功")
        print(f"       参数量: {sum(p.numel() for p in policy.parameters()):,}")
    except Exception as e:
        print(f"[Fail] 模型实例化失败: {e}")
        return

    # --- 3. 构造伪造数据 ---
    # 模拟特征提取器输出的 State Vector
    fake_state = torch.randn(batch_size, state_dim).to(device)
    
    # 模拟真实的 Ground Truth 动作 (离散索引 0 ~ 99)
    fake_gt_actions = torch.randint(0, num_bins, (batch_size, action_dim)).to(device)

    # --- 4. 测试前向传播 (Forward / Training Loss) ---
    print("\n[Step 1] 测试训练过程 (Forward Loss)...")
    try:
        loss = policy(fake_state, fake_gt_actions)
        print(f"       Loss值: {loss.item():.4f}")
        
        if torch.isnan(loss):
            print("[Fail] Loss 是 NaN！")
        else:
            print("[Pass] Loss 计算成功且数值有效")
            
        # 简单测试一下反向传播是否报错
        loss.backward()
        print("[Pass] Backward 反向传播成功")
        
    except Exception as e:
        print(f"[Fail] Forward 过程报错: {e}")

    # --- 5. 测试采样过程 (Inference / Sampling) ---
    print("\n[Step 2] 测试推理过程 (Sampling)...")
    try:
        policy.eval() # 切换到评估模式
        with torch.no_grad():
            # 进行 10 步去噪
            pred_tokens = policy.sample(fake_state, num_steps=10)
            
        print(f"       输出形状: {pred_tokens.shape}")
        
        # 验证 1: 形状必须是 (Batch, Action_Dim)
        if pred_tokens.shape == (batch_size, action_dim):
            print("[Pass] 输出形状正确")
        else:
            print(f"[Fail] 输出形状错误，期望 {(batch_size, action_dim)}")

        # 验证 2: 检查是否有残留的 MASK token
        mask_id = num_bins
        if (pred_tokens == mask_id).any():
            print("[Warning] 采样结果中包含未填充的 MASK token (可能是步数太少或模型未收敛)")
        else:
            print("[Pass] 所有 Mask 都已成功填充")

        # 验证 3: 检查数值范围
        if (pred_tokens >= 0).all() and (pred_tokens < num_bins).all():
            print("[Pass] 输出动作索引在合法范围内")
        else:
            print("[Fail] 输出动作索引越界")
            
        print(f"       采样动作示例 (第一条数据): {pred_tokens[0].cpu().tolist()}")

    except Exception as e:
        print(f"[Fail] Sampling 过程报错: {e}")
        import traceback
        traceback.print_exc()

    print("\n========== 测试结束 ==========")

if __name__ == "__main__":
    # 假设你已经把上一段的类定义复制到了这里，或者已经import了
    # 这里直接运行测试
    test_discrete_diffusion_policy()