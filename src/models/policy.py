import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torchutils.common import get_device
from src.registry import MODEL


@MODEL.register
class AntennaRowPolicyModel(nn.Module):
    """
    天线拓扑生成的 RL Policy Model（支持 batch，多矩阵版）

    设计：
    - 一次前向生成 n_matrices 个 n_rows x row_bits 的二值矩阵
    - 逐行自回归：共 n_matrices * n_rows 步，每一步动作空间大小为 2^row_bits
      action ∈ [0, 2^row_bits - 1]，转成 row_bits 位二进制即为该行
    - 步与矩阵的对应关系：第 k 步属于第 (k // n_rows) 个矩阵的第 (k % n_rows) 行

    默认对应本项目的三层结构：
    - n_matrices=3（TM2 / TM3 / V1）、n_rows=19、row_bits=15
    - 即 57 步，每步词表 2^15 = 32768

    forward 返回：
    - actions:    torch.LongTensor, shape=(B, n_steps)
    - topo_bits:  torch.LongTensor, shape=(B, n_matrices, n_rows, row_bits)
    - logp_sum:   torch.Tensor,     shape=(B,)  # 每个样本序列 log-prob 之和
    - entropy_sum:torch.Tensor,     shape=(B,)  # 每个样本序列 entropy 之和
    """

    def __init__(
        self,
        n_matrices: int = 3,
        n_rows: int = 19,
        row_bits: int = 15,
        hidden_size: int = 64,
        temperature=None,
        tanh_constant: float = 1.5,
        # ---- 稀疏通道（V1）密度约束 ----
        # 训练数据里 V1 通道极度稀疏：285 个像素中白点数 mean=6.84、max=20，
        # 单行白点 max=6，71.5% 的行全空；而 TM2/TM3 的密度约 0.45 / 0.51。
        # 若不加约束，策略会把 V1 刷白到训练分布之外（实测 top-k 设计密度达
        # 0.70~0.81，z-score≈+37），此时代理模型纯属外推，会报出虚假的优异
        # S 参数（reward hacking），而真实仿真中该结构并不存在——训练集里
        # 1122 个好设计（S11<-15dB 且 S21>-3dB）的 V1 白点数 max 也只有 20。
        #
        #   v1_max_total:   整个 V1 矩阵的白点总数上限（训练集 max=20）
        #   v1_max_per_row: 单行白点上限（训练集 max=6）
        #
        # 两者均为 None 时退化为原行为（不约束）。
        v1_matrix_index: int = 2,
        v1_max_total: int | None = None,
        v1_max_per_row: int | None = None,
    ):
        super().__init__()
        if n_matrices <= 0 or n_rows <= 0 or row_bits <= 0:
            raise ValueError(
                f"n_matrices / n_rows / row_bits 必须为正，当前为 "
                f"{n_matrices} / {n_rows} / {row_bits}")

        self.n_matrices = n_matrices
        self.n_rows = n_rows
        self.row_bits = row_bits
        self.n_steps = n_matrices * n_rows
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.device = get_device()

        self.vocab_size = 2 ** row_bits

        # ---- 稀疏通道约束参数 ----
        self.v1_matrix_index = int(v1_matrix_index)
        self.v1_max_total = None if v1_max_total is None else int(v1_max_total)
        self.v1_max_per_row = None if v1_max_per_row is None else int(v1_max_per_row)
        if not (0 <= self.v1_matrix_index < n_matrices):
            raise ValueError(
                f"v1_matrix_index 必须落在 [0, {n_matrices}) 内，当前为 "
                f"{self.v1_matrix_index}")
        if self.v1_max_per_row is not None and not (0 <= self.v1_max_per_row <= row_bits):
            raise ValueError(
                f"v1_max_per_row 必须落在 [0, {row_bits}] 内，当前为 "
                f"{self.v1_max_per_row}")
        if self.v1_max_total is not None and self.v1_max_total < 0:
            raise ValueError(f"v1_max_total 不能为负，当前为 {self.v1_max_total}")
        self._v1_constrained = (
            self.v1_max_total is not None or self.v1_max_per_row is not None)

        self.action_embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.start_token = nn.Parameter(torch.zeros(hidden_size))

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.logits_head = nn.Linear(hidden_size, self.vocab_size)

        # 每个动作（15 bit 行编码）对应的白点数，用于约束 V1 的密度。
        # 注册为非持久化 buffer：不会进入 state_dict，因此与旧 checkpoint
        # 以及并行保存/加载逻辑保持兼容。
        self.register_buffer(
            "_row_popcount",
            torch.tensor([bin(i).count("1") for i in range(self.vocab_size)],
                         dtype=torch.long),
            persistent=False)

        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for p in self.parameters():
            if p.requires_grad:
                p.data.uniform_(-init_range, init_range)

    # 注意：cache 对 batch_size 生效；device 如果可能变化，建议不要跨 device 复用
    @functools.lru_cache(maxsize=64)
    def _zeros(self, batch_size: int):
        return torch.zeros((batch_size, self.hidden_size), device=self.device)

    def _scale_logits(self, logits: torch.Tensor):
        if self.temperature is not None:
            logits = logits / self.temperature
        if self.tanh_constant is not None:
            logits = self.tanh_constant * torch.tanh(logits)
        return logits

    def _sample_action(self, probs: torch.Tensor):
        """
        probs: (B, vocab_size)
        return:
          action: (B,)
          logp: (B,)
          entropy: (B,)
        """
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample()
        logp = m.log_prob(action)
        entropy = m.entropy()
        return action, logp, entropy

    @staticmethod
    def _int_to_bits(action_int: torch.Tensor, num_bits: int) -> torch.Tensor:
        """
        action_int: (B, n_steps) 或 (B,) 的 int64
        return:     (..., num_bits) 的 0/1 bits
        """
        shifts = torch.arange(num_bits - 1, -1, -1, device=action_int.device, dtype=torch.long)
        # 通过 broadcast，把最后一维变成 bits
        bits = (action_int.unsqueeze(-1) >> shifts) & 1
        return bits

    def forward(self, batch_size: int = 1, force_uniform: bool = False):
        """
        batch_size: 每次采样多少个 architecture / topology
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正数，当前为 {batch_size}")

        # buffers
        actions_buf = []      # 每步 action: (B,)
        logp_buf = []         # 每步 logp: (B,)
        entropy_buf = []      # 每步 entropy: (B,)

        # 初始隐状态
        hx = self._zeros(batch_size)
        cx = self._zeros(batch_size)

        # 第一步输入：start token，扩展到 batch
        embed = self.start_token.unsqueeze(0).expand(batch_size, -1)  # (B, H)

        # V1 通道已占用的白点预算（仅在启用约束时使用）：(B,)
        used_white = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for step in range(self.n_steps):
            hx, cx = self.lstm(embed, (hx, cx))

            if force_uniform:
                logits = torch.zeros((batch_size, self.vocab_size), device=self.device)
            else:
                logits = self.logits_head(hx)
                logits = self._scale_logits(logits)

            # 稀疏通道约束：把会超出密度上限的动作概率置 0（logits = -inf）。
            # popcount=0 的全空行恒满足两个上限，故合法集合非空，
            # softmax 不会出现全 -inf 导致的 NaN。
            if self._v1_constrained and (step // self.n_rows) == self.v1_matrix_index:
                popcount = self._row_popcount.unsqueeze(0)           # (1, V)
                legal = torch.ones_like(popcount, dtype=torch.bool)  # (1, V)
                if self.v1_max_per_row is not None:
                    legal = legal & (popcount <= self.v1_max_per_row)
                if self.v1_max_total is not None:
                    legal = legal & (popcount + used_white.unsqueeze(1)
                                     <= self.v1_max_total)
                logits = logits.masked_fill(~legal, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            action, logp, entropy = self._sample_action(probs)

            if self._v1_constrained and (step // self.n_rows) == self.v1_matrix_index:
                used_white = used_white + self._row_popcount[action]

            actions_buf.append(action)      # (B,)
            logp_buf.append(logp)           # (B,)
            entropy_buf.append(entropy)     # (B,)

            # 下一步输入：action embedding
            embed = self.action_embedding(action)  # (B, H)

        # (B, n_steps)
        actions = torch.stack(actions_buf, dim=1).to(torch.long)

        # (B, n_steps, row_bits) -> (B, n_matrices, n_rows, row_bits)
        topo_bits = self._int_to_bits(actions, self.row_bits).to(torch.long)
        topo_bits = topo_bits.reshape(
            batch_size, self.n_matrices, self.n_rows, self.row_bits)

        # 每个样本的序列 logp/entropy 之和：(B,)
        logp_sum = torch.stack(logp_buf, dim=1).sum(dim=1)
        entropy_sum = torch.stack(entropy_buf, dim=1).sum(dim=1)

        return actions, topo_bits, logp_sum, entropy_sum


if __name__ == "__main__":
    policy_model = AntennaRowPolicyModel(
        n_matrices=3, n_rows=19, row_bits=15, hidden_size=64)

    actions, topo_bits, logp, entropy = policy_model(batch_size=3)

    print("actions shape:", actions.shape)         # (3, 57)
    print("topo_bits shape:", topo_bits.shape)     # (3, 3, 19, 15)
    print("logp shape:", logp.shape)               # (3,)
    print("entropy shape:", entropy.shape)         # (3,)
