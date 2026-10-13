from src.torchutils.registry import Registry

# 损失函数注册表 - 用于注册和管理所有损失函数实现
LOSS = Registry("loss")

# 数据集注册表 - 用于注册和管理所有数据加载器和数据集类
DATA = Registry("data")

# 模型注册表 - 用于注册和管理所有神经网络模型架构
MODEL = Registry("model")

# 优化器注册表 - 用于注册和管理所有优化算法（如 Adam、SGD 等）
OPTIMIZER = Registry("optimizer")

# 学习率调度器注册表 - 用于注册和管理所有学习率调度策略
SCHEDULER = Registry("scheduler")

# 奖励函数注册表 - 用于注册和管理强化学习的奖励函数
REWARD = Registry("reward")

