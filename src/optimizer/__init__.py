import torch.optim as optim

from src.registry import OPTIMIZER

OPTIMIZER.register(optim.SGD)
OPTIMIZER.register(optim.Adam)
OPTIMIZER.register(optim.AdamW)
OPTIMIZER.register(optim.LBFGS)

