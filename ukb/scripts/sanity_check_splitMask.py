import numpy as np
mask = np.load("interim_data/split_mask.npy")
print("dtype:", mask.dtype)
print("shape:", mask.shape)
train = mask.sum()
test = (~mask).sum()
print(f"Train samples: {train}")
print(f"Test samples:  {test}")
print(f"Train/Test ratio: {train/(train+test):.2%} / {test/(train+test):.2%}")