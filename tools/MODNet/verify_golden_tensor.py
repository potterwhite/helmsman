import numpy as np

# 按你的模型真实尺寸改这里（举例）
H, W = 512, 512

data = np.fromfile(
    "build/golden/debug/golden_reference_tensor.bin",
    dtype=np.float32
)

print("Total elements:", data.size)
print("Expected:", 1 * 3 * H * W)
print("Min / Max:", data.min(), data.max())
