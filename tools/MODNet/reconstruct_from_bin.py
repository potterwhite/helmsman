import numpy as np
import cv2

H, W = 512, 896   # ← 用推导出来的尺寸

data = np.fromfile(
    "build/golden/debug/golden_reference_tensor.bin",
    dtype=np.float32
).reshape(1, 3, H, W)

print("Min / Max:", data.min(), data.max())

img = data[0].transpose(1, 2, 0)
img = ((img + 1.0) * 127.5).astype('uint8')

cv2.imwrite("build/golden/debug/reconstructed_from_golden.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved build/golden/debug/reconstructed_from_golden.png")
