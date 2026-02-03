# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
