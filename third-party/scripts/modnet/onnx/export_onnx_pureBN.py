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

"""
Export ONNX model for Pure-BN MODNet (retrained with BatchNorm2d only, no IBNorm).

Use this script ONLY for checkpoints trained with the Pure-BN architecture
(third-party/scripts/modnet/src/models/modnet.py, IBNorm surgically replaced
with nn.BatchNorm2d). Do NOT use for the original MODNet pretrained checkpoint.

Arguments:
    --ckpt-path:   path of the Pure-BN checkpoint (e.g. checkpoints/modnet_bn_best.ckpt)
    --output-path: path for saving the ONNX model

Example:
    cd third-party/sdk/MODNet.git
    PYTHONPATH=. python -m onnx.export_onnx_pureBN \
        --ckpt-path checkpoints/modnet_bn_best.ckpt \
        --output-path checkpoints/modnet_bn_best.onnx
"""

import os
import sys
import argparse
from collections import OrderedDict

# ---------------------------------------------------------------------------
# sys.path surgery — must happen before any torch/onnx imports.
#
# Problem: This script lives in (or is symlinked into) MODNet.git/onnx/.
# When Python runs it, MODNet.git/ ends up on sys.path as '' (CWD).
# That causes `import onnx` to resolve to the local MODNet.git/onnx/
# *directory* instead of the pip-installed onnx package.  PyTorch's internal
# ONNX exporter does `import onnx` too, so it also hits the wrong package and
# crashes with: AttributeError: module 'onnx' has no attribute 'load_from_string'
#
# Fix: strip '' and the absolute CWD from sys.path before importing torch/onnx,
# then add MODNet.git/ back *after* those imports so src.models.modnet is found.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# sys.path surgery — must happen before any torch/onnx imports.
#
# Problem: This script lives in (or is symlinked into) MODNet.git/onnx/.
# When Python runs it, '' (CWD = MODNet.git/) is on sys.path.
# That causes `import onnx` to resolve to the local MODNet.git/onnx/
# *directory* instead of the pip-installed onnx package.  PyTorch's internal
# ONNX exporter does a lazy `import onnx` inside torch.onnx.export(), so
# even if we import torch first, that lazy call still hits the wrong package
# and crashes with:
#   AttributeError: module 'onnx' has no attribute 'load_from_string'
#
# Fix (3 steps):
#   1. Strip '' (CWD) from sys.path so local onnx/ won't shadow pip onnx.
#   2. `import onnx` immediately — caches pip onnx in sys.modules['onnx'].
#      Any later `import onnx` (including inside torch.onnx.export) will hit
#      the cache and return the correct pip package, regardless of sys.path.
#   3. Re-add MODNet.git/ so src.models.modnet is importable.
# ---------------------------------------------------------------------------
_modnet_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cwd_abs = os.path.abspath('.')

for _entry in list(sys.path):
    if _entry in ('', _cwd_abs, _modnet_root):
        sys.path.remove(_entry)

import onnx  # noqa: E402 — must be after '' removal; caches pip onnx in sys.modules

import torch          # noqa: E402
import torch.nn as nn # noqa: E402

# Restore MODNet.git/ so src.models.modnet is importable.
# onnx is already in sys.modules (pip version), so re-adding this path
# cannot shadow it anymore.
if _modnet_root not in sys.path:
    sys.path.insert(0, _modnet_root)

from src.models.modnet import MODNet as MODNetPureBN  # noqa: E402


class MODNetONNXWrapper(nn.Module):
    """Thin wrapper that fixes inference=True and returns only pred_matte.

    The Pure-BN MODNet.forward(img, inference) returns (pred_semantic,
    pred_detail, pred_matte).  For ONNX export we only need pred_matte and
    we want a single-input / single-output graph, matching the interface of
    export_onnx_modified.py.
    """
    def __init__(self, modnet):
        super(MODNetONNXWrapper, self).__init__()
        self.modnet = modnet

    def forward(self, img):
        _, _, pred_matte = self.modnet(img, inference=True)
        return pred_matte


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True,
                        help='path of the Pure-BN checkpoint to convert')
    parser.add_argument('--output-path', type=str, required=True,
                        help='path for saving the ONNX model')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find checkpoint path: {args.ckpt_path}')
        exit(1)

    # Load Pure-BN model (backbone_pretrained=False: we load weights from ckpt)
    modnet_base = MODNetPureBN(backbone_pretrained=False)

    state_dict = torch.load(args.ckpt_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Strip 'module.' prefix added by nn.DataParallel during training
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    modnet_base.load_state_dict(new_state_dict)
    modnet_base.to(device)
    modnet_base.eval()

    # Wrap for ONNX: single input, single output (pred_matte only)
    modnet = MODNetONNXWrapper(modnet_base)
    modnet.eval()

    batch_size = 1
    height = 512
    width = 512
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)

    torch.onnx.export(
        modnet,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
    )

    print(f'ONNX model saved to: {args.output_path}')
