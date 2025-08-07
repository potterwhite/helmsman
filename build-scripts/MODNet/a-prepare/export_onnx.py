"""
Export ONNX model of MODNet with:
    input shape: (batch_size, 3, height, width)
    output shape: (batch_size, 1, height, width)

Arguments:
    --ckpt-path: path of the checkpoint that will be converted
    --output-path: path for saving the ONNX model

Example:
    python export_onnx.py \
        --ckpt-path=modnet_photographic_portrait_matting.ckpt \
        --output-path=modnet_photographic_portrait_matting.onnx
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from . import modnet_onnx


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the ONNX model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    # define model & load checkpoint
    modnet = modnet_onnx.MODNet(backbone_pretrained=False)
    # ---------- modification start --------------------
    # modnet = nn.DataParallel(modnet).cuda()
    # state_dict = torch.load(args.ckpt_path)
    # modnet.load_state_dict(state_dict)

    state_dict = torch.load(args.ckpt_path, map_location=device)
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    modnet.load_state_dict(new_state_dict)

    modnet.to(device)
    # ---------- modification end --------------------

    modnet.eval()

    # prepare dummy_input
    batch_size = 1
    height = 512
    width = 512
    # ---------- modification start --------------------
    # dummy_input = Variable(torch.randn(batch_size, 3, height, width)).cuda()
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)
    # ---------- modification end --------------------


    # export to onnx model
    torch.onnx.export(
        # ---------- modification start --------------------
        # modnet.module, dummy_input, args.output_path, export_params = True,
        modnet, dummy_input, args.output_path, export_params = True,
        opset_version=11,
        # ---------- modification end --------------------
        input_names = ['input'], output_names = ['output'],
        dynamic_axes = {'input': {0:'batch_size', 2:'height', 3:'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
