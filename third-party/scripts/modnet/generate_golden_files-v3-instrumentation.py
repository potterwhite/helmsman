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
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image

import onnx
import onnxruntime

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path',
                        type=str,
                        help='path of the input image (a file)')
    parser.add_argument(
        '--output-path',
        type=str,
        help='paht for saving the predicted alpha matte (a file)')
    parser.add_argument('--model-path',
                        type=str,
                        help='path of the ONNX model')
    parser.add_argument('--debug-file-path',
                        type=str,
                        help='output path of the the debug reference files')
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory to store all intermediate and golden artifacts")

    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.image_path):
        print('Cannot find the input image: {0}'.format(args.image_path))
        exit()
    if not os.path.exists(args.model_path):
        print('Cannot find the ONXX model: {0}'.format(args.model_path))
        exit()
    if args.debug_file_path:
        debug_dir = os.path.dirname(args.debug_file_path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"{debug_dir} not exist so create it first.")
    # if not os.path.exists(args.debug_file_path):
    #     print('Cannot find the output debug files path: {0}'.format(args.debug_file_path))
    #     exit()
    if os.path.isdir(args.output_path):
        raise ValueError(
            f"--output-path must be a file path with extension, got directory: {args.output_path}"
        )

    ref_size = 512

    # Get x_scale_factor & y_scale_factor to resize image
    def get_scale_factor(im_h, im_w, ref_size):

        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h

        return x_scale_factor, y_scale_factor

    def echoIMG(input):
        flat = input.reshape(-1)

        print("first 10 values:", flat[:10])
        print("min:", flat.min(), "max:", flat.max())

        # byte-level
        # b = flat[:10].tobytes()
        # print("bytes:", list(b))
        raw = flat[:10].tobytes()  # 前 10 个 float = 40 个 byte
        print(" ".join(f"{b:02x}" for b in raw))

    ##############################################
    #  Main Inference part
    ##############################################

    # read image
    im = cv2.imread(args.image_path)
    # im.tofile(f"{args.debug_file_path}/debug_00_01_imread.bin")

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im.tofile(f"{args.debug_file_path}/debug_00_02_cvtColor.bin")

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]
    # im.tofile(f"{args.debug_file_path}/debug_00_03_rgb3.bin")

    # --------------------------------------------------
    # Processing -- 1st. normalize values to scale it between -1 to 1

    # # *** Method A ***
    # # *** error due to float64 ***
    # im = (im - 127.5) / 127.5
    # # ******

    # *** Method B ***
    # *** Worked perfect ***
    # im = im.astype(np.float32)
    im = im * np.float32(1.0 / 127.5) - np.float32(1.0)
    # im = np.ascontiguousarray(im, dtype=np.float32)
    # ******

    # Save to Local Disk
    # print(im.dtype)
    first_bin_name = "debug_01_normalized.bin"
    im.tofile(f"{args.debug_file_path}/{first_bin_name}")
    # print(im.dtype)
    # print(im.flags)
    # echoIMG(im)
    print(f"1st-Dumped {args.debug_file_path}/{first_bin_name} successfully.")

    # --------------------------------------------------
    # Processing -- 2nd. resize image
    im_h, im_w, im_c = im.shape
    print(f"Width={im_w}, Height={im_h}, Channel={im_c}")
    x, y = get_scale_factor(im_h, im_w, ref_size)
    print(f"x_scale_factor={x}, y_scale_factor={y}")

    # resize image
    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)
    print(
        f"Resized Width={im.shape[1]}, Resized Height={im.shape[0]}, Channel={im.shape[2]}"
    )
    second_bin_name = "debug_02_resized.bin"
    im.tofile(f"{args.debug_file_path}/{second_bin_name}")
    print(f"2nd-Dumped {args.debug_file_path}/{second_bin_name} successfully.")

    # --------------------------------------------------
    # Processing -- 3rd. transpose image shape from HWC to CHW
    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    third_bin_name = "debug_03_transposed.bin"
    im.tofile(f"{args.debug_file_path}/{third_bin_name}")
    print(f"3rd-Dumped {args.debug_file_path}/{third_bin_name} successfully.")

    # --------------------------------------------------
    # Processing -- 4th. golden file: add batch dimension & convert to float32
    im = np.expand_dims(im, axis=0).astype('float32')
    fourth_bin_name = "debug_04_golden_reference_tensor-Input.bin"
    im.tofile(f"{args.debug_file_path}/{fourth_bin_name}")
    print(f"4th-Dumped {args.debug_file_path}/{fourth_bin_name} successfully.")

    # ############################
    # IMPORTANT: INFERENCE STEP
    # Initialize session and get prediction
    session = onnxruntime.InferenceSession(args.model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})
    # ############################

    # ------------------------
    # 5th. Output tensor dump
    output_tensor = result[0]  # 形状通常是 [1, 1, H, W]

    print("Output shape:", output_tensor.shape)
    print("Output dtype:", output_tensor.dtype)

    # 确保连续内存（非常重要）
    output_tensor = np.ascontiguousarray(output_tensor, dtype=np.float32)

    # ⭐ Dump 原始推理输出（不要 squeeze）
    fifth_bin_name = "debug_05_inference-Output.bin"
    output_tensor.tofile(f"{args.debug_file_path}/{fifth_bin_name}")
    print(f"5th-Dumped {args.debug_file_path}/{fifth_bin_name} successfully.")

    # ------------------------
    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)

    if args.output_path:
        if os.path.isdir(args.output_path):
            raise ValueError("--output-path must be a file, not directory")

        cv2.imwrite(args.output_path, matte)
