from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
from PIL import ImageOps

import torch
from torchvision import transforms

from architectures.resnet_encoder import *
from architectures.depth_decoder import *

from utils.torch_utils import time_sync

def get_depth_matrix(image_path, image_id, model_name):
    '''
    This function generates the depth that is converted from floating point matrix to the absolute value matrix for each pixel.
    args:
    image_path - path of the image that is currently detected by the ODM module
    image_id - ID of the image required to save the output image
    model_name - The depth model name (in our case stereo_model) to generate the depth map

    returns:
    disp_resized_np_real - Depth matrix with absolute distance
    save_path - 
    depth_time
    
    '''

    # Specifying the encoder and decoder path
    encoder_path = os.path.join("depth_models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("depth_models", model_name, "depth.pth")

    # Loading the encoder and decoder network
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()

    input_image = pil.open(image_path).convert('RGB')
    input_image = ImageOps.exif_transpose(input_image)
    original_width, original_height = input_image.size

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    depth_time_1 = time_sync()
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)
    depth_time_2 = time_sync()
    depth_time = depth_time_2 - depth_time_1

    disp = outputs[("disp", 0)]

    disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)

    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    disp_resized_np_real = 5.4 / ((9.99*disp_resized_np) + 0.01)

    save_path = f"Output/depth_maps/{image_id}.png"
    plt.imsave(save_path, disp_resized_np, cmap="magma")
    plt.figure().clear(True)

    return disp_resized_np_real, save_path, depth_time


def get_depth_value(depth_matrix, h, w, depth_map, c1):

    # Center coordinate with dimension (0.53h − 0.48h) × (0.53w − 0.48w) in the selected window below
    coords_yyxx = [(c1[1] + int(0.48 * h), c1[1] + int(0.53 * h), c1[0] +int(0.48 * w), c1[0] + int(0.53 * w))]

    # Calculate the mean of depth values within the selected window
    abs_center_depth = [depth_matrix[y1: y2, x1: x2].mean() for y1, y2, x1, x2 in coords_yyxx]

    # Drawing the green center box in the depth map where we extracted the mean depth value
    for i, coords in enumerate(coords_yyxx):
        cv2.rectangle(depth_map, (coords[2], coords[0]), (coords[3], coords[1]), color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)

    # return the mean depth value and the associated depth_map
    return abs_center_depth[0], depth_map