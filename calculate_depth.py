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

from depth_utils import download_model_if_doesnt_exist

def get_depth_matrix(image_path, image_id, model_name):

    encoder_path = os.path.join("depth_models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("depth_models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
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
    vmax = np.percentile(disp_resized_np, 100)
    disp_resized_np_real = 5.4 / ((9.99*disp_resized_np) + 0.01)
    # plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)

    if not os.path.exists("Output/depth_maps"):
        os.mkdir("Output/depth_maps")

    # plt.savefig(f"Output/Depth Maps/{image_id.split('.')[0]}.png", dpi=300, cmap="magma", bbox_inches = "tight")
    save_path = f"Output/depth_maps/{image_id}.png"
    plt.imsave(save_path, disp_resized_np, cmap="magma")
    plt.figure().clear(True)

    return disp_resized_np_real, save_path, depth_time

if __name__ == "__main__":
    image = "test.jpg"

    # model_name = "mono_640x192"
    # model_name = "mono+stereo_640x192"
    depth_model = "stereo_640x192"

    disp_resized_np_real = get_depth_matrix(image, depth_model)
    print(disp_resized_np_real)