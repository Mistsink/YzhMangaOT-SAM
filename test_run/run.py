import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import SamProcessor, SamModel
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from torch.optim import Adam
import monai
import cv2
from PIL import Image
from paddleocr import PaddleOCR
from ..utils.read import read_img
from ..utils.show import show_image
from ..utils.prompt import get_center_points_from_boxes

def run():
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)


    image = read_img('../images/04.jpg')

    # ocr to get bounding box
    #   then get center point prompts
    ocr = PaddleOCR()
    result = ocr.ocr(image)

    score_threshold = 0.7
    boxes = [line_info[0] for line_info in result for line_info in line_info if line_info[1][1] > score_threshold]

    point_prompts = get_center_points_from_boxes(boxes)


    # input_points: list[batch_size, n-prompts, 2]
    #   processor 输出的 input_points 会在后续进行 mask 预测
    #   processor 会进行图片尺寸的调整，所以不能直接用 point_prompts
    inputs = processor(images=image, input_points=[[point_prompts[0]]], return_tensors="pt")
    for k,v in inputs.items():
        print(k,v.shape)
    img_emb = model.get_image_embeddings(inputs['pixel_values'].to('cuda'))

    outs = model(input_points=inputs['input_points'].to('cuda'),
                image_embeddings=img_emb,
                multimask_output=True
                )
    masks = processor.post_process_masks(
            outs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )

    # show mask image
    from copy import deepcopy
    for mask in masks[0][0]:
        o_img = deepcopy(image)
        # 这里图片是原始图片，故而 points 使用原始的 point_prompts
        show_image(o_img, points=np.array([point_prompts[0]]), mask=mask.cpu())
