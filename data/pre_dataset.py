from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import SamProcessor

class PreDataset(Dataset):
  '''
  TODO: 
  原始图片数据集
  据此处理后得到图片的 embedding
  '''
  def __init__(self, dataset, processor: SamProcessor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx: int) -> dict:
    '''
    @return: 
    '''
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    # prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    # inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
    inputs = self.processor(image, return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs