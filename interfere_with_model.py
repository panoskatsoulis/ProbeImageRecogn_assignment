import torch
from tqdm.auto import tqdm
from Models.tinyvgg import model
from dataset import test_data_loader
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

import matplotlib.pyplot as plotter
import matplotlib.patches as patches

def print_image_with_inference():
    image, bbox = _img
    print(f"entry file: {_idx}")
    fig, item = plotter.subplots()
    box_x1, box_y1, width, height = bbox
    rect = patches.Rectangle((box_x1, box_y1), width, height,
                             linewidth=2, edgecolor='r', facecolor='none')
    item.imshow(image)
    item.add_patch(rect)
    fig.suptitle(f"idx: {_idx}, bbox: {bbox}")
    fig.savefig(f"{_dir}/image_{_idx}.png")
    plotter.close(fig)


for image, bbox in tqdm(test_data_loader):
    decision = bbox[0]
    _bbox = bbox[1:]
    
