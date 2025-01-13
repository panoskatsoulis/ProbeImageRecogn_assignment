import torch
from tqdm.auto import tqdm

## loading model and setting dict
from Models.tinyvgg import model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

## fetching data and splitting correctly
from dataset import flyability_test, images_fullinfo
from sklearn.model_selection import train_test_split
flyability_images = ProbesDataset(
    _dir = "probe_dataset/probe_images",
    labels = images_fullinfo
    #transform = transforms.ToTensor()
)
_, data_test = train_test_split(flyability, test_size=0.2, random_state=10)
test_data_tensors = torch.utils.data.DataLoader(flyability_test,
                                                batch_size=1,
                                                shuffle=False)

import matplotlib.pyplot as plotter
import matplotlib.patches as patches
def print_image_with_inference(_idx, _img, inference_bbox):
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


for image, bbox in tqdm(test_data_loader)):
    decision = bbox[0]
    _bbox = bbox[1:]
    
