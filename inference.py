import torch, os
from tqdm.auto import tqdm

## loading model and setting dict
from Models.tinyvgg import model
MODEL_SAVE = 'Models/tinyvgg_state_dict.pkl'
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(f=MODEL_SAVE,
                                 map_location=torch.device(device)))
model.to(device)

## fetching data and splitting correctly
from dataset import ProbesDataset, images_fullinfo, flyability_test, idx_test
from sklearn.model_selection import train_test_split
flyability_images = ProbesDataset(
    _dir = "probe_dataset/probe_images",
    labels = images_fullinfo
    #transform = transforms.ToTensor()
)
_, data_test, _, test_images_idx = train_test_split(flyability_images,
                                            range(len(flyability_images)),
                                            test_size=0.2, random_state=10)
print(f"test indices {test_images_idx}")
test_data_tensors = torch.utils.data.DataLoader(flyability_test,
                                                batch_size=1,
                                                shuffle=False)
print(f"tensor indices {idx_test}")

import matplotlib.pyplot as plotter
import matplotlib.patches as patches
def print_image_with_inference(_dir, _idx, _img, inference_bbox, prob):
    image, bbox = _img
    fig, item = plotter.subplots()
    box1_x, box1_y, width1, height1 = bbox[1:]
    rect_orig = patches.Rectangle((box1_x, box1_y), width1, height1,
                                  linewidth=2, edgecolor='r', facecolor='none')
    box2_x, box2_y, width2, height2 = inference_bbox[1:]
    rect_infr = patches.Rectangle((box2_x, box2_y), width2, height2,
                                  linewidth=2, edgecolor='y', facecolor='none')
    item.imshow(image)
    item.add_patch(rect_orig)
    item.add_patch(rect_infr)
    fig.suptitle(f"idx: {_idx}\noriginal bbox: {bbox[1:]}\ninference prob {prob:.1f}% - bbox {inference_bbox[1:]}")
    fig.savefig(f"{_dir}/image_{_idx}.png")
    plotter.close(fig)


## prep metrics
from torchvision.ops.boxes import box_iou
def make_xy1wh_xy1xy2(tensor):
    x1, y1, w, h = tensor[1:] if len(tensor)==5 else tensor
    return torch.Tensor([x1, y1, x1+w, y1+h])

target_dir = "visualize_data/inference"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
with torch.inference_mode():
    inference_dec_set, inference_bbox_set = [], []
    target_dec_set, target_bbox_set = [], []
    for idx, ((tensor_in, tensor_out), image_bbox) in tqdm(enumerate(zip(test_data_tensors,data_test))):
        image, bbox = image_bbox
        decision = bbox[0]
        _bbox = bbox[1:]
        tensor_in = tensor_in.to(device)
        inference_bbox = model(tensor_in) # unfold batch dimension
        #print(f"test {idx}: original bbox {bbox}, inference_bbox {inference_bbox}")
        #print(f"test: inference_bbox {inference_bbox}, original_bbox {tensor_out}")
        if device != 'cpu':
            tensor_out.cpu()
            inference_bbox.cpu()
        target_bbox_set.append(tensor_out.squeeze(0).numpy()[1:])
        inference_bbox_set.append(inference_bbox.squeeze(0).numpy()[1:])
        target_dec_set.append(tensor_out.squeeze(0).numpy()[0])
        dec_prob = torch.sigmoid(inference_bbox.squeeze(0)[0])
        #print(f"logit: {dec_prob*100:.2f}%")
        inference_dec_set.append(torch.sigmoid(inference_bbox.squeeze(0)[0])) # logit -> prob
        print_image_with_inference(target_dir, idx, image_bbox, inference_bbox.squeeze(0).numpy(), dec_prob*100)
        print(f"IoU: {box_iou(make_xy1wh_xy1xy2(tensor_out),make_xy1wh_xy1xy2(inference_bbox))}")
    #print(f'Accuracy: {acc(inference_dec_set,target_dec_set):.2f}')
