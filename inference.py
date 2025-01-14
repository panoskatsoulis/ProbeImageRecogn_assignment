import torch, os, argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Runs inference on a given directory with images (or by default to the testing sample) and returns results in "visualize_data/inference/"')
parser.add_argument('--input_dir', type=str, default='', help='Directory with images to run inference. (default: runs over the test sample)')
parser.add_argument('--iou_cut', type=float, default=0.35, help='Cut to define accuracy, IoU>cut: probe found, IoU<cut: probe not found.')
args = parser.parse_args()

## loading model and setting dict
from Models.tinyvgg import model
MODEL_SAVE = 'Models/tinyvgg_state_dict.pkl'
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(f=MODEL_SAVE, weights_only=True,
                                 map_location=torch.device(device)))
model.to(device)

## fetching data and splitting correctly
from dataset import ProbesDataset
targetToLoadTensors = None
fullValidationMode = False
if args.input_dir:
    print(f"Runninng inference over given images in os.getcwd()/{args.input_dir}... (Full Validation Mode)")
    fullValidationMode = True
    from torchvision import transforms
    labels = list({'idx' : i, 'file_name' : _f, 'bbox' : [0,0,0,0]}
                  for i,_f in enumerate(os.listdir(args.input_dir)))
    input_images = ProbesDataset(_dir = args.input_dir, labels = labels,
                                 transform = transforms.ToTensor())
    data_test = ProbesDataset(_dir = args.input_dir, labels = labels,
                              transform = None)
    targetToLoadTensors = input_images
else:
    print(f"Runninng inference over testing sample from dataset.py...")
    from dataset import images_fullinfo, flyability_test, idx_test
    from sklearn.model_selection import train_test_split
    flyability_images = ProbesDataset(
        _dir = "probe_dataset/probe_images",
        labels = images_fullinfo
        #transform = transforms.ToTensor()
    )
    _, data_test, _, test_images_idx = train_test_split(flyability_images,
                                                        range(len(flyability_images)),
                                                        test_size=0.2, random_state=10)
    #print(f"test indices {test_images_idx}") ## these compare the indices to validate that
    #print(f"tensor indices {idx_test}")      ## testing samples are always the same betewwn machines
    targetToLoadTensors = flyability_test

test_data_tensors = torch.utils.data.DataLoader(targetToLoadTensors,
                                                batch_size=1,
                                                shuffle=False)

import matplotlib.pyplot as plotter
import matplotlib.patches as patches
import numpy
def print_image_with_inference(_dir, _idx, _img, inference_bbox, iou):
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
    title = f"idx: {_idx}\ninference bbox {numpy.round(inference_bbox[1:],decimals=1)}"
    if not fullValidationMode:
        title = f"idx: {_idx}\noriginal bbox: {bbox[1:]}\ninference IoU {iou:.2f} - bbox {numpy.round(inference_bbox[1:],decimals=1)}"
    if iou<args.iou_cut and not fullValidationMode:
        item.text(2, 40 if box1_y>50 else 390,
                  'probe couldn\'t be recognized',
                  fontsize=24, color='orange')
    fig.suptitle(title)
    fig.savefig(f"{_dir}/image_{_idx}.png")
    plotter.close(fig)


## prep metrics
from torchvision.ops.boxes import box_iou
def make_xy1wh_xy1xy2(tensor):
    x1, y1, w, h = tensor[1], tensor[2], tensor[3], tensor[4]
    x = torch.Tensor([x1, y1, x1+w, y1+h])
    return x

target_dir = "visualize_data/inference"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

with torch.inference_mode():
    inference_dec_set, inference_bbox_set = [], [] # output is a 5 length 1D tensor
    target_dec_set, target_bbox_set = [], []       # dec, (bbox)
    acc_good_count = 0
    for idx, ((tensor_in, tensor_out), image_bbox) in tqdm(enumerate(zip(test_data_tensors,data_test))):
        image, bbox = image_bbox
        decision = bbox[0]
        _bbox = bbox[1:]
        tensor_in = tensor_in.to(device)
        inference_bbox = model(tensor_in)
        #print(f"test {idx}: original bbox {bbox}, inference_bbox {inference_bbox}")
        #print(f"test: inference_bbox {inference_bbox}, original_bbox {tensor_out}")
        if device != 'cpu':
            tensor_out.cpu()
            inference_bbox.cpu()
        target_bbox_set.append(tensor_out.squeeze(0).numpy()[1:])
        inference_bbox_set.append(inference_bbox.squeeze(0).numpy()[1:])
        target_dec_set.append(tensor_out.squeeze(0).numpy()[0])
        dec_prob = torch.sigmoid(inference_bbox.squeeze(0)[0]) # doesnt work as expected (ignore this prob)
        #print(f"logit: {dec_prob*100:.2f}%")
        inference_dec_set.append(dec_prob) # logit -> prob
        iou = box_iou(make_xy1wh_xy1xy2(tensor_out.squeeze()).unsqueeze(0),
                      make_xy1wh_xy1xy2(inference_bbox.squeeze()).unsqueeze(0))
        print_image_with_inference(target_dir, idx, image_bbox, inference_bbox.squeeze().numpy(), iou.squeeze())
        #print(f"IoU: {iou}")
        if iou>args.iou_cut: acc_good_count += 1
    if not fullValidationMode:
        print(f"Accuracy(>{args.iou_cut}): {100*acc_good_count/len(test_data_tensors):.1f}%")
    print(f"Results have been stored in {os.getcwd()}/{target_dir}")
