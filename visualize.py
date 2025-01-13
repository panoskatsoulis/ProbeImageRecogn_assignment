import random, os, matplotlib
import matplotlib.pyplot as plotter
import matplotlib.patches as patches

from sklearn.model_selection import train_test_split
from dataset import ProbesDataset, images_fullinfo
#from dataset import flyability_train, flyability_test
#print(annotations, images, flyability_train, flyability_test)

flyability = ProbesDataset(
    _dir = "/home/kpanos/flyability_interview_project/probe_dataset/probe_images",
    labels = images_fullinfo
    #transform = transforms.ToTensor()
)
data_train, data_test = train_test_split(flyability, test_size=0.2, random_state=10)
# take 2 random items
imgfile_train, img_train = random.choice(data_train)
imgfile_test, img_test = random.choice(data_test)
#imginfo_train = list(filter(lambda x: x['file_name']==os.path.basename(imgfile_train), images_fullinfo))[0]
#imginfo_test = list(filter(lambda x: x['file_name']==os.path.basename(imgfile_test), images_fullinfo))[0]


def print_image(_dir, _img, _idx):
    image, bbox = _img
    print(f"entry file: {_idx}")
    fig, item = plotter.subplots()
    # box_x1, box_y1, box_x2, box_y2 = _imginfo['bbox']
    # width  = box_x2 - box_x1
    # height = box_y2 - box_y1
    box_x1, box_y1, width, height = bbox
    rect = patches.Rectangle((box_x1, box_y1), width, height,
                             linewidth=2, edgecolor='r', facecolor='none')
    item.imshow(image)
    item.add_patch(rect)
    fig.suptitle(f"idx: {_idx}, bbox: {bbox}")
    fig.savefig(f"{_dir}/image_{_idx}.png")
    plotter.close(fig)


print(f"entries in train sample are {len(data_train)} of type {type(img_train)}")
target_dir = "/home/kpanos/flyability_interview_project/visualize_data/training"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
random.shuffle(data_train)
for i, image in enumerate(data_train):
    print_image(target_dir, image, i)
    #if i >= 6: break

print(f"entries in test sample are {len(data_test)} of type {type(img_test)}")
target_dir = "/home/kpanos/flyability_interview_project/visualize_data/testing"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
# random.shuffle(data_test)
for i, image in enumerate(data_test):
    print_image(target_dir, image, i)
    #if i >= 6: break
