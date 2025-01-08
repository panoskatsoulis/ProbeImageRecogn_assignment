# PyTorch, torchvision
import torch, torchvision, os, json
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image

# matplotlib for visualization
#import matplotlib.pyplot as plt

class ProbesDataset(torch.utils.data.Dataset):
    def __init__(self, _dir, _transform=None):
        self.dir = _dir
        self.transform = _transform
        self.labels = os.listdir(_dir) #to make it with json

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, _idx):
        _name = os.path.join(self.dir, self.labels[_idx])
        _image = Image.open(_name)
        if self.transform:
            _image = self.transform(_image)
        return (_name, _image)

flyability_dataset = ProbesDataset(
    _dir = "/home/kpanos/flyability_interview_project/probe_dataset/probe_images",
    _transform = transforms.ToTensor()
    #_transform = transforms.Compose([transforms.Resize((256, 256)), # maybe shrink them to fit in memory
    #                                transforms.ToTensor()])
)

# Function to read a JSON file and convert it to a dictionary
json_path = '/home/kpanos/flyability_interview_project/probe_dataset/probe_labels.json'
flyability_labels = []
with open(json_path, 'r') as _f:
    flyability_labels = json.load(_f)
annotations = flyability_labels['annotations']
images = flyability_labels['images']

from sklearn.model_selection import train_test_split
flyability_train, flyability_test = train_test_split(flyability_dataset, test_size=0.2, random_state=10)

from copy import deepcopy
def associate_dict_byId(dict_a, dict_b):
    _res = deepcopy(dict_a)
    for img in _res:
        _id = img['id']
        toAppend = list(filter(lambda x: x['image_id'] == _id, dict_b))[0]
        #print(f"> {img}, {toAppend}")
        img['bbox'] = toAppend['bbox']
        #print(f"--> {img}")
    return _res

images_fullinfo = associate_dict_byId(images, annotations)

#train_loader = torch.utils.data.DataLoader(flyability_train,
#                                          batch_size=4,
#                                          shuffle=True,
#                                          num_workers=4)

if __name__ == "__main__":
    print(f"full: {len(flyability_dataset)},  train: {len(flyability_train)}, test: {len(flyability_test)}")

    # Print the dictionary
    import pprint
    pprint.pprint(images_fullinfo[0], depth=2)
    #
    #pprint.pprint(annotations[0], depth=2)
    #sel_images_0 = list(filter(lambda x: x['id']==annotations[0]['image_id'], images))
    #pprint.pprint(sel_images_0, depth=2)


