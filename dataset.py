# PyTorch, torchvision
import torch, torchvision, os, json
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image

# matplotlib for visualization
#import matplotlib.pyplot as plt

class ProbesDataset(torch.utils.data.Dataset):
    def __init__(self, _dir, labels, transform=None):
        self.dir = _dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, _idx):
        #from pdb import set_trace; set_trace()
        filename = list(filter(lambda x: x['idx'] == _idx, self.labels))
        data_out = list(filter(lambda x: x['idx'] == _idx, self.labels))
        if len(data_out)==0:
            white_image = Image.new('RGB', (640, 400), color='white')
            default_out = [0., 0., 0., 0., 0.]
            return (white_image, default_out)
        # print(f"idx {_idx}")
        # print(f"filename {filename}, len {len(filename)}")
        # print(f"data_out {data_out}, len {len(data_out)}")
        assert (len(filename)==1 and len(data_out)==1), "multiple images with the same id in dataset, check dataset"
        filename = filename[0]['file_name']
        data_out = [1.] + data_out[0]['bbox']
        name = os.path.join(self.dir, filename)
        data_in = Image.open(name)
        if self.transform:
            data_in = self.transform(data_in)
            data_out = torch.tensor(data_out)
        return (data_in, data_out)

# Function to read a JSON file and convert it to a dictionary
json_path = '/home/kpanos/flyability_interview_project/probe_dataset/probe_labels.json'
flyability_labels = []
with open(json_path, 'r') as _f:
    flyability_labels = json.load(_f)
annotations = flyability_labels['annotations']
images = flyability_labels['images']

from copy import deepcopy
def associate_dict_byId(dict_a, dict_b):
    _res = deepcopy(dict_a)
    for idx, img in enumerate(_res):
        _id = img['id'] # id and id_image are ONLY used to associate these 2 dictionaries
        toAppend = list(filter(lambda x: x['image_id'] == _id, dict_b))[0]
        #print(f"> {img}, {toAppend}")
        img['bbox'] = toAppend['bbox']
        #print(f"--> {img}")
        img['idx'] = idx # this keeps track of the idx to be used anywhere else, id and id_image have numbers missing
    return _res

images_fullinfo = associate_dict_byId(images, annotations)
images_output = images_fullinfo

flyability_dataset = ProbesDataset(
    _dir = "/home/kpanos/flyability_interview_project/probe_dataset/probe_images",
    labels = images_fullinfo,
    transform = transforms.ToTensor()
    #_transform = transforms.Compose([transforms.Resize((256, 256)), # maybe shrink them to fit in memory
    #                                transforms.ToTensor()])
)

from sklearn.model_selection import train_test_split
flyability_train, flyability_test = train_test_split(flyability_dataset, test_size=0.2, random_state=10)


train_data_loader = torch.utils.data.DataLoader(flyability_train,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=4)

test_data_loader = torch.utils.data.DataLoader(flyability_train,
                                              batch_size=4,
                                              shuffle=False,
                                              num_workers=4)

if __name__ == "__main__":
    print(f"full: {len(flyability_dataset)},  train: {len(flyability_train)}, test: {len(flyability_test)}")

    # Print the dictionary
    import pprint
    pprint.pprint(images_fullinfo[0], depth=2)
    #
    print(type(flyability_dataset))
    pprint.pprint(flyability_dataset[0], depth=2)


