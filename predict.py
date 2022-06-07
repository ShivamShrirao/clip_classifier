import os
from glob import glob
import shutil

import clip
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(project="clip_cls_36", id="gdspqrrh", resume='must')
CONFIG = wandb.config
print(CONFIG)

datadir = "data/rbg_newtest_filtered/"
num_classes = 36

class ImageData(Dataset):
    def __init__(self, root_dir):
        self.im_paths = glob(os.path.join(root_dir, "*"))

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        return img, im_path


train_data = ImageData(datadir)
dataloader = DataLoader(train_data, batch_size=512, pin_memory=True, drop_last=False, num_workers=8)

num_classes = 36
# for i in range(num_classes):
#     os.makedirs(f"data/newtest_finetune/{i}", exist_ok=True)

input_nc = {"RN50": 1024, "ViT-B/32": 512}[CONFIG["clip_type"]]

clip_model, preprocess = clip.load(CONFIG["clip_type"], device)
clip_model.eval()
mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)
std = 255 * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


get_activation = {
    'q_gelu': QuickGELU,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU
}

cls_head = nn.Sequential(
    # nn.Dropout(CONFIG["dropout"]),
    nn.Linear(input_nc, CONFIG["hid_dim"]),
    get_activation[CONFIG["activation"]](),
    nn.Dropout(CONFIG["dropout"]),
    nn.Linear(CONFIG["hid_dim"], num_classes)
).to(device).eval()

# cls_head.load_state_dict(torch.load(wandb.restore("best_weights_new.pth").name))
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)
        self.std = 255 * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)
        self.clip_model, preprocess = clip.load(CONFIG["clip_type"], device)
        self.clip_model = self.clip_model.float()
        self.cls_head = nn.Sequential(
            # nn.Dropout(CONFIG["dropout"]),
            nn.LazyLinear(CONFIG["hid_dim"]),
            get_activation[CONFIG["activation"]](),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(CONFIG["hid_dim"], num_classes)
        ).to(device).train()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean).div_(self.std)
        x = self.clip_model.visual(x)
        x = self.cls_head(x)
        return x
cls_head = Classifier()
cls_head.load_state_dict(torch.load("weights/new2.pth"))

labels = np.array([0,45,90,135,180,225,270,315])
# for i in labels:
#     os.makedirs(f"data/newtest_finetune/{i}", exist_ok=True)
# os.makedirs("data/newtest_finetune/reject",exist_ok=True)
labeldict = {
    0:0,
    1:'reject',
    2:45,
    3:45,
    4:45,
    5:45,
    6:'reject',
    7:90,
    8:90,
    9:90,
    10:'reject',
    11:135,
    12:135,
    13:135,
    14:135,
    15:'reject',
    16:180,
    17:180,
    18:180,
    19:'reject',
    20:225,
    21:225,
    22:225,
    23:225,
    24:'reject',
    25:270,
    26:270,
    27:270,
    28:'reject',
    29:315,
    30:315,
    31:315,
    32:315,
    33:'reject',
    34:0,
    35:0
}
def get_gt(path):
    labels = {"1":45,"2":90,"3":135,"4":180,"5":225,"6":270,"7":315,"8":0}
    lb = None
    if 'reshoot' in path:
        lb = path.split("_")[-3]
    else:
        lb = path.split('.')[0][-1]
    if lb in labels.keys():
        return labels[lb]
    else:
        return None

total = 0
correct = 0

with torch.inference_mode(), amp.autocast():
    for images, paths in tqdm(dataloader):
        images = images.to(device, non_blocking=True)
        # images = images.unsqueeze(1)
        # images = images.repeat(1, 3, 1, 1)
        # images = (images - mean).div_(std)
        # features = clip_model.encode_image(images)
        x = cls_head(images)
        x = torch.softmax(x, dim=1)
        top_class = x.argmax(dim=1).cpu().numpy()
        for pred, pth in zip(top_class, paths):
            # print(labeldict[pred],lbl)
            # break
            # try:
            #     lbl = labels[pth.split('.')[0][-1]]
            #     if abs(lbl-labeldict[pred]) <= 15 or (345<=labeldict[pred]<360 and lbl==0):
            #         correct+=1
            #     total+=1
            # except Exception as e:
            #     print(e)
            p = labeldict[pred]
            if p != 'reject':
                gt = get_gt(pth)
                if p == gt:
                    correct+=1
                if gt is not None:
                    total+=1
            else:
                total+=1
            # shutil.copy(pth, f"data/newtest_finetune/{p}")
        # break
print(correct)
print(total)
print(correct/total)