import os
from glob import glob

import clip
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.auto import tqdm

import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/labeled_ola"
num_classes = 36

run = wandb.init(project="clip_cls_36", id="moqhf0te", resume='must')
CONFIG = wandb.config
CONFIG = dict(CONFIG)
run.finish()
wandb.init(project="clip_cls_36", config=CONFIG)
CONFIG = wandb.config 

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

get_activation = {
    'q_gelu': QuickGELU,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU
}

class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.im_paths = glob(os.path.join(root_dir, "*", "*"))
        self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x))
        self.label_dict = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        label = self.label_dict[im_path.split(os.sep)[-2]]
        return img, label


def load_split_train_test(datadir, valid_size=.125):
    train_data = ImageDataset(datadir)
    print(train_data.label_dict)
    test_data = ImageDataset(datadir)
    indices = list(range(len(train_data)))
    split = int(np.floor(valid_size * len(train_data)))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=256,
                                              pin_memory=True, drop_last=False, num_workers=8)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=256,
                                             pin_memory=True, drop_last=False, num_workers=8)

    return trainloader, testloader



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

trainloader, testloader = load_split_train_test(DATA_DIR, .05)
model = Classifier()
model.load_state_dict(torch.load('weights/resnet.pth'))
# model.cls_head.load_state_dict(torch.load(wandb.restore("best_weights_new.pth").name))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["max_lr"], weight_decay=CONFIG["weight_decay"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=CONFIG["weight_decay"])
scaler = amp.GradScaler()
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr=CONFIG["max_lr"],
                                          steps_per_epoch=len(trainloader),
                                          epochs=CONFIG["epochs"],
                                          pct_start=CONFIG["pct_start"],
                                          anneal_strategy=CONFIG["anneal_strategy"]
                                          )
                                          
wandb.define_metric("train_loss", summary="min")
wandb.define_metric("test_loss", summary="min")
wandb.define_metric("accuracy", summary="max")
global_accuracy = 0
# model.clip_model.requires_grad_(False)
for epoch in range(14, CONFIG["epochs"]+1):
    model.train()
    # if epoch==10:
    #     model.clip_model.requires_grad_(True)
    losses = AverageMeter()
    with tqdm(total=len(trainloader), desc=f"Epoch {epoch:>3}/{CONFIG['epochs']}") as pbar:
        for images, lbl in trainloader:
            images = images.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            pbar.update(1)
            with amp.autocast():
                pred = model(images)
                loss = criterion(pred, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            losses.update(loss.detach_(), images.size(0))
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            scaler.update()

        model.eval()
        test_losses = AverageMeter()
        accs = AverageMeter()
        with torch.no_grad():
            for images, lbl in testloader:
                images = images.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                with amp.autocast():
                    pred = model(images)
                    loss = criterion(pred, lbl)
                    ps = pred.softmax(dim=1)
                    acc = (ps.argmax(dim=1) == lbl).float().mean()
                test_losses.update(loss.detach_(), images.size(0))
                accs.update(acc.detach_(), images.size(0))
        accuracy = accs.avg.item()
        info = {
            "train_loss": round(losses.avg.item(), 6),
            "test_loss": round(test_losses.avg.item(), 6),
            "accuracy": round(accuracy, 6),
            "lr": scheduler.get_last_lr()[0],
        }
        pbar.set_postfix(info)
        wandb.log(info)
        if accuracy > global_accuracy:
            global_accuracy = accuracy
            print(f"Saving best model: {accuracy:.4f}")
            # torch.save(model.state_dict(), f"{wandb.run.dir}/best_weights_new.pth")
            torch.save(model.state_dict(),"weights/resnet.pth")