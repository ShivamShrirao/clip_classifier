#!/usr/bin/env python
# coding: utf-8

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

CONFIG = dict(
    clip_type='ViT-B/32',
    epochs=1000,
    max_lr=1e-3,
    pct_start=0.1,
    anneal_strategy='linear',
    weight_decay=2e-4,
    batch_size=4096,
    dropout=0.25,
    hid_dim=512,
    activation='relu'
)

wandb.init(project="clip_cls_9", config=CONFIG)
CONFIG = wandb.config
print(CONFIG)


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


device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/labeled_ola/"


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.im_paths = glob(os.path.join(root_dir, "*", "*"))
        self.classes = os.listdir(root_dir)
        self.label_dict = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)[:,:,::-1]
        img = cv2.resize(img, (224, 224))
        label = self.label_dict[im_path.split(os.sep)[-2]]
        img = img.transpose(2,0,1)
        return img, label


def load_split_train_test(datadir, valid_size=.125):
    train_data = ImageDataset(datadir)
    test_data = ImageDataset(datadir)
    indices = list(range(len(train_data)))
    split = int(np.floor(valid_size * len(train_data)))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=512,
                                              pin_memory=True, drop_last=False, num_workers=8)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=512,
                                             pin_memory=True, drop_last=False, num_workers=8)

    return trainloader, testloader


def get_features(dataloader):
    clip_model, preprocess = clip.load(CONFIG["clip_type"], device)
    clip_model.eval()
    mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)
    std = 255 * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)
    all_features = []
    all_labels = []

    with torch.inference_mode(), amp.autocast():
        for images, labels in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # images = images.unsqueeze(1)
            # images = images.repeat(1, 3, 1, 1)
            images = (images - mean).div_(std)
            features = clip_model.encode_image(images)
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


# Calculate the image features
features_cache = f"{CONFIG['clip_type']}_features.pth"
if os.path.exists(features_cache):
    ft_dict = torch.load(features_cache)
    train_features = ft_dict["train_features"]
    train_labels = ft_dict["train_labels"]
    test_features = ft_dict["test_features"]
    test_labels = ft_dict["test_labels"]
else:
    trainloader, testloader = load_split_train_test(DATA_DIR, .05)
    train_features, train_labels = get_features(trainloader)
    test_features, test_labels = get_features(testloader)
    torch.save({
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels
    }, features_cache)

torch.cuda.empty_cache()

print(torch.unique(train_labels, return_counts=True))
print(torch.unique(test_labels, return_counts=True))

num_classes = 9


class EncodedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


feat_dataset_train = EncodedDataset(train_features, train_labels)
feat_dataset_test = EncodedDataset(test_features, test_labels)
feat_loader_train = DataLoader(feat_dataset_train, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=False)
feat_loader_test = DataLoader(feat_dataset_test, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=False)


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
    nn.Linear(len(train_features[0]), CONFIG["hid_dim"]),
    get_activation[CONFIG["activation"]](),
    nn.Dropout(CONFIG["dropout"]),
    nn.Linear(CONFIG["hid_dim"], num_classes)
).to(device).train()

global_accuracy = 0

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(cls_head.parameters(), lr=CONFIG["max_lr"], weight_decay=CONFIG["weight_decay"])
scaler = amp.GradScaler()
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr=CONFIG["max_lr"],
                                          steps_per_epoch=len(feat_loader_train),
                                          epochs=CONFIG["epochs"],
                                          pct_start=CONFIG["pct_start"],
                                          anneal_strategy=CONFIG["anneal_strategy"]
                                          )

wandb.define_metric("train_loss", summary="min")
wandb.define_metric("test_loss", summary="min")
wandb.define_metric("accuracy", summary="max")


for epoch in range(1, CONFIG["epochs"]+1):
    cls_head.train()
    losses = AverageMeter()
    with tqdm(total=len(feat_loader_train), desc=f"Epoch {epoch:>3}/{CONFIG['epochs']}") as pbar:
        for feats, lbl in feat_loader_train:
            pbar.update(1)
            with amp.autocast():
                pred = cls_head(feats)
                loss = criterion(pred, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            losses.update(loss.detach_(), feats.size(0))
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            scaler.update()

        cls_head.eval()
        test_losses = AverageMeter()
        accs = AverageMeter()
        with torch.no_grad():
            for feats, lbl in feat_loader_test:
                with amp.autocast():
                    pred = cls_head(feats)
                    loss = criterion(pred, lbl)
                    ps = pred.softmax(dim=1)
                    acc = (ps.argmax(dim=1) == lbl).float().mean()
                test_losses.update(loss.detach_(), feats.size(0))
                accs.update(acc.detach_(), feats.size(0))
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
            torch.save(cls_head.state_dict(), f"{wandb.run.dir}/best_weights_new.pth")
