##################################################
## main script for model training and evaluation
##################################################
## Author: Ahmad Ammari
# ################################################


import torch
from torch.optim.lr_scheduler import StepLR
import config
from util import (
    get_model_instance_segmentation,
    collate_fn,
    get_transform,
    myOwnDataset,
)
from engine import train_one_epoch, evaluate


# which PyTorch version
print("Torch version:", torch.__version__)

# create own Dataset
my_dataset = myOwnDataset(
    root=config.train_data_dir, annotation=config.train_coco, transforms=get_transform()
)

# create validation Dataset
dataset_val = myOwnDataset(
    root=config.val_data_dir, annotation=config.val_coco, transforms=get_transform()
)

# create own DataLoader
data_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=config.train_batch_size,
    shuffle=config.train_shuffle_dl,
    num_workers=config.num_workers_dl,
    collate_fn=collate_fn,
)

# create DataLoader for Validation Data
data_loader_test = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=config.val_batch_size,
    shuffle=config.val_shuffle_dl,
    num_workers=config.num_workers_dl,
    collate_fn=collate_fn)


# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# DataLoader is iterable over Dataset
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)

# create object detection model
model = get_model_instance_segmentation(config.num_classes)

# move model to the right device
model.to(device)

# model parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
)

# learning rate scheduler decreases the learning rate by 10x every 2 epochs
lr_scheduler = StepLR(optimizer,
                      step_size=config.step_size,
                      gamma=config.gamma)

len_dataloader = len(data_loader)

# Model Training and Evaluation
for epoch in range(config.num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the validation dataset
    evaluate(model, data_loader_test, device=device)

# Model Persistence
torch.save(model.state_dict(), 'model.pth')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'ckpt.pth')
