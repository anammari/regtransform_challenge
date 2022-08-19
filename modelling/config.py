##################################################
## configuration script for storing paths to
## datasets and COCO files and model paraneters
##################################################
## Author: Ahmad Ammari
# ################################################

# path to own data and COCO files
train_data_dir = "IB-SEC.v2i.coco/train"
train_coco = "IB-SEC.v2i.coco/_annotations.coco.modified.filtered.train.json"
val_data_dir = "IB-SEC.v2i.coco/valid"
val_coco = "IB-SEC.v2i.coco/_annotations.coco.modified.valid.json"
test_data_dir = "IB-SEC.v2i.coco/test"
test_coco = "IB-SEC.v2i.coco/_annotations.coco.modified.test.json"


# Batch size
train_batch_size = 1
val_batch_size = 1
test_batch_size = 1


# Params for dataloader
train_shuffle_dl = True
val_shuffle_dl = False
test_shuffle_dl = True
num_workers_dl = 4


# Params for training


# Three classes
num_classes = 4
num_epochs = 1  # originally 6; setting to 1 to save processing time

# Hyperparameters
lr = 0.005
momentum = 0.9
weight_decay = 0.005
step_size = 2
gamma = 0.1
