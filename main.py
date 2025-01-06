import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from typing import Optional, Callable
import os
import timm
import numpy as np
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import datetime
from torch.utils.data.dataloader import default_collate
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True  # Disable for CPU, it is slower!
scaler = GradScaler(device, enabled=enable_half)


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset: Dataset, runtime_transforms: Optional[v2.Transform], cache: bool):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.runtime_transforms = runtime_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        if self.runtime_transforms is None:
            return image, label
        return self.runtime_transforms(image), label
    
class CIFAR100_noisy_fine(Dataset):
    """
    See https://github.com/UCSC-REAL/cifar-10-100n, https://www.noisylabels.com/ and `Learning with Noisy Labels
    Revisited: A Study Using Real-World Human Annotations`.
    """

    def __init__(
        self, root: str, train: bool, transform: Optional[Callable], download: bool
    ):
        cifar100 = CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
        data, targets = tuple(zip(*cifar100))

        if train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} need {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            
            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]
    
mean=(0.507, 0.4865, 0.4409)
sd=(0.2673, 0.2564, 0.2761)

# mean = (0.5071, 0.4867, 0.4408)
# sd = (0.2675, 0.2565, 0.2761)

# mean = (0.4914, 0.4822, 0.4465) 
# sd = (0.2023, 0.1994, 0.2010)

basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

test_transforms = v2.Compose([
    basic_transforms,
    v2.Normalize(mean, sd, inplace=True)
])

# runtime_transforms = v2.Compose([
#     v2.RandomCrop(size=32, padding=4),
#     v2.RandomHorizontalFlip(0.5),
#     v2.Normalize(mean, sd, inplace=True),
# ])

runtime_transforms = v2.Compose([
    v2.RandomCrop(size=32, padding=4),
    v2.RandomHorizontalFlip(0.5),
    v2.RandAugment(num_ops=2, magnitude=9), 
    v2.RandomErasing(p=0.1),  
    v2.Normalize(mean, sd, inplace=True),
])

train_set = CIFAR100_noisy_fine('./fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100', download=False, train=True, transform=basic_transforms)
test_set = CIFAR100_noisy_fine('./fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100', download=False, train=False, transform=test_transforms)
train_set = SimpleCachedDataset(train_set, runtime_transforms, True)
test_set = SimpleCachedDataset(test_set, None, True)


cutmix = v2.CutMix(num_classes=100)
mixup = v2.MixUp(num_classes=100, alpha=0.2)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


train_loader = DataLoader(train_set, batch_size=100, shuffle=True, 
                        collate_fn=collate_fn, pin_memory=pin_memory, 
                        drop_last=True)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

# CONVNEXT PRETRAINED - 1
model = timm.create_model("convnext_base", pretrained=True)
model.head = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(model.num_features, 100),
)

UNFREEZE_EPOCH = 15

# Freeze earlier layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the final fully connected layer
for param in model.head.parameters():
    param.requires_grad = True


model = model.to(device)
# model = torch.jit.script(model)  # does not work for this model
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)

# optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

lr_warmup_epochs = 5  
total_epochs = 100

warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.01, 
    end_factor=1.0, 
    total_iters=lr_warmup_epochs
)
main_scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=total_epochs - lr_warmup_epochs
)


def train():
    model.train()
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total

warm_up_epochs = 25
loss_threshold = 2
dynamic_threshold_decay = 0.995

def train_loss_filtering(epoch):
    model.train()
    correct = 0
    total = 0
    global loss_threshold

    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        
        if isinstance(labels, torch.Tensor):
            # No mixup/cutmix was applied
            labels = labels.to(device, non_blocking=True)
            targets_a = targets_b = labels
            lam = 1.
        else:
            # mixup/cutmix was applied
            targets_a, targets_b, lam = labels
            targets_a = targets_a.to(device, non_blocking=True)
            targets_b = targets_b.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets_a)  # Use only primary targets for loss

        if epoch >= warm_up_epochs:
            # Instead of filtering the batch, use the loss information to weight samples
            sample_losses = torch.nn.functional.cross_entropy(outputs, targets_a, reduction='none')
            weights = (sample_losses < loss_threshold).float()
            if weights.sum() == 0:
                continue
            
            # Apply weights to the loss instead of filtering
            loss = (loss * weights).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets_a.size(0)
        correct += predicted.eq(targets_a).sum().item()

    if epoch >= warm_up_epochs:
        loss_threshold *= dynamic_threshold_decay

    return 100.0 * correct / total

@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return accuracy, avg_loss

@torch.inference_mode()
def inference():
    model.eval()
    
    labels = []
    
    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)
    
    return labels

def save_checkpoint(epoch, model, optimizer, best_val_acc, timestamp, checkpoint_name="checkpoint", checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_{timestamp}.pth")
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def load_checkpoint(checkpoint_path="checkpoint.pth"):
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None, None, 0.0

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    best_val_acc = state["best_val_acc"]
    start_epoch = state["epoch"] + 1
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {state['epoch']})")
    return start_epoch, best_val_acc

best = 0.0
epochs = list(range(total_epochs))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train_loss_filtering(epoch)

        if epoch < lr_warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        val_acc, avg_loss = val()
        if val_acc > best:
            best = val_acc
            save_checkpoint(epoch, model, optimizer, best, timestamp, checkpoint_name="convnext")
        if epoch == UNFREEZE_EPOCH:
            for param in model.parameters():
                param.requires_grad = True
        
        current_lr = optimizer.param_groups[0]['lr']
        tbar.set_description(
            f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}, "
            f"Val loss: {avg_loss:.2f}, LR: {current_lr:.6f}"
        )
        # tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}, Val loss: {avg_loss:.2f}")

# Validation

# @torch.inference_mode()
# def _val(model, tta=False):
#     model.eval()
#     correct = 0
#     total = 0
#     total_loss = 0.0

#     for inputs, targets in test_loader:
#         inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
#         with torch.autocast(device.type, enabled=False):
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#         # Basic Test Time Augmentation
#         if tta:
#             outputs += model(v2.RandomHorizontalFlip(1)(inputs))

#         total_loss += loss.item()

#         predicted = outputs.argmax(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     avg_loss = total_loss / len(test_loader)
#     accuracy = 100.0 * correct / total

#     return accuracy

# BEST_CHECKPOINT_PATH = "./checkpoints/convnext_best3_20250106_190405_69_66.pth"

# best_model = timm.create_model("convnext_base", pretrained=True)
# best_model.head = nn.Sequential(
#     nn.AdaptiveAvgPool2d((1, 1)),
#     nn.Flatten(),
#     nn.Dropout(0.5),
#     nn.Linear(best_model.num_features, 100),
# )
# best_model = best_model.to(device)

# checkpoint = torch.load(BEST_CHECKPOINT_PATH)
# best_model.load_state_dict(checkpoint["model_state_dict"])

# _val(best_model, tta=True)