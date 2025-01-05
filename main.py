import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import timm
import torch
from torch import GradScaler, Tensor, nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_default_device()
cudnn.benchmark = True
pin_memory = True
enable_half = True  # Disable for CPU, it is slower!
scaler = GradScaler(device, enabled=enable_half)


class SimpleCachedDataset(Dataset):
    def __init__(
        self, dataset: Dataset, runtime_transforms: Optional[v2.Transform], cache: bool
    ):
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


mean = (0.507, 0.4865, 0.4409)
sd = (0.2673, 0.2564, 0.2761)

basic_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

test_transforms = v2.Compose([basic_transforms, v2.Normalize(mean, sd, inplace=True)])

runtime_transforms = v2.Compose(
    [
        v2.RandomCrop(size=32, padding=4),
        v2.RandomHorizontalFlip(0.5),
        v2.Normalize(mean, sd, inplace=True),
    ]
)

train_set = CIFAR100_noisy_fine(
    "./fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100",
    download=False,
    train=True,
    transform=basic_transforms,
)
test_set = CIFAR100_noisy_fine(
    "./fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100",
    download=False,
    train=False,
    transform=test_transforms,
)
train_set = SimpleCachedDataset(train_set, runtime_transforms, True)
test_set = SimpleCachedDataset(test_set, None, True)

train_loader = DataLoader(train_set, batch_size=50, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

model = timm.create_model("resnext50_32x4d", pretrained=True)
model.fc = nn.Linear(2048, 100)

# model = timm.create_model("convnext_base", pretrained=True)
# model.head = nn.Sequential(
#     nn.AdaptiveAvgPool2d((1, 1)),
#     nn.Flatten(),
#     nn.Linear(model.num_features, 100),
# )

model = model.to(device)
# model = torch.jit.script(model)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


def train():
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
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


@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


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


best = 0.0
epochs = list(range(30))
with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train()
        val_acc = val()
        if val_acc > best:
            best = val_acc
        tbar.set_description(
            f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}"
        )
data = {"ID": [], "target": []}


for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("./submission.csv", index=False)
