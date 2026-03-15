import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ExternalDataset(Dataset):
    """ 用于阶段一真实 HR 图片加载 (如 DIV2K)，工程发挥补充 """
    def __init__(self, root_dir, patch_size=128):
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(img)

class SRDataset(Dataset):
    """ 用于阶段二评测的图片加载 """
    def __init__(self, root_dir, is_train=False):
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(img), os.path.basename(self.image_files[idx])
