import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

def print_image_info(img, step):
    """辅助函数，用于打印图像的基本信息（如大小、模式）"""
    print(f"Step: {step}")
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
    print("-" * 50)

def get_dataloaders(data_dir, batch_size=32):
    # 训练集的数据增强变换
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集和测试集的预处理（不包括数据增强）
    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val_test)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_val_test)

    # 输出每个数据集中的图片数量
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def show_preprocessed_images(data_loader):
    """显示批处理中的几张图像"""
    data_iter = iter(data_loader)
    images, _ = next(data_iter)

    # 选择前8张图像
    grid = make_grid(images[:8], nrow=4)  # 4列展示8张图片
    grid = grid.permute(1, 2, 0).cpu().numpy()  # 转换为matplotlib兼容的格式

    # 恢复标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    grid = std * grid + mean  # 恢复标准化
    grid = np.clip(grid, 0, 1)  # 限制像素值在0到1之间

    # 使用matplotlib显示图片
    plt.imshow(grid)
    plt.title("Preprocessed Images")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    data_dir = 'chest_xray/'
    train_loader, val_loader, test_loader = get_dataloaders(data_dir)

    # 显示几张预处理后的图片
    show_preprocessed_images(train_loader)
