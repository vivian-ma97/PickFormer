import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from torchvision import transforms


class ImageSliceDataset(Dataset):
    def __init__(self, img_npy_folder, mask_folder, step=512, transform=None, visualize_full=False,
                 visualize_slice=False, save_img_folder=None, save_mask_folder=None):
        """
        初始化数据集，包含图像和标签的切片参数
        :param img_npy_folder: 保存图像 .npy 文件的文件夹路径
        :param mask_folder: 保存标签（mask）图像的文件夹路径
        :param step: 切片的步长
        :param transform: 图像和标签的预处理操作
        :param visualize_full: 是否可视化完整图像
        :param visualize_slice: 是否可视化图像切片
        :param save_img_folder: 保存图像切片的文件夹路径
        :param save_mask_folder: 保存标签切片的文件夹路径
        """
        self.img_npy_folder = img_npy_folder
        self.mask_folder = mask_folder
        self.step = step
        self.transform = transform
        self.visualize_full = visualize_full
        self.visualize_slice = visualize_slice
        self.save_img_folder = save_img_folder
        self.save_mask_folder = save_mask_folder
        self.img_npy_files = [os.path.join(img_npy_folder, f) for f in os.listdir(img_npy_folder) if f.endswith('.npy')]


        # 确保 mask 文件夹中的图像与 npy 文件的顺序一致
        self.mask_files = [os.path.join(mask_folder, f.replace('.npy', '.png')) for f in os.listdir(img_npy_folder) if
                           f.endswith('.npy')]

        if len(self.img_npy_files) != len(self.mask_files):
            raise ValueError("The number of image files and mask files must be the same.")

        if not self.img_npy_files:
            raise ValueError(f"No .npy files found in the directory: {img_npy_folder}")

        # 创建切片索引列表
        self.slice_indices = []
        for idx, img_npy_file in enumerate(self.img_npy_files):
            data = np.load(img_npy_file)
            mask = Image.open(self.mask_files[idx]).convert('L')  # 加载 mask 图像，并转换为灰度模式

            # 可视化完整图像（如果需要）
            if self.visualize_full:
                self.visualize_full_image(data, title=f"Full Image {idx}")

            img_width, img_height = data.shape[1], data.shape[0]
            min_point, change_points = self.find_change_points(data)

            # 检查突变点位置是否合适进行上下裁剪
            if min_point - 350 >= 0 and min_point + 100 <= img_height:
                # 固定垂直裁剪范围
                y_start = min_point - 300
                y_end = min_point + 100

                # 裁剪图像和标签的上下部分，仅保留 (y_start, y_end) 的区域
                cropped_image = data[y_start:y_end, :]
                cropped_mask = np.array(mask)[y_start:y_end, :]

                # 沿着 x 轴进行滑动窗口裁剪
                for x in range(0, img_width, self.step):
                    if x + self.step > img_width:
                        # 最后一个切片需要重叠处理
                        x = img_width - self.step

                    image_slice = cropped_image[:, x:x + self.step]
                    mask_slice = cropped_mask[:, x:x + self.step]

                    image_slice = np.array(image_slice)
                    # 将切片图像和标签调整为 512x512 大小
                    resized_image_slice = Image.fromarray(image_slice).resize((512, 512))
                    resized_mask_slice = Image.fromarray(mask_slice).resize((512, 512))

                    # 保存切片的图像和标签
                    if self.save_img_folder and self.save_mask_folder:
                        # 确保目录存在
                        os.makedirs(self.save_img_folder, exist_ok=True)
                        os.makedirs(self.save_mask_folder, exist_ok=True)

                        # 构造图像和标签保存路径
                        img_save_path = os.path.join(self.save_img_folder, f"image_slice_{idx}_{x}.png")
                        mask_save_path = os.path.join(self.save_mask_folder, f"mask_slice_{idx}_{x}.png")

                        # 保存图像切片
                        plt.imsave(img_save_path, resized_image_slice, cmap='gray')
                        # 保存标签切片
                        plt.imsave(mask_save_path, resized_mask_slice, cmap='gray')

                    # 可视化切片图像和标签（如果需要）
                    if self.visualize_slice:
                        self.visualize_slice_image(np.array(resized_image_slice), title=f"Image Slice {idx}_{x}")
                        self.visualize_slice_image(np.array(resized_mask_slice), title=f"Mask Slice {idx}_{x}")

                    # 保存切片的索引和信息
                    self.slice_indices.append((img_save_path, mask_save_path))

            else:
                # 如果没有有效的突变点或突变点不在裁剪范围内，则跳过该图像并打印其文件名
                print(f"Skipping image {img_npy_file}, no valid change points or invalid cropping range.")

        if not self.slice_indices:
            raise ValueError(
                "No valid slices were generated from the .npy files. Please check your change point detection logic.")

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """
        获取某个大图像中的一个切片及其对应的标签
        """
        img_path, mask_path = self.slice_indices[idx]

        # 加载图像切片和标签切片
        image = Image.open(img_path).convert('RGB')  # 图像
        mask = Image.open(mask_path).convert('L')  # 标签（mask）

        # 转换为 Tensor
        image_tensor = transforms.ToTensor()(image)
        mask_tensor = transforms.ToTensor()(mask)

        # 对切片进行转换（如果有指定转换）
        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def find_change_points(self, data):
        """
        使用 Sobel 算子计算梯度并找到突变点
        """
        Gx = sobel(data, axis=1)
        Gy = sobel(data, axis=0)

        # 计算梯度的大小
        Gmag = np.sqrt(Gx ** 2 + Gy ** 2)

        # 设置阈值
        threshold = np.max(Gmag) * 0.8

        # 识别突变点
        change_points = np.where(Gmag > threshold)[0]

        # 取出 y 方向的突变点并去除重复值
        unique_change_points = np.unique(change_points)

        # 找到最低的突变点
        if len(unique_change_points) > 0:
            min_point = np.max(unique_change_points)
        else:
            min_point = 0

        return min_point, unique_change_points

    def visualize_full_image(self, data, title="Full Image Visualization"):
        """
        可视化输入的完整灰度图像并保存
        """
        plt.figure(figsize=(6, 4))
        plt.imshow(data, cmap='gray')
        plt.title(title)
        plt.xlabel('Width')
        plt.ylabel('Height')

        if self.save_img_folder:
            full_img_save_path = os.path.join(self.save_img_folder, f"full_image.png")
            os.makedirs(self.save_img_folder, exist_ok=True)
            plt.savefig(full_img_save_path, bbox_inches='tight')  # 保存完整图像
        plt.show(block=False)

    def visualize_slice_image(self, slice_data, title="Slice Image Visualization"):
        """
        可视化输入的图像切片并保存
        """
        plt.figure(figsize=(4, 4))
        plt.imshow(slice_data, cmap='gray')
        plt.title(title)
        plt.xlabel('Width')
        plt.ylabel('Height')

        if self.save_img_folder:
            slice_img_save_path = os.path.join(self.save_img_folder, f"slice_image.png")
            os.makedirs(self.save_img_folder, exist_ok=True)
            plt.savefig(slice_img_save_path, bbox_inches='tight')  # 保存图像切片
        plt.show(block=False)
        plt.close()


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class ToTensor:
    def __call__(self, image, mask):
        return transforms.ToTensor()(image), transforms.ToTensor()(mask)


class DualResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = transforms.Resize(self.size)(image)
        mask = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST)(mask)
        return image, mask