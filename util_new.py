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
        Initialize the dataset with parameters for image–mask slicing and preprocessing.
        
        :param img_npy_folder: Directory containing the input image .npy files.
        :param mask_folder: Directory containing the corresponding ground-truth mask images.
        :param step: Stride of the sliding window for patch extraction.
        :param transform: Transformations applied to both images and masks.
        :param visualize_full: Flag indicating whether to visualize the full image.
        :param visualize_slice: Flag indicating whether to visualize the extracted patches.
        :param save_img_folder: Directory to store the sliced image patches.
        :param save_mask_folder: Directory to store the sliced mask patches.
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



        self.mask_files = [os.path.join(mask_folder, f.replace('.npy', '.png')) for f in os.listdir(img_npy_folder) if
                           f.endswith('.npy')]

        if len(self.img_npy_files) != len(self.mask_files):
            raise ValueError("The number of image files and mask files must be the same.")

        if not self.img_npy_files:
            raise ValueError(f"No .npy files found in the directory: {img_npy_folder}")

    
        self.slice_indices = []
        for idx, img_npy_file in enumerate(self.img_npy_files):
            data = np.load(img_npy_file)
            mask = Image.open(self.mask_files[idx]).convert('L')  # 加载 mask 图像，并转换为灰度模式


            if self.visualize_full:
                self.visualize_full_image(data, title=f"Full Image {idx}")

            img_width, img_height = data.shape[1], data.shape[0]
            min_point, change_points = self.find_change_points(data)


            if min_point - 350 >= 0 and min_point + 100 <= img_height:
         
                y_start = min_point - 300
                y_end = min_point + 100

                
                cropped_image = data[y_start:y_end, :]
                cropped_mask = np.array(mask)[y_start:y_end, :]

            
                for x in range(0, img_width, self.step):
                    if x + self.step > img_width:
                   
                        x = img_width - self.step

                    image_slice = cropped_image[:, x:x + self.step]
                    mask_slice = cropped_mask[:, x:x + self.step]

                    image_slice = np.array(image_slice)
            
                    resized_image_slice = Image.fromarray(image_slice).resize((512, 512))
                    resized_mask_slice = Image.fromarray(mask_slice).resize((512, 512))

                
                    if self.save_img_folder and self.save_mask_folder:
              
                        os.makedirs(self.save_img_folder, exist_ok=True)
                        os.makedirs(self.save_mask_folder, exist_ok=True)

                        img_save_path = os.path.join(self.save_img_folder, f"image_slice_{idx}_{x}.png")
                        mask_save_path = os.path.join(self.save_mask_folder, f"mask_slice_{idx}_{x}.png")

                        plt.imsave(img_save_path, resized_image_slice, cmap='gray')

                        plt.imsave(mask_save_path, resized_mask_slice, cmap='gray')


                    if self.visualize_slice:
                        self.visualize_slice_image(np.array(resized_image_slice), title=f"Image Slice {idx}_{x}")
                        self.visualize_slice_image(np.array(resized_mask_slice), title=f"Mask Slice {idx}_{x}")

          
                    self.slice_indices.append((img_save_path, mask_save_path))

            else:
                print(f"Skipping image {img_npy_file}, no valid change points or invalid cropping range.")

        if not self.slice_indices:
            raise ValueError(
                "No valid slices were generated from the .npy files. Please check your change point detection logic.")

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_indices[idx]
        image = Image.open(img_path).convert('RGB')  
        mask = Image.open(mask_path).convert('L')  

        image_tensor = transforms.ToTensor()(image)
        mask_tensor = transforms.ToTensor()(mask)

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def find_change_points(self, data):
        Gx = sobel(data, axis=1)
        Gy = sobel(data, axis=0)

        Gmag = np.sqrt(Gx ** 2 + Gy ** 2)
        
        threshold = np.max(Gmag) * 0.8
        
        change_points = np.where(Gmag > threshold)[0]

        unique_change_points = np.unique(change_points)

        if len(unique_change_points) > 0:
            min_point = np.max(unique_change_points)
        else:
            min_point = 0

        return min_point, unique_change_points

    def visualize_full_image(self, data, title="Full Image Visualization"):
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
        plt.figure(figsize=(4, 4))
        plt.imshow(slice_data, cmap='gray')
        plt.title(title)
        plt.xlabel('Width')
        plt.ylabel('Height')

        if self.save_img_folder:
            slice_img_save_path = os.path.join(self.save_img_folder, f"slice_image.png")
            os.makedirs(self.save_img_folder, exist_ok=True)
            plt.savefig(slice_img_save_path, bbox_inches='tight')
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
