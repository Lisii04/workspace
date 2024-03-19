import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

# 加载图像的函数


def load_image(filename):
    """
    从文件加载图像并返回PIL图像对象。

    Args:
        filename (str): 图像文件的路径。

    Returns:
        PIL.Image.Image: 加载的图像对象。
    """
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

# 获取唯一掩码值的函数


def unique_mask_values(idx, mask_dir, mask_suffix):
    """
    获取给定索引的掩码文件的唯一值。

    Args:
        idx (str): 图像索引。
        mask_dir (str): 掩码文件所在的目录。
        mask_suffix (str): 掩码文件的后缀。

    Returns:
        numpy.ndarray: 掩码的唯一值。
    """
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(
            f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

# 自定义数据集类


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        """
        初始化BasicDataset类的实例。

        Args:
            images_dir (str): 图像文件所在的目录。
            mask_dir (str): 掩码文件所在的目录。
            scale (float, optional): 图像缩放比例。默认为1.0。
            mask_suffix (str, optional): 掩码文件的后缀。默认为空字符串。
        """
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # 获取所有图像文件的文件名（不包括隐藏文件）
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(
            join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            # 并行处理获取每个掩码文件的唯一值
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir,
                       mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        # 将所有掩码文件的唯一值合并并排序
        self.mask_values = list(
            sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        """
        对图像进行预处理，包括缩放和转换为张量。

        Args:
            mask_values (list): 掩码的唯一值列表。
            pil_img (PIL.Image.Image): PIL图像对象。
            scale (float): 图像缩放比例。
            is_mask (bool): 是否为掩码图像。

        Returns:
            numpy.ndarray or torch.Tensor: 预处理后的图像或掩码。
        """
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # 根据指定的缩放比例对图像进行缩放
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(
            img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # 对图像和掩码进行预处理
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask,
                               self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

# Carvana数据集类，继承自BasicDataset类


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        """
        初始化CarvanaDataset类的实例。

        Args:
            images_dir (str): 图像文件所在的目录。
            mask_dir (str): 掩码文件所在的目录。
            scale (float, optional): 图像缩放比例。默认为1.0。
        """
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
