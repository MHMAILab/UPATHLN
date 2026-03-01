from torch.utils.data import Dataset
from openslide import OpenSlide
import torchvision.transforms as transforms
import numpy as np
import cv2

IMAGENET_MEAN = [0.5840, 0.4081, 0.5929] 
IMAGENET_STD = [0.1883, 0.2188, 0.1600]


class InferDataset(Dataset):
    """
    Data producer that generate all the patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """

    def __init__(
        self,
        wsi_path,
        mask_path,
        img_size,
        crop_size,
        mask_resize=1,
        downsample=64,
        transforms=None,
    ):
        self.wsi_path = wsi_path
        self.slide = OpenSlide(wsi_path)

        self.downsample_4x = 10  # For lymph nodes level0=40x, 4x at low magnification
        self.downsample_10x = 4  # For lymph nodes level0=40x, 10x at high magnification
        
        # Get thumbnail using OpenSlide
        slide_dimensions = self.slide.dimensions
        thumb_size = (int(slide_dimensions[0]/self.downsample_4x), 
                      int(slide_dimensions[1]/self.downsample_4x))
        thumb = self.slide.get_thumbnail(thumb_size)
        self.thumb_4x = np.array(thumb)
        
        self.img_size = img_size
        self.crop_size = crop_size
        self.transforms = self.get_transforms() if transforms is None else transforms
        self.mask_resize = mask_resize
        self.mask_downsample = downsample * mask_resize

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Reduce computation
        self.mask = cv2.resize(
            mask,
            (int(mask.shape[1] / mask_resize), int(mask.shape[0] / mask_resize)),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_x, mask_y = np.where(self.mask == 255)
        self.mask_white = list(zip(mask_x, mask_y))

    def __len__(self):
        return len(self.mask_white)

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(self.crop_size),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __getitem__(self, index):
        mask_x, mask_y = self.mask_white[index]
        wsi_x_center = (mask_x + 0.5) * self.mask_downsample
        wsi_y_center = (mask_y + 0.5) * self.mask_downsample
        wsi_x_topleft = int(
            wsi_x_center - self.img_size * self.downsample_10x / 2
        )
        wsi_y_topleft = int(
            wsi_y_center - self.img_size * self.downsample_10x / 2
        )

        img_10x = self.slide.read_region(
            (wsi_y_topleft, wsi_x_topleft),
            0,
            (
                int(self.img_size * self.downsample_10x),
                int(self.img_size * self.downsample_10x),
            ),
        )
        img_10x = img_10x.resize((self.img_size, self.img_size))
        img_10x = img_10x.convert("RGB")
        img_10x = self.transforms(img_10x)


        wsi_i_topleft = int(wsi_x_center // self.downsample_4x - self.img_size // 2)
        wsi_j_topleft = int(wsi_y_center // self.downsample_4x - self.img_size // 2)
        img_4x = self.thumb_4x[
            wsi_i_topleft : wsi_i_topleft + self.img_size,
            wsi_j_topleft : wsi_j_topleft + self.img_size,
            :,
        ]
        transform_4x = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        img_4x = transform_4x(img_4x)

        return {
            "img_10x": img_10x,
            "img_4x": img_4x,
            "mask_x": mask_x,
            "mask_y": mask_y,
        }
