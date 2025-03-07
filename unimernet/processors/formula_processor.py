from unimernet.common.registry import registry
from omegaconf import OmegaConf
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from unimernet.processors.base_processor import BaseProcessor
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision.transforms.functional import resize
import random


class FormulaImageBaseProcessor(BaseProcessor):

    def __init__(self, image_size):
        super(FormulaImageBaseProcessor, self).__init__()
        self.input_size = [int(_) for _ in image_size]
        assert len(self.input_size) == 2

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    @staticmethod
    def crop_margin_numpy(img: np.ndarray) -> np.ndarray:
        """Crop margins of image using NumPy operations"""
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Normalize and threshold
        if gray.max() == gray.min():
            return img

        normalized = (((gray - gray.min()) / (gray.max() - gray.min())) * 255).astype(np.uint8)
        binary = 255 * (normalized < 200).astype(np.uint8)

        # Find bounding box
        coords = cv2.findNonZero(binary)  # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

        # Return cropped image
        return img[y:y + h, x:x + w]

    def prepare_input(self, img, random_padding: bool = False):
        """
        Convert PIL Image or numpy array to properly sized and padded image after:
            - crop margins
            - resize while maintaining aspect ratio
            - pad to target size
        """
        if img is None:
            return None

        # Handle numpy array
        elif isinstance(img, np.ndarray):
            try:
                img = self.crop_margin_numpy(img)
            except Exception:
                # might throw an error for broken files
                return None

            if img.shape[0] == 0 or img.shape[1] == 0:
                return None

            # Resize while preserving aspect ratio
            h, w = img.shape[:2]
            scale = min(self.input_size[0] / h, self.input_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Calculate padding
            pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

            # Create and apply padding
            channels = 3 if len(img.shape) == 3 else 1
            padded_img = np.full((self.input_size[0], self.input_size[1], channels), 255, dtype=np.uint8)
            padded_img[pad_height:pad_height + new_h, pad_width:pad_width + new_w] = resized_img

            return padded_img

        # Handle PIL Image
        elif isinstance(img, Image.Image):
            try:
                img = self.crop_margin(img.convert("RGB"))
            except OSError:
                # might throw an error for broken files
                return None

            if img.height == 0 or img.width == 0:
                return None

            # Resize while preserving aspect ratio
            img = resize(img, min(self.input_size))
            img.thumbnail((self.input_size[1], self.input_size[0]))
            new_w, new_h = img.width, img.height

            # Calculate and apply padding
            padding = self._calculate_padding(new_w, new_h, random_padding)
            return np.array(ImageOps.expand(img, padding))

        else:
            return None

    def _calculate_padding(self, new_w, new_h, random_padding):
        """Calculate padding values for PIL images"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

        return (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

    def _get_padding_values(self, new_w, new_h, random_padding):
        """Get padding values based on image dimensions and padding strategy"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2

        return pad_width, pad_height



@registry.register_processor("formula_image_train")
class FormulaImageTrainProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size=384):
        super().__init__(image_size)

        # Import weather-related augmentations only when initializing this class
        from unimernet.processors.formula_processor_helper.nougat import Bitmap, Dilation, Erosion
        from unimernet.processors.formula_processor_helper.weather import Fog, Frost, Snow, Rain, Shadow

        self.transform = alb.Compose(
            [
                alb.Compose(
                    [
                        Bitmap(p=0.05),
                        alb.OneOf([Fog(), Frost(), Snow(), Rain(), Shadow()], p=0.2),
                        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0,
                                             interpolation=3,
                                             value=[255, 255, 255],
                                             p=1),
                        alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255],
                                           p=.5)],
                    p=.15),
                # alb.InvertImg(p=.15),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                alb.GaussNoise(10, p=.2),
                alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                alb.ImageCompression(95, p=.3),
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        img = self.prepare_input(item, random_padding=True)
        if img is None:
            return img
        return self.transform(image=img)['image'][:1]


    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [384, 384])

        return cls(
            image_size=image_size,
        )


@registry.register_processor("formula_image_multi_scale_train")
class FormulaImageMultiScaleTrainProcessor(FormulaImageTrainProcessor):
    def __init__(self, all_scales):
        for i, scales in enumerate(all_scales):
            all_scales[i] = [int(_) for _ in scales]
        super(FormulaImageMultiScaleTrainProcessor, self).__init__(all_scales[0])
        self.all_scales = all_scales

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        all_scales = cfg.get("all_scales", [[384, 384]])
        return cls(
            all_scales=all_scales
        )

    def reset_scale(self):
        self.input_size = random.choice(self.all_scales)


@registry.register_processor("formula_image_eval")
class FormulaImageEvalProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size):
        super().__init__(image_size)

        self.transform = alb.Compose(
            [
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.transform(image=image)['image'][:1]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [384, 384])

        return cls(image_size=image_size)
