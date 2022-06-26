""" Main engines. """
import json
from pathlib import Path
from typing import Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from matplotlib.patches import Rectangle
import pandas as pd
from PIL.Image import Image
import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms import ToPILImage


class ObjDetDataset:
    """ Interface that facilitates interaction with the dataset. """
    def __init__(self,
                 annot_path: str,
                 image_path: str) -> None:
        """
        Constructor. Load and reorganize instances annotation JSON.

        Args:
            annot_path: path to COCO-style instance annotation file
            image_path: path to image directory corresponding to annotations
        """
        with open(annot_path, 'r', encoding='UTF-8') as file:
            info = json.load(file)

        # keep object detection annot, toss instance segmentation annot
        for obj_annot in info['annotations']:
            del obj_annot['segmentation']
        annot = pd.DataFrame(info['annotations'])
        images = pd.DataFrame(info['images'])

        # define mappings from image id to sample number and vice versa
        # use "images" not "annot" because some images have no COCO objects
        img_to_samp = {
            img_id: samp_num for samp_num, img_id
            in enumerate(images.id.unique())
        }
        samp_to_img = {v: k for k, v in img_to_samp.items()}

        # link object-level metadata to sample number
        # makes it easier to pull out every bounding box in an image
        annot['samp'] = annot.image_id.map(img_to_samp)
        annot = annot.set_index('samp').sort_index()
        images['samp'] = images.id.map(img_to_samp)
        images = images.set_index('samp').sort_index()

        # assign to attributes
        self.image_path = Path(image_path)
        self.img_to_samp = img_to_samp
        self.samp_to_img = samp_to_img
        self.annotations = annot
        self.images = images
        self.n_images = len(samp_to_img)
        self.categories = pd.DataFrame(info['categories']).set_index('id')

    def __len__(self) -> int:
        """ Returns: number of images in dataset. """
        return self.n_images

    def __getitem__(self, samp: int):
        """
        Get a dataset sample.

        Args:
            samp: sample_number

        Returns: sample image, sample bounding boxes
        """
        boxes = self.get_boxes(samp)
        img = self.get_image(samp)
        return img, boxes

    def get_boxes(self, samp: int) -> Tensor:
        """
        Get object bounding boxes.

        Args:
            samp: sample number

        Returns: bounding boxes (shape ? x 4)
        """
        try:
            boxes = self.annotations.bbox.loc[samp]
            if isinstance(boxes, pd.Series):
                boxes = boxes.tolist()
            # pylint: disable=no-member
            boxes = torch.tensor(boxes)
            # pylint: enable=no-member
        except KeyError:
            # pylint: disable=no-member
            boxes = torch.zeros(0, 4)  # sample has no bounding boxes
            # pylint: enable=no-member
        return boxes

    def get_image(self,
                  samp: int,
                  as_pil: bool = False) -> Union[Tensor, Image]:
        """
        Get image.

        Args:
            samp:   sample number
            as_pil: toggles return type b/w torch tensor or pillow image

        Returns: image
        """
        file_name = self.images.file_name.loc[samp]
        img = read_image(str(self.image_path / file_name))
        if as_pil:
            img = ToPILImage()(img)
        return img

    def plot_samp(self, samp: int, alpha: float = 0.2) -> Tuple[Figure, Axes]:
        """
        Plot sample image with bounding box overlays.

        Args:
            samp:  sample number
            alpha: transparency value for bounding boxes

        Returns: plot figure, plot axes
        """
        fig, axes = plt.subplots()
        axes.imshow(self.get_image(samp, True))
        for box in self.get_boxes(samp):
            axes.add_patch(
                Rectangle(
                    xy=box[:2],
                    width=box[2],
                    height=box[3],
                    facecolor='r',
                    edgecolor='w',
                    linewidth=2,
                    alpha=alpha
                )
            )
        return fig, axes
