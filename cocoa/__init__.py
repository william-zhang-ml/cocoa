""" Main engines. """
import json
from pathlib import Path
from typing import Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from matplotlib.patches import Rectangle
import numpy as np
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

        # normalize bounding box with respect to size of image
        image_width = annot.index.map(images.width)
        image_height = annot.index.map(images.height)
        boxes = np.stack(annot.bbox)
        annot['x1'] = boxes[:, 0] / image_width
        annot['y1'] = boxes[:, 1] / image_height
        annot['width'] = boxes[:, 2] / image_width
        annot['height'] = boxes[:, 3] / image_height

        # cast
        for col in annot:
            if annot.dtypes[col] == np.float64:
                annot[col] = annot[col].astype(np.float32)

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

    def __getitem__(self, samp: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a dataset sample.

        Args:
            samp: sample_number

        Returns: sample image, normalized bounding boxes, and labels
        """
        img = self.get_image(samp)
        boxes = self.get_normalized_boxes(samp)
        labels = self.get_labels(samp)
        return img, boxes, labels

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

    def get_boxes(self, samp: int) -> Tensor:
        """
        Get object bounding boxes.

        Args:
            samp: sample number

        Returns: bounding boxes (shape ? x 4)
        """
        try:
            # below indexing needed so 1 box vs many box case gives same type
            boxes = self.annotations.loc[samp:samp]['bbox'].tolist()
            # pylint: disable=no-member
            boxes = torch.tensor(boxes).view(-1, 4)
            # pylint: enable=no-member
        except KeyError:
            # pylint: disable=no-member
            boxes = torch.zeros(0, 4)  # sample has no bounding boxes
            # pylint: enable=no-member
        return boxes.float()  # cast for safety, will not cast if correct type

    def get_normalized_boxes(self, samp: int) -> Tensor:
        """
        Get object bounding boxes normalized with respect to image size.

        Args:
            samp: sample number

        Returns: bounding boxes (shape ? x 4)
        """
        try:
            # below indexing needed so 1 box vs many box case gives same type
            boxes = self.annotations.loc[samp:samp][
                ['x1', 'y1', 'width', 'height']
            ].values
            # pylint: disable=no-member
            boxes = torch.tensor(boxes).view(-1, 4)
            # pylint: enable=no-member
        except KeyError:
            # pylint: disable=no-member
            boxes = torch.zeros(0, 4)  # sample has no bounding boxes
            # pylint: enable=no-member
        return boxes.float()  # cast for safety, will not cast if correct type

    def get_labels(self, samp: int) -> Tensor:
        """
        Get object labels.

        Args:
            samp: sample number

        Returns: bounding box labels (shape ?)
        """
        try:
            # below indexing needed so 1 box vs many box case gives same type
            labels = self.annotations.loc[samp:samp]['category_id'].values - 1
            # pylint: disable=no-member
            labels = torch.tensor(labels)
            # pylint: enable=no-member
        except KeyError:
            # pylint: disable=no-member
            labels = torch.zeros(0)  # sample has no bounding boxes
            # pylint: enable=no-member
        return labels.long()  # cast for safety, will not cast if correct type

    def plot_samp(self,
                  samp: int,
                  size: Tuple[int, int] = None,
                  alpha: float = 0.2) -> Tuple[Figure, Axes]:
        """
        Plot sample image with bounding box overlays.

        Args:
            samp:  sample number
            alpha: transparency value for bounding boxes

        Returns: plot figure, plot axes
        """
        img = self.get_image(samp, True)
        if size is None:
            boxes = self.get_boxes(samp)
        else:
            img = img.resize(size)
            boxes = self.get_normalized_boxes(samp)
            boxes[:, 0] *= size[0]
            boxes[:, 1] *= size[1]
            boxes[:, 2] *= size[0]
            boxes[:, 3] *= size[1]
        fig, axes = plt.subplots()
        axes.imshow(img)
        for box in boxes:
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
