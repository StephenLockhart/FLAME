# Copyright (c) OpenMMLab. All rights reserved.
import logging
import re
import os
import os.path as osp
from typing import Sequence

import numpy as np
import torch
from mmengine.visualization import Visualizer
import cv2

from mmagic.registry import VISUALIZERS
from mmagic.structures import DataSample
from mmagic.utils import print_colored_log


@VISUALIZERS.register_module()
class SingleImageVisualizer(Visualizer):
    """Visualize and save each image separately.

    This visualizer will save each image in a separate file, organized by
    the original filename in subdirectories.

    Image to be visualized can be:
        - torch.Tensor or np.array
        - Image sequences of shape (T, C, H, W)
        - Multi-channel image of shape (1/3, H, W)
        - Single-channel image of shape (C, H, W)

    Args:
        fn_key (str): key used to determine file name for saving image.
            Usually it is the path of some input image.
        img_keys (str): keys, values of which are images to visualize.
        pixel_range (dict): min and max pixel value used to denormalize images.
        bgr2rgb (bool): whether to convert the image from BGR to RGB.
        name (str): name of visualizer. Default: 'visualizer'.
        *args and \**kwargs: Other arguments are passed to `Visualizer`.
    """

    def __init__(self,
                 fn_key: str,
                 img_keys: Sequence[str],
                 pixel_range={},
                 bgr2rgb=False,
                 name: str = 'visualizer',
                 *args,
                 **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        self.fn_key = fn_key
        self.img_keys = img_keys
        self.pixel_range = pixel_range
        self.bgr2rgb = bgr2rgb
        self.frame_count = {}  # track the frame count of each video sequence

    def add_datasample(self, data_sample: DataSample, step=0) -> None:
        """Process and save each image separately.

        Args:
            data_sample (DataSample): Single data_sample from data_batch.
            step (int): Global step value to record. Default: 0.
        """
        merged_dict = {
            **data_sample.to_dict(),
        }

        if 'output' in merged_dict.keys():
            merged_dict.update(**merged_dict['output'])

        # get information about the dataset currently being processed
        fn_val = merged_dict[self.fn_key]
        # unify into a list to ease frame-wise naming
        if isinstance(fn_val, list):
            path_list = fn_val
        else:
            path_list = [fn_val]

        # use the parent directory name as the sequence name (consistent with the dataset, e.g. 000, 0900)
        dataset_key = os.path.dirname(path_list[0])
        sequence_name = os.path.basename(dataset_key)

        # initialize the frame counter
        if sequence_name not in self.frame_count:
            self.frame_count[sequence_name] = 0

        # process each image type that needs to be saved
        for k in self.img_keys:
            if k not in merged_dict:
                print_colored_log(
                    f'Key "{k}" not in data_sample or outputs',
                    level=logging.WARN)
                continue

            img = merged_dict[k]

            # PixelData handling
            if isinstance(img, dict) and ('data' in img):
                img = img['data']

            # convert tensor to numpy
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)
                elif img.ndim == 4:
                    img = img.transpose(0, 2, 3, 1)

            # determine the naming strategy from the original path
            # extract the digit width and starting index
            base_name = os.path.splitext(os.path.basename(path_list[0]))[0]
            m = re.search(r'(\d+)$', base_name)
            if m:
                digits = len(m.group(1))
                start_idx = int(m.group(1))
            else:
                digits = 5
                start_idx = 0

            # handle multi-frame images
            if img.ndim == 4:
                num_frames = img.shape[0]
                # if a full frame-path list with matching length is provided, save each frame under its original name
                if isinstance(fn_val, list) and len(fn_val) == num_frames:
                    for frame, fpath in zip(img, path_list):
                        self._save_single_image(frame, sequence_name, k, source_path=fpath)
                        self.frame_count[sequence_name] += 1
                else:
                    # name sequentially from the starting index, keeping the original extension
                    for i in range(num_frames):
                        self._save_single_image(
                            img[i], sequence_name, k,
                            number=start_idx + self.frame_count[sequence_name],
                            digits=digits,
                            source_path=path_list[0])
                        self.frame_count[sequence_name] += 1
            else:
                # single-frame case: use the current counter or the original filename
                if isinstance(fn_val, list) and len(fn_val) >= 1:
                    self._save_single_image(img, sequence_name, k, source_path=path_list[0])
                else:
                    self._save_single_image(
                        img, sequence_name, k,
                        number=start_idx + self.frame_count[sequence_name],
                        digits=digits,
                        source_path=path_list[0])
                self.frame_count[sequence_name] += 1

    def _save_single_image(self, img: np.ndarray, sequence_name: str,
                          img_type: str, source_path: str = '',
                          number: int = None, digits: int = 5) -> None:
        """Save a single image frame.

        Args:
            img (np.ndarray): Image to save.
            sequence_name (str): Name of the sequence (used as subdirectory).
            img_type (str): Type of the image (e.g., 'pred_img').
            source_path (str): Reference path to infer filename and extension.
            number (int): If provided, format as zero-padded number.
            digits (int): Zero-pad width for ``number``.
        """
        # convert grayscale to RGB
        if img.ndim == 2:
            img = np.stack((img, img, img), axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate((img, img, img), axis=2)

        # BGR to RGB
        if self.bgr2rgb:
            img = img[..., ::-1]

        # normalization
        if img.dtype != np.uint8:
            if img_type in self.pixel_range:
                min_, max_ = self.pixel_range.get(img_type)
                img = ((img - min_) / (max_ - min_)) * 255
            img = img.clip(0, 255).round().astype(np.uint8)

        # build the save path (using the original sequence-name directory)
        save_dir = osp.join(self._vis_backends['LocalVisBackend']._save_dir,
                            sequence_name)
        
        # ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # decide the extension and filename from the original path
        ext = '.png'
        if source_path:
            _, ext0 = osp.splitext(source_path)
            if ext0:
                ext = ext0

        if number is not None:
            filename = f'{number:0{digits}d}{ext}'
        else:
            # use the original filename if available
            if source_path:
                filename = osp.basename(source_path)
            else:
                filename = f'{self.frame_count[sequence_name]:05d}{ext}'

        save_path = osp.join(save_dir, filename)
        
        # save the image directly with cv2 instead of via vis_backend
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.bgr2rgb else img)
