'''
Copyright (C) 2021  Shiavm Pandey
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from roifile import ImagejRoi
from torch.nn.functional import interpolate
from torchvision.datasets import VisionDataset


class AikfmDataset(VisionDataset):

    def __init__(self, root: str,
                 sample : str = None,
                 keep_positive : bool = True,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 img_size: Tuple[int, int] = (600, 600)) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.n_samples = 7
        self.target_samples = ["Sample"+str(x) for x in range(1, self.n_samples+1)]
        if sample is not None:
            assert(sample in self.target_samples)
            self.target_samples = [sample]

        self.train_data = None
        self.test_data = None
        self.keep_positive = keep_positive
        self.img_size = img_size
        self.sample_data_ : Dict[str, List[List[int]]] = {}
        self.img_samples_ : List[str] = []
        data_path_suffix_ = "Training/Training"
        self.train_data_path = os.path.join(self.root, data_path_suffix_)

        self.get_data_info()

    def get_data_info(self) -> Any:
        self.train_data = []
        for sample in self.target_samples:
            sample_folder_path = os.path.join(self.train_data_path, sample)

            sample_images_ = [sample + "/images/" + x[:-4] for x in \
                os.listdir(os.path.join(sample_folder_path, "images"))]
            self.img_samples_ += sample_images_

            sample_mask_path = os.path.join(sample_folder_path, "masks")
            mask_items_list = os.listdir(sample_mask_path)

            self.sample_data_.update({x: [] for x in sample_images_})

            for _path in mask_items_list:
                if _path.endswith('.roi'):
                    roi_file_path = os.path.join(sample_mask_path, _path)
                    roi = ImagejRoi.fromfile(roi_file_path)
                    coords = [roi.left, roi.right, roi.top, roi.bottom]
                    self.sample_data_[sample + "/images/" + _path[:-4]] += [coords]
                else:
                    for x in os.listdir(os.path.join(sample_mask_path, _path)):
                        __path = os.path.join(_path, x)
                        roi_file_path = os.path.join(sample_mask_path, __path)
                        roi = ImagejRoi.fromfile(roi_file_path)
                        coords = [roi.left, roi.right, roi.top, roi.bottom]
                        self.sample_data_[sample + "/images/" + _path] += [coords]

            img_samples__ = []
            for key, value in self.sample_data_.items():
                if (value.__len__()):
                    img_samples__ += [key]

            self.img_samples_ = img_samples__

    def __getitem__(self,
                    idx: int) -> Any:
        name_ = self.img_samples_[idx]
        img_path = os.path.join(self.train_data_path, name_ + '.png')
        img = np.asarray(Image.open(img_path), dtype=np.float32)/255.
        img = torch.as_tensor(img, dtype=torch.float32)

        img_mask = torch.zeros(img.shape[0], img.shape[1], 1, dtype=torch.uint8)
        for box in self.sample_data_[name_]:
            img_mask[box[2]:box[3], box[0]:box[1]] = 1.

        img_mask = torch.permute(img_mask, dims=(2, 0, 1))
        img = torch.permute(img, dims=(2, 0, 1))

        target = img_mask

        img = interpolate(img.unsqueeze(dim=0), self.img_size, mode='bilinear').squeeze(dim=0)
        target = interpolate(target.unsqueeze(dim=0), self.img_size, mode='bilinear').squeeze(dim=0)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self) -> int:
        return self.img_samples_.__len__()

class AikfmDatasetInference(VisionDataset):

    def __init__(self, root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transforms, transform, target_transform)

        self._test_dir = os.path.join(self.root, 'Test/Test/images')
        self.images : List[str] = [os.path.join(self._test_dir, img_name) for img_name in os.listdir(self._test_dir)]

    def __getitem__(self, index: int) -> Any:
        img_path = self.images[index]

        img = np.asarray(Image.open(img_path), dtype=np.float32)/255.
        img = torch.as_tensor(img, dtype=torch.float32)

        target = img_path[index].split('/')[-1][:-4]

        img = img.permute(dims=(2, 0, 1))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self) -> int:
        return self.images.__len__()

if __name__ == "__main__":
    # Run tests here
    ds = AikfmDataset("~/DKLabs/AI-KFM/AI-KFM/data")
    n = ds.__len__()
    img, target = ds.__getitem__(5)

    print(f'Number of images : {n}, \nImage shape : {img.shape}, \ntarget bboxes : {target["bboxs"]}, \n target mask shape: {target["mask"].shape}')
