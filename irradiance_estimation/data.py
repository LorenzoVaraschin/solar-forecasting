import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import os

class SkyImageDataset(Dataset):
    def __init__(self, hdf5_path, img_transform, mask_radius, image_size, timedelta, return_dhi_dni=False, split="train"):
        """
        This dataset is used for training models with sky images as input, of shape CxHxW.

        Args:
            - hdf5_path: path to the hdf5 dataset. This dataset should contains the following values:
                -images: uint8 128x128 images (unmasked) in the range 0-255
                -irradiance (GHI/DNI/DHI): there are 4 different values for each irradiance type. The PRE value is the value in the prior
                 minute. The ANCHOR value is the value in the current minute. The POST value is the value in the next minute. The CLEAR SKY
                 value is the clear sky value. The PRE/ANCHOR/POST values are used to easily apply a timedelta to the irradiance
                 measurements.
                -zenith angle: the zenith angle, used to compute GHI from DHI and DNI.
                -timestamp: the timestamp of each image for reference.
                -sample idxs: !!!IMPORTANT!!! the indexes in this dataset are the actual indexes used for the video/image related projects
                 and correspond to the anchor indexes. Some samples at beggining/end of day cannot be used as anchors, due to minimum video
                 length, therefore len(sample_idxs) < len(images).
           - img_transform: torchvision transform to apply to the images.
           - mask_radius: the radius of the circular mask to apply to each image. If 0, then no mask is applied.
           - image_size: the size of the images. Must be the same size as the size of the hdf5 dataset.
           - timedelta: int value to use as timedelta on the irradiance measurements
           - return_dhi_dni: when True, returns DHI and DNI to use for model training, otherwise return only GHI
           - split: the train/test/val split of the dataset.
        """

        self.hdf5_path = os.path.join(os.path.expanduser("~"), hdf5_path)
        self.img_transform = img_transform
        self.timedelta = timedelta
        self.return_dhi_dni = return_dhi_dni
        self.split = split
        self.hdf5_file = None
        
        if mask_radius != 0:
            self.image_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            self.image_mask = cv2.circle(self.image_mask, (image_size//2, image_size//2), mask_radius, (1, 1, 1), -1)
            self.image_mask = torch.tensor(self.image_mask.astype(np.float32)).permute(2, 0, 1) 
        else:
            self.image_mask = None
            
        # Open temporarily just to get the total number of samples
        with h5py.File(self.hdf5_path, 'r') as f:
            self.dataset_length = len(f[self.split]['sample_idxs'])
            assert image_size == f.attrs['image_size'], f"Image size mismatch between the provided hdf5 file path {f.attrs['image_size']} and provided {image_size}"

    def apply_timedelta(self, irradiance, sample_idx):
        # Get the current valued irradiance
        anchor_irr = self.group[f'{irradiance}/anchor'][sample_idx]
        
        if self.timedelta > 0:
            # Get the previous valued irradiance
            pre_irr = self.group[f'{irradiance}/pre'][sample_idx]
            
            # Get the irradiance value for every second (0:00 - 1:00)
            minute_irr = torch.linspace(pre_irr, anchor_irr, 61)

            # Return the timedelta value
            return minute_irr[-self.timedelta-1]
        elif self.timedelta < 0:
            # Get the posterior valued irradiance
            post_irr = self.group[f'{irradiance}/post'][sample_idx]

            # Get the irradiance value for every second (1:00 - 2:00)
            minute_irr = torch.linspace(anchor_irr, post_irr, 61)

            # Return the timedelta value
            return minute_irr[-self.timedelta]

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True, rdcc_nbytes=500 * (1024**2))
            # Create a shortcut to the specific split group (e.g., /train)
            self.group = self.hdf5_file[self.split]

        # Get the sample idx
        sample_idx = self.group['sample_idxs'][idx]
        
        # Get the corresponding image
        image = self.group['images'][sample_idx]
        
        # Convert to torch.float32 and normalize to [0, 1]
        image = torch.tensor(image).float() / 255.0
        
        # Convert (H, W, C) -> (C, H, W)
        image = image.permute(2, 0, 1)

        # Apply mask, if needed
        if isinstance(self.image_mask, torch.Tensor):
            image = image * self.image_mask

        # Apply image transforms
        image = self.img_transform(image)

        # Get the irradiance values and apply timedelta, if needed
        if self.timedelta != 0:
            ghi = self.apply_timedelta('GHI', sample_idx)

            if self.return_dhi_dni:
                dhi = self.apply_timedelta('DHI', sample_idx)
                dni = self.apply_timedelta('DNI', sample_idx)
        else:
            ghi = torch.tensor(self.group['GHI/anchor'][sample_idx])

            if self.return_dhi_dni:
                dhi = torch.tensor(self.group['DHI/anchor'][sample_idx])
                dni = torch.tensor(self.group['DNI/anchor'][sample_idx])

        # Get clear sky irradiance values
        ghi_solis = torch.tensor(self.group['GHI/clear_sky'][sample_idx])
        ghi_kt_solis = ghi / ghi_solis
        
        if self.return_dhi_dni:
            dhi_solis = torch.tensor(self.group['DHI/clear_sky'][sample_idx])
            dhi_kt_solis = dhi / dhi_solis
            dni_solis = torch.tensor(self.group['DNI/clear_sky'][sample_idx])
            dni_kt_solis = dni / dni_solis

        # get timestamp
        timestamp = torch.tensor(int(self.group['timestamp'][sample_idx].decode('utf-8')))
        if self.return_dhi_dni:
            # Get zenith angle to compute GHI from DHI and DNI
            zenith_angle = self.group['zenith_angle'][sample_idx]
            return{
                "image": image,                  # Shape: (3, 128, 128)
                "ghi": ghi,                      # Shape: ()
                "ghi_solis": ghi_solis,          # Shape: ()
                "ghi_kt_solis": ghi_kt_solis,    # Shape: ()
                "dhi": dhi,                      # Shape: ()
                "dhi_solis": dhi_solis,          # Shape: ()
                "dhi_kt_solis": dhi_kt_solis,    # Shape: ()
                "dni": dni,                      # Shape: ()
                "dni_solis": dni_solis,          # Shape: ()
                "dni_kt_solis": dni_kt_solis,    # Shape: ()
                "zenith_angle": zenith_angle,    # Shape: ()
                "timestamp": timestamp,          # Shape: ()
            }
        else:
            return {
                "image": image,                 # Shape: (3, 128, 128)
                "ghi": ghi,                     # Shape: ()
                "ghi_solis": ghi_solis,         # Shape: ()
                "ghi_kt_solis": ghi_kt_solis,   # Shape: ()
                "timestamp": timestamp,         # Shape: ()
            }