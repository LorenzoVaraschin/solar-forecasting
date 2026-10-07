from omegaconf import OmegaConf
from torchvision.transforms import v2
from data import SkyImageDataset
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

def get_batch(dataset, idxs_):
    """
    Create a dataset batch with the desired idxs_ samples
    """
    samples = [dataset[i] for i in idxs_]
    batch = default_collate(samples)
    return batch
    
def get_config():
    """
    Reads configs from a yaml file and terminal.
    """
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf
    
def create_dataloaders(config):
    """
    Create a train, test and val dataloader from a config file.
    """
    train_transforms, eval_transforms = list(), list()

    train_transforms.append(v2.Resize(config.dataset.image_size))
    eval_transforms.append(v2.Resize(config.dataset.image_size))

    if config.dataset.use_crop:
        train_transforms.append(v2.RandomResizedCrop(size=config.dataset.image_size, scale=(0.78, 1), ratio=(1, 1)))
    if config.dataset.use_hflip:
        train_transforms.append(v2.RandomHorizontalFlip(p=0.5))
    if config.dataset.use_normalization:
        train_transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        eval_transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transforms = v2.Compose(train_transforms)
    eval_transforms = v2.Compose(eval_transforms)
    
    train_dataset = SkyImageDataset(
        hdf5_path = config.dataset.hdf5_path,
        img_transform = train_transforms,
        mask_radius = config.dataset.mask_radius,
        image_size = config.dataset.image_size,
        timedelta = config.dataset.timedelta,
        return_dhi_dni = config.model.dhi_dni_estimation,
        split="train"
    )
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = config.dataset.batch_size,
        num_workers = 4,
        prefetch_factor = 2,
        pin_memory = True,
        shuffle = True
    )

    test_dataset = SkyImageDataset(
        hdf5_path = config.dataset.hdf5_path,
        img_transform = eval_transforms,
        mask_radius = config.dataset.mask_radius,
        image_size = config.dataset.image_size,
        timedelta = 0,
        return_dhi_dni = config.model.dhi_dni_estimation,
        split="test"
    )
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = config.dataset.batch_size,
        num_workers = 4,
        prefetch_factor = 2,
        pin_memory = True,
        shuffle = False
    )
    
    val_dataset = SkyImageDataset(
        hdf5_path = config.dataset.hdf5_path,
        img_transform = eval_transforms,
        mask_radius = config.dataset.mask_radius,
        image_size = config.dataset.image_size,
        timedelta = config.dataset.timedelta,
        return_dhi_dni = config.model.dhi_dni_estimation,
        split="val"
    )
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = config.dataset.batch_size,
        num_workers = 4,
        prefetch_factor = 2,
        pin_memory = True,
        shuffle = False
    )

    return {
        "train": train_dataloader,
        "test": test_dataloader,
        "val": val_dataloader,
    }