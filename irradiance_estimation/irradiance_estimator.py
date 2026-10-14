import lightning as L
import inspect
from torch import nn
import backbones
import torch
from torchmetrics import MetricCollection
import torchmetrics
from utils import get_batch
from PIL import ImageDraw
import torchvision.transforms.v2.functional as F
import numpy as np
import torchvision
from torchvision.transforms import v2
from omegaconf import OmegaConf
import math
import h5py
import os

class IrradianceEstimator(L.LightningModule):
    def __init__(self, config):
        super().__init__()    
        self.save_hyperparameters(config)
        config = OmegaConf.create(config)
        self.config = config
        self.uncertainty_type = config.model.uncertainty.type
        
        #================= create a dict containing all possible backbones =================
        backbones_dict = {model_name: model_class for model_name, model_class in inspect.getmembers(backbones, inspect.isclass)}
        if config.model.backbone not in backbones_dict.keys():
            raise ValueError(f"Invalid model backbone '{config.model.backbone}'. Must be one of {list(backbones_dict.keys())}")       
        self.backbone = backbones_dict[config.model.backbone](config)
        
        if config.model.uncertainty.head_only.flag == True:
            self._load_and_freeze_weights(layer_="backbone")
        
        #================= create irradiance heads =================
        num_outs = 1
        if config.model.uncertainty.type == "continuous_gaussian":
            num_outs += 1 # variance of gaussian distribution
        elif "histloss" in config.model.uncertainty.type:
            num_outs = config.model.uncertainty.nbins
            self._init_histloss_params()

        if (config.model.uncertainty.type == "continuous_gaussian") and (config.model.uncertainty.head_only.flag == True):
            self.irr_head = backbones.ParallelLinearHead(in_feat=self.backbone._get_final_layer_shape())
            self._load_and_freeze_weights(layer_="head")
        else:
            self.irr_head = nn.Linear(in_features=self.backbone._get_final_layer_shape(), out_features=num_outs)
            
        #================= create model metrics =================
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
                
        base_metrics = {
            "rmse": torchmetrics.regression.MeanSquaredError(squared=False),
            "mae": torchmetrics.regression.MeanAbsoluteError()
        } 
        collection = MetricCollection(base_metrics)
        
        self.train_metrics["ghi"] = collection.clone(prefix=f"train/ghi_")
        self.val_metrics["ghi"] = collection.clone(prefix=f"val/ghi_")
        self.test_metrics["ghi"] = collection.clone(prefix=f"test/ghi_")
        
        #================= create loss functions =================
        if config.model.uncertainty.type == "continuous_gaussian":
            self.criterion = torch.nn.GaussianNLLLoss()
        elif  "histloss" in config.model.uncertainty.type:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif config.model.uncertainty.type == "none":
            self.criterion = torch.nn.MSELoss()

        #================= save all val GHI preds and trues to log largest errors samples in the final epoch =================
        self.all_val_preds = list()
        self.all_val_trues = list()

    def forward(self, x):
        img_embeds = self.backbone(x)
        irr_preds = self.irr_head(img_embeds)
        return irr_preds
        
    def training_step(self, batch, batch_idx):
        imgs = batch["image"]

        outs_ = self(imgs)

        total_loss = self._log_metrics(split_name="train", metrics_=self.train_metrics, outs_=outs_, batch=batch)

        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs = batch["image"]
        
        outs_ = self(imgs)

        self._log_metrics(split_name="validation", metrics_=self.val_metrics, outs_=outs_, batch=batch)
                    
    def test_step(self, batch, batch_idx):
        imgs = batch["image"]

        outs_ = self(imgs)

        self._log_metrics(split_name="test", metrics_=self.test_metrics, outs_=outs_, batch=batch)

    def on_validation_epoch_end(self):
        # log images with 10 largest GHI estimation errors
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            # convert list to numpy array
            self.all_val_preds = np.array(self.all_val_preds)
            self.all_val_trues = np.array(self.all_val_trues)

            # sort errors in descending order
            self.sorted_errors = np.argsort(np.abs(self.all_val_preds-self.all_val_trues))[::-1]

            # create a batch with 10 largest errors
            batch = get_batch(self.trainer.val_dataloaders.dataset, self.sorted_errors[:10])

            # invert normalization for visualization purposes
            if self.config.dataset.use_normalization:
                inv_normalize = v2.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                batch["image"] = inv_normalize(batch["image"])

            # annotate timestamps onto each image
            timestamps = [str(ts_.item())[:8]+"_"+str(ts_.item())[8:] for ts_ in batch["timestamp"]]
            annotated_imgs = list()
            for i, img in enumerate(batch["image"]):
                pil_img = F.to_pil_image(img)
                draw = ImageDraw.Draw(pil_img)
                draw.text(xy=(0, 0), text=f"{timestamps[i]}\nTrue: {self.all_val_trues[self.sorted_errors[i]]:.2f}", fill=(255, 255, 255)) 
                draw.text(xy=(0, 115), text=f"Pred: {self.all_val_preds[self.sorted_errors[i]]:.2f}", fill=(255, 255, 255)) 
                annotated_imgs.append(F.to_image(pil_img))
            annotated_imgs = torch.stack(annotated_imgs)
                
            imgs = torchvision.utils.make_grid(annotated_imgs, nrow=5)
            self.trainer.logger.log_image(key="largest_errors", images=[imgs])

    def configure_optimizers(self):
        optimizer_name = self.config.training.optimizer
        learning_rate = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        momentum = self.config.training.momentum
        lr_scheduler = self.config.training.lr_scheduler
        total_steps = self.trainer.estimated_stepping_batches

        # exclude normalization and bias parameters from weight decay application
        exclude = (lambda n, p: p.ndim < 2 or "layer_norm" in n or "bias" in n or 'norm' in n)
        include = lambda n, p: not exclude(n, p)
        named_parameters = list(self.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        
        # get optimizer
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                [
                    {"params": gain_or_bias_params, "weight_decay": weight_decay},
                    {"params": rest_params, "weight_decay": weight_decay},
                ],
                lr=learning_rate,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": weight_decay},
                    {"params": rest_params, "weight_decay": weight_decay},
                ],
                lr=learning_rate,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                [
                    {"params": gain_or_bias_params, "weight_decay": weight_decay},
                    {"params": rest_params, "weight_decay": weight_decay},
                ],
                lr=learning_rate,
                momentum=momentum
            )

        # get learning rate scheduler
        if lr_scheduler == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                final_div_factor=1e4
            )
        elif lr_scheduler == "cos_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_steps, eta_min=learning_rate/10)
        elif lr_scheduler == "constant":
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def _init_histloss_params(self):
        """
        Function to initialize the bin border, center and sigma parameters for the histogram loss. 
        Adapted from https://github.com/marthawhite/Histogram_loss
        """
        # get min and max train values for kt target
        hdf5_path = os.path.join(os.path.expanduser("~"), self.config.dataset.hdf5_path)
        with h5py.File(hdf5_path, 'r') as f:
            sample_idx = f["train"]["sample_idxs"][:]
            ghi = f["train"]["GHI/anchor"][sample_idx]
            ghi_cs = f["train"]["GHI/clear_sky"][sample_idx]
            all_kt = ghi / ghi_cs
        if self.config.training.target == "kt":
            all_irr = torch.tensor(all_kt)
        elif self.config.training.target == "ghi":
            all_irr = torch.tensor(ghi)

        self.nbins = self.config.model.uncertainty.nbins
        # following the HLoss paper formulas
        if self.uncertainty_type == "histloss_gauss":
            sigma_ratio = self.config.model.uncertainty.sigma_ratio
            pad_ratio = self.config.model.uncertainty.pad_ratio

            bin_width = (all_irr.max() - all_irr.min()) / (self.nbins - 2*sigma_ratio*pad_ratio)
            
            
            sigma = sigma_ratio * bin_width
            self.register_buffer("sigma", sigma)
            
            self.pad = pad_ratio * self.sigma

            borders = torch.linspace(all_irr.min()-self.pad, all_irr.max()+self.pad, self.nbins+1).unsqueeze(1)
        else:
            nbins_pad = self.config.model.uncertainty.nbins_pad
            bin_width = (all_irr.max() - all_irr.min()) / (self.nbins - 2*nbins_pad)
            
            self.pad = nbins_pad * bin_width
            borders = torch.linspace(all_irr.min()-self.pad, all_irr.max()+self.pad, self.nbins+1).unsqueeze(1)

            hist_start = torch.min(borders)
            self.register_buffer("hist_start", hist_start)
            if self.uncertainty_type == "histloss_uniform":
                self.eps = self.config.model.uncertainty.sigma_ratio
                self.scale = 1 - self.eps * self.nbins

        self.register_buffer("bin_width", bin_width)
        self.register_buffer("borders", borders)
        
        centers = (self.borders[:-1] + self.borders[1:]) / 2
        self.register_buffer("centers", centers)
        if self.uncertainty_type == "histloss_projected":
            self.register_buffer("centers_low", torch.min(centers))


    def transform_target(self, scalar_target):
        """
        Function to transform scalar targets into a binned probability vector. This probability can be of the following forms:
        -hl_gaussian
        -hl_onebin
        -hl_uniform

        Args:
            - scalar_target: the scalar target to transform of shape (B,)

        Returns:
            - vector_target: the probability vector of each scalar target. Shape (B, nbins)

        Adapted from: https://github.com/marthawhite/Histogram_loss
        """

        if self.uncertainty_type == "histloss_gauss":
            border_targets = torch.erf((self.borders - scalar_target) / (math.sqrt(2.0) * self.sigma))
            two_z = border_targets[-1] - border_targets[0] 
            vector_target = (border_targets[1:] - border_targets[:-1]) / two_z
            vector_target = vector_target.transpose(0, 1)
        elif "projected" not in self.uncertainty_type:
            hist_position = (scalar_target-self.hist_start) / self.bin_width
            vector_target = torch.nn.functional.one_hot(hist_position.long(), num_classes=self.nbins).float()
            if self.uncertainty_type == "histloss_uniform":
                vector_target = vector_target * self.scale + self.eps
        elif self.uncertainty_type == "histloss_projected":
            # hl-projected
            vector_target = torch.zeros((self.nbins, scalar_target.shape[0]))
            idx = ((scalar_target - self.centers_low) / self.bin_width).int() 
            
            p = (scalar_target - self.centers[:, 0][idx]) / self.bin_width
            
            # scatter_add_ is non deterministic on the GPU
            idx, p = idx.cpu(), p.cpu()
            vector_target.scatter_add_(0, idx.view(1, -1)  , 1-p.view(1, -1))
            vector_target.scatter_add_(0, idx.view(1, -1)+1,   p.view(1, -1))
            vector_target = vector_target.to(self.device).permute(1, 0)

        return vector_target

    def get_preds(self, outs_, batch):
        total_loss = 0
        
        if self.uncertainty_type == "none":
            preds = outs_[:, 0]
            if self.config.training.target == "kt":
                target = batch[f"ghi_kt_solis"]
                irr_preds = preds * batch[f"ghi_solis"]
            elif self.config.training.target == "ghi":
                target = batch[f"ghi"]
                irr_preds = preds
            loss = self.criterion(preds, target)
            total_loss += loss
            target_preds = preds
            return {
                "total_loss": total_loss,
                "target_preds": target_preds,
                "irr_preds": irr_preds
            }
        elif self.uncertainty_type == "continuous_gaussian":
            preds_mean = outs_[:, 0]
            preds_var = torch.exp(outs_[:, 1])
            if self.config.training.target == "kt":
                target = batch[f"ghi_kt_solis"]
                irr_preds = preds_mean * batch[f"ghi_solis"]
            elif self.config.training.target == "ghi":
                target = batch[f"ghi"]
                irr_preds = preds_mean
            loss = self.criterion(preds_mean, target, preds_var)
            total_loss += loss
            target_preds = preds_mean
            return {
                "total_loss": total_loss,
                "target_preds": target_preds,
                "irr_preds": irr_preds,
                "target_var": preds_var
            }
        elif "histloss" in self.uncertainty_type:
            hist_preds = outs_
            hist_probs = torch.nn.functional.softmax(hist_preds, dim=-1)
            preds_mean = torch.matmul(hist_probs, self.centers).view(-1)
            if self.config.training.target == "kt":
                target = self.transform_target(batch[f"ghi_kt_solis"])
                irr_preds = preds_mean * batch[f"ghi_solis"]
            elif self.config.training.target == "ghi":
                target = self.transform_target(batch[f"ghi"])
                irr_preds = preds_mean
            loss = self.criterion(hist_preds, target)
            total_loss += loss
            target_preds = preds_mean
            return {
                "total_loss": total_loss,
                "target_preds": target_preds,
                "irr_preds": irr_preds,
                "hist_probs": hist_probs
            }
            
    def _log_metrics(self, split_name, metrics_, outs_, batch):
        if split_name == "train":
            prog_bar = True
            on_step = True
        else:
            prog_bar = False
            on_step = False
            
        preds_ = self.get_preds(outs_, batch)
        total_loss, irr_preds = preds_["total_loss"], preds_["irr_preds"]
        
        irr_target = batch["ghi"]
        metrics_["ghi"].update(irr_preds, irr_target)
        self.log_dict(metrics_["ghi"], on_step=on_step, on_epoch=True, prog_bar=prog_bar, logger=True)
        self.log(f"{split_name}/loss", total_loss.detach(), on_step=on_step, on_epoch=True, prog_bar=prog_bar, logger=True)

        if split_name == "train":
            return total_loss
        elif (split_name == "validation") and (self.trainer.current_epoch == self.trainer.max_epochs - 1):
            self.all_val_preds.extend(irr_preds.cpu().numpy().tolist())
            self.all_val_trues.extend(irr_target.cpu().numpy().tolist())

    def _load_and_freeze_weights(self, layer_):
        """
        Load pretrained weights of the desired layer, if available, and freeze them. This is used when training a seperate head to predict the uncertainty. The user should make sure to check if the preprocessing/hyperparemters of the saved checkpoints are consistent with the current model.
        """
        
        assert self.config.model.uncertainty.head_only.flag == True, "Backbone weights should only be loaded and frozen when training the uncertainty head separately"
        
        # Get model backbone checkpoint path
        pretrained_ckpt_path = os.path.join(os.getcwd(), self.config.model.uncertainty.head_only.pretrained_ckpt_path)
        
        # Sanity check prints
        print("\n\n")
        print(f"LOADING THE CHECKPOINT FROM THE FOLLOWING MODEL {os.path.basename(pretrained_ckpt_path)}")
        print(f"CURRENT UNCERTAINTY IS SET TO {self.config.model.uncertainty.type}")
        print("\n\n")

        if os.path.exists(pretrained_ckpt_path):
            model_state_dict = torch.load(pretrained_ckpt_path, map_location=self.device)["state_dict"]

            # Load only the backbone weights, head will be trained from scratch. If using gaussian NLL loss, also freeze the part of the head that computes the mean (layer_=="head")
            if layer_ == "backbone":
                backbone_weights = {
                    k.removeprefix("backbone."): v
                    for k, v in model_state_dict.items()
                    if k.startswith("backbone")
                }
                self.backbone.load_state_dict(backbone_weights)
                for param_ in self.backbone.parameters():
                    param_.requires_grad = False
                self.backbone.eval()
            elif layer_ == "head":
                # Load only mean head state dict when training with guassian NLL
                head_weights = {
                    k.replace("irr_head.", "mean_head."): v[[0]]
                    for k, v in model_state_dict.items()
                    if k.startswith("irr_head")
                }
                self.irr_head.load_state_dict(head_weights, strict=False)
                for name_, param_ in self.irr_head.named_parameters():
                    if "mean_head" in name_:
                        param_.requires_grad = False
        else:
            raise RuntimeError(f"Unable to find checkpoints path {pretrained_ckpt_path}")