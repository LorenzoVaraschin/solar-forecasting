from utils import get_config, create_dataloaders
import lightning as L
from irradiance_estimator import IrradianceEstimator
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import wandb
from omegaconf import OmegaConf

def main():
    for seed_ in [0, 1, 2]:
        # read training configuration
        config = get_config()
        config.training.random_seed = seed_
        
        # set the seed for reproducibility
        L.seed_everything(config.training.random_seed)
        
        # create dataloaders
        dataloaders_ = create_dataloaders(config)
        
        # create model
        model = IrradianceEstimator(OmegaConf.to_container(config, resolve=True))
        
        # create callbacks (model checkpoint | lr monitor)
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                dirpath=os.path.join(os.getcwd(), "model_checkpoints", config.experiment.group_name, config.experiment.job_type if config.experiment.job_type is not None else ""),
                filename=config.experiment.name,
                monitor=config.experiment.ckpt_monitor
            )
        ]
        
        # create model logger
        model_logger = WandbLogger(
            project=config.experiment.project,
            name=config.experiment.name,
            group=config.experiment.group_name,
            job_type=config.experiment.job_type
        )

        # create trainer
        trainer = L.Trainer(
            accelerator=config.training.accelerator,
            max_epochs=config.training.epochs,
            logger=model_logger,
            callbacks=callbacks
        )
        
        # log code and config files
        trainer.logger.experiment.log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
        
        trainer.fit(model=model, train_dataloaders=dataloaders_["train"], val_dataloaders=dataloaders_["val"])
        trainer.test(dataloaders=dataloaders_["test"])
        wandb.finish()

if __name__ == "__main__":
    main()