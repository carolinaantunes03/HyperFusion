from train import *
from pl_wrap_obstetrics import  PlModelWrapEnsemble
from models.model_ensemble import ModelsEnsembleClassification
import os
import re
import torch
from argparse import ArgumentParser
import yaml

def main(config: EasyDict):
    # Optional: Wandb logger
    logger = wandb_interface(config)

    # Create the data module
    data_module_name = config.data_module.pop("data_module_name")
    data_module = globals()[data_module_name](config.data_module)
    data_module.setup()
    config.data_module_instance = data_module

    # Prepare task-specific config
    arrange_config4task(config)

    # Load model ensemble
    model_ensemble = load_model_ensemble(config)

    # Trainer setup
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.trainer.gpu,
        enable_checkpointing=False,
    )

    # Run evaluation
    pl_model = PlModelWrapEnsemble(
        model=model_ensemble,
        class_names=data_module.class_names
    )
    print(type(pl_model))
    trainer.test(pl_model, datamodule=data_module)
   
    
    

def load_model_ensemble(config: EasyDict):
    wrapper_class = globals()[config.lightning_wrapper.wrapper_name]
    model_ensemble = ModelsEnsembleClassification()

    base_dir = config.checkpointing.ckpt_dir
    print(f"Loading checkpoints from: {base_dir}")

    for fold in range(5):  # or config.data_module.k_folds
        checkpoint_path = os.path.join(
            base_dir,
            f"fold_{fold}",
            "test",
            "fold_0",
            "best_val.ckpt"
        )

        if os.path.exists(checkpoint_path):
            print(f"Loading fold {fold}: {checkpoint_path}")
            pl_model = wrapper_class.load_from_checkpoint(checkpoint_path)
            model_ensemble.append(pl_model.model)
        else:
            print(f"Missing checkpoint: {checkpoint_path}")

    print(f"\nTotal models loaded: {len(model_ensemble.models)}")
    return model_ensemble


def arrange_config4task(config: EasyDict):
    """Prepare config values for obstetrics evaluation"""
    config.lightning_wrapper.batch_size = config.data_module_instance.batch_size
    config.lightning_wrapper.class_names = getattr(config.data_module_instance, "class_names", ["class0", "class1"])
    # checkpoint callback not needed for evaluation
    config.checkpointing.CheckpointCallback = None
    config.checkpointing.callback_kwargs = dict()


def wandb_interface(config: EasyDict):
    """Optional: initialize Wandb logger"""
    wandb_args = config.wandb
    if wandb_args.enable:
        os.makedirs(wandb_args.logs_dir, exist_ok=True)
        logger = WandbLogger(
            project=wandb_args.project_name,
            name=config.experiment_name,
            save_dir=wandb_args.logs_dir
        )
    else:
        logger = False
    return logger


if __name__ == "__main__":
    default_cfg_path = os.path.join(
        os.getcwd(),
        "experiments/mode_delivery_prediction/default_train_config.yml"
    )

    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", default=default_cfg_path, type=str)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Load config
    assert os.path.exists(args.config_path), f"Config not found: {args.config_path}"
    with open(args.config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    # Debug mode adjustments
    if args.debug:
        config.data_module.num_workers = 0
        config.trainer.gpu = [1]
        config.wandb.enable = False

    main(config)