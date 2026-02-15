import hydra
import torch
import wandb
import yaml


@hydra.main(config_path="configs/training", config_name="taxpose_vtamp")
def setup(cfg):
    trainer, model, dm = setup_main(cfg)


def main():
    with open("train_config.yaml", "r") as config:
        args = yaml.load(config, Loader=yaml.FullLoader)

    if args["enable_wandb"]:
        wandb.init()
        # !!! We need to look into wandb logging but only after we are done with the full training pipeline

    # Pre-processing for training:
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    main()
