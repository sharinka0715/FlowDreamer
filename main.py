import os
import json
import tyro
import torch
import lightning
from pprint import pprint
from typing import Literal
from dataclasses import dataclass
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from diffusers import AutoencoderKL, PNDMScheduler
from diffusers.models.model_loading_utils import load_state_dict
from diffusers.utils.import_utils import is_xformers_available

from depth_anything_v2.dpt import DepthAnythingV2
from dataset.single_view_sequential_dataset import SingleViewSequentialDataset
from models.unet_2d_condition_custom import UNet2DConditionCustomModel
from models.cond_unet_2d import ConditionalUNet2D
from models.flow_diffusion_module import FlowDiffusionModule


@dataclass
class Args:
    """FlowDreamer training and inference pipeline arguments."""

    dataset_dir: str  # The dataset directory
    pretrained_path: str  # The path to the pretrained Stable Diffusion model that will be loaded
    depth_est_path: str  # The path to the depth estimation model that will be loaded
    ckpt_path: str | None = None  # The path to the FlowDreamer model that will be loaded
    log_path: str = "./logs"  # The path of the local logging directory
    evaluation_output_path: str = "./evaluation_output"  # The path to the evaluation output (videos)
    exp_name: str = "debug"  # TensorBoard experiment name

    debug: bool = False  # Whether to run in debug mode (only for training)
    evaluate: bool = False  # Whether to run evaluation instead of training

    precision: Literal["32", "16", "16-mixed", "bf16", "bf16-mixed"] = "bf16-mixed"  # The training precision
    seed: int = 42  # The seed of the program
    batch_size: int = 16  # The training batch size (in timesteps) on each device
    num_workers: int = 16  # The number of workers in dataloader
    lr: float = 1e-4  # The learning rate
    flow_scale: float = 100.0  # The scaling factor for the flow for better training stability
    eval_length: int = (
        15  # The maximum length of the evaluation trajectory (can be set to very large for long horizon evaluation)
    )
    training_data_amount: int = (
        7680000  # The total data amount used during training, which will be converted to steps based on the batch size and the world size
    )


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    pprint(args)

    if args.precision != "32":
        torch.set_float32_matmul_precision("high")

    seed_everything(args.seed)

    if not args.evaluate:
        train_dataset = SingleViewSequentialDataset(
            args.dataset_dir,
            split="train",
            seq_len=1,
            mode="discard",
        )
        val_dataset = SingleViewSequentialDataset(
            args.dataset_dir,
            split="val",
            seq_len=15,
            mode="padding",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=SingleViewSequentialDataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=SingleViewSequentialDataset.collate_fn,
        )
        sample = val_dataset[0]
    else:
        test_dataset = SingleViewSequentialDataset(
            args.dataset_dir,
            split="test",
            seq_len=args.eval_length,
            mode="padding",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=SingleViewSequentialDataset.collate_fn,
        )
        sample = test_dataset[0]

    action_dim = sample["action"].shape[1]
    height = sample["rgbs"].shape[2]
    width = sample["rgbs"].shape[3]

    # FlowNet at stage 1, input batch, output flow image
    flownet = ConditionalUNet2D(in_channels=4, out_channels=3, base_channels=128, action_dim=action_dim, num_heads=8)
    flownet.train()

    # Scheduler and Tokenizer are configs, no trainable parameters
    noise_scheduler = PNDMScheduler.from_pretrained(args.pretrained_path, subfolder="scheduler")

    # Load model
    vae = AutoencoderKL.from_pretrained(args.pretrained_path, subfolder="vae")
    with open(os.path.join(args.pretrained_path, "unet", "config.json"), "r") as fp:
        unet_config = json.load(fp)
    unet_config["in_channels"] = 8
    unet_config["encoder_hid_dim_type"] = "text_proj"
    unet_config["encoder_hid_dim"] = action_dim
    unet_config["extra_in_channels"] = 4
    unet = UNet2DConditionCustomModel.from_config(unet_config)

    state_dict = load_state_dict(os.path.join(args.pretrained_path, "unet", "diffusion_pytorch_model.safetensors"))
    del state_dict["conv_in.weight"]
    del state_dict["conv_in.bias"]
    unet.load_state_dict(state_dict, strict=False)

    depth_est = DepthAnythingV2(
        **{"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": 6}
    )
    depth_est.load_state_dict(torch.load(args.depth_est_path, map_location="cpu", weights_only=True))
    depth_est.eval()
    depth_est.requires_grad_(False)

    # Set gradient and training mode. No need to set EMA.
    vae.requires_grad_(False)
    unet.train()

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    evaluation_output_path = os.path.join(args.evaluation_output_path, args.exp_name)

    if args.evaluate:
        model = FlowDiffusionModule.load_from_checkpoint(
            args.ckpt_path,
            map_location="cpu",
            strict=False,
            scheduler=noise_scheduler,
            vae=vae,
            unet=unet,
            flownet=flownet,
            depth_est=depth_est,
            height=height,
            width=width,
            flow_scale=args.flow_scale,
            evaluation_output_path=evaluation_output_path,
        )
        trainer = lightning.Trainer(
            logger=False,
            precision=args.precision,
            accelerator="gpu",
            strategy="auto",
        )
        trainer.test(model=model, dataloaders=test_loader)
    else:
        model = FlowDiffusionModule(
            noise_scheduler,
            vae,
            unet,
            flownet,
            depth_est,
            height=height,
            width=width,
            lr=args.lr,
            flow_scale=args.flow_scale,
            evaluation_output_path=evaluation_output_path,
        )
        # PyTorch Lightning training setting
        os.makedirs(args.log_path, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=args.log_path,
            name=args.exp_name,
        )
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        max_steps = args.training_data_amount // (args.batch_size * world_size)
        checkpoint_callback = ModelCheckpoint(every_n_train_steps=max_steps // 4, save_last=True)

        print(f"Training {max_steps} steps on {world_size} devices and batch size {args.batch_size}.")

        trainer = lightning.Trainer(
            limit_val_batches=16,
            check_val_every_n_epoch=1,
            max_steps=max_steps,
            logger=logger,
            callbacks=[checkpoint_callback],
            precision=args.precision,
            accelerator="gpu",
            strategy="ddp" if not args.debug else "auto",
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.ckpt_path,
        )
