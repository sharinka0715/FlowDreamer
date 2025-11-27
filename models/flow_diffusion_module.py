import os
import torch
import numpy as np
import piqa
from PIL import Image
import lightning
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from diffusers import AutoencoderKL, PNDMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.optimization import get_scheduler

from depth_anything_v2.dpt import DepthAnythingV2
from models.unet_2d_condition_custom import UNet2DConditionCustomModel
from models.cond_unet_2d import ConditionalUNet2D
from utils.visualization import flow_auto_to_image


class FlowDiffusionModule(lightning.LightningModule):
    def __init__(
        self,
        scheduler: PNDMScheduler,
        vae: AutoencoderKL,
        unet: UNet2DConditionCustomModel,
        flownet: ConditionalUNet2D,
        depth_est: DepthAnythingV2,
        height,
        width,
        evaluation_output_path: str,
        lr=1e-4,
        flow_scale=10.0,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.vae = vae
        self.unet = unet
        self.lr = lr

        self.flownet = flownet
        self.depth_est = depth_est
        self.flow_scale = flow_scale
        self.evaluation_output_path = evaluation_output_path

        self.h = height
        self.w = width

        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.h, self.w)),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def flownet_inference(self, batch, batch_idx):
        batch_size, seq_len = batch["action"].shape[0:2]

        flow_loss = 0.0
        flows_pred = []
        flows_gt = []

        for t in range(seq_len):
            latents = self.vae.encode(self.transforms(batch["rgbs"][:, t])).latent_dist.sample()
            rgbs = self.vae.decode(latents, return_dict=False)[0]
            current_imgs = torch.cat([rgbs, batch["depths"][:, t].unsqueeze(1)], dim=1).contiguous()
            # [B, 4, H, W], [B, D] -> [B, 3, H, W]
            flow_pred = self.flownet(current_imgs, batch["action"][:, t])
            gt_flows = batch["flows"][:, t]

            flow_loss += torch.nn.functional.mse_loss(flow_pred, gt_flows * self.flow_scale)
            flows_pred.append(flow_pred)
            flows_gt.append(gt_flows)

        flow_loss /= seq_len
        # [B, T, 3, H, W]
        flows_pred = torch.stack(flows_pred, dim=1)
        flows_gt = torch.stack(flows_gt, dim=1)

        return flows_pred, flow_loss, flows_gt

    def depth_estimation(self, rgbs=None, latents_now=None):
        """
        Depth Anything V2 only receives numpy images as input, so there might be loss of efficiency.
        """
        if rgbs is None:
            rgbs = self.vae.decode(latents_now / self.vae.config.scaling_factor, return_dict=False)[0]
        batch_size = rgbs.shape[0]
        loadimages = []
        for i in range(batch_size):
            loadimages.append(
                torch.clamp((rgbs[i] + 1.0) / 2.0 * 255, 0, 255).byte().permute(1, 2, 0).flip(-1).cpu().numpy()
            )
        preds = self.depth_est.infer_image_batch(loadimages, input_size=518)
        pred_depths = []
        for i in range(batch_size):
            resized_pred = Image.fromarray(preds[i]).resize((rgbs.shape[-1], rgbs.shape[-2]), Image.NEAREST)
            pred_depths.append(torch.from_numpy(np.array(resized_pred)))
        pred_depths = torch.stack(pred_depths, dim=0).to(rgbs.device, rgbs.dtype)

        # [B, 1, H, W]
        return pred_depths.unsqueeze(1)

    def flownet_inference_ar(self, latents_now, action):
        rgbs = self.vae.decode(latents_now / self.vae.config.scaling_factor, return_dict=False)[0]
        pred_depths = self.depth_estimation(rgbs)
        current_imgs = torch.cat([rgbs, pred_depths], dim=1).contiguous()
        flow_pred = self.flownet(current_imgs, action)
        return flow_pred, pred_depths

    def training_step(self, batch, batch_idx):
        batch_size, seq_len = batch["action"].shape[0:2]

        flows, flow_loss, _ = self.flownet_inference(batch, batch_idx)
        depth_and_flows = torch.cat([batch["depths"][:, :-1].unsqueeze(2), flows], dim=2)

        diffusion_loss = 0.0
        for t in range(seq_len):
            # Diffusion VAE requires input that are normalized to [-1, 1]
            # And the image size should be in the proportion of 8
            latents_now = self.vae.encode(self.transforms(batch["rgbs"][:, t])).latent_dist.sample()
            latents_now *= self.vae.config.scaling_factor
            latents_next = self.vae.encode(self.transforms(batch["rgbs"][:, t + 1])).latent_dist.sample()
            latents_next *= self.vae.config.scaling_factor

            # Add a noise onto latent
            noise = torch.randn_like(latents_now)
            bsz = latents_now.shape[0]
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents_now.device,
            )
            timesteps = timesteps.long()
            noisy_latents = self.scheduler.add_noise(latents_next, noise, timesteps)
            input_latents = torch.cat([noisy_latents, latents_now], dim=1)
            # Action condition, need shape [B, 1, D]
            encoder_hidden_states = batch["action"][:, t].unsqueeze(1)

            model_pred = self.unet(
                input_latents,
                timesteps,
                encoder_hidden_states,
                depth_and_flows[:, t],
                return_dict=False,
            )[0]
            diffusion_loss += F.mse_loss(model_pred, noise, reduction="mean")

        diffusion_loss /= seq_len

        loss = diffusion_loss + flow_loss
        self.log(
            "train/total_loss",
            loss.item(),
            prog_bar=True,
            batch_size=batch["action"].shape[0],
            sync_dist=True,
        )
        self.log(
            "train/diffusion_loss",
            diffusion_loss.item(),
            batch_size=batch["action"].shape[0],
            sync_dist=True,
        )
        self.log(
            "train/flow_loss",
            flow_loss.item(),
            batch_size=batch["action"].shape[0],
            sync_dist=True,
        )

        return loss

    def _inference_ar_batch(self, batch, batch_idx, output_dir=None):
        psnr_metric = piqa.PSNR(reduction="none").to(self.device)
        ssim_metric = piqa.SSIM(reduction="none").to(self.device)

        batch_size, seq_len = batch["action"].shape[0:2]
        latents_now = self.vae.encode(self.transforms(batch["rgbs"][:, 0])).latent_dist.sample()
        latents_now *= self.vae.config.scaling_factor

        psnr_values = []
        ssim_values = []

        sample_videos = [[] for _ in range(batch_size)]

        for t in range(seq_len):
            if not batch["masks"][:, t].any():
                break
            # prepare timesteps
            timesteps, _ = retrieve_timesteps(self.scheduler, 50, latents_now.device)
            # prepare latents
            noise_latents = torch.randn_like(latents_now)
            bsz = latents_now.shape[0]

            flows, pred_depths = self.flownet_inference_ar(latents_now, batch["action"][:, t])

            for i, ts in enumerate(timesteps):
                noise_latents = self.scheduler.scale_model_input(noise_latents, ts)
                input_latents = torch.cat([noise_latents, latents_now], dim=1)
                encoder_hidden_states = batch["action"][:, t].unsqueeze(1)

                noise_pred = self.unet(
                    input_latents,
                    ts,
                    encoder_hidden_states,
                    torch.cat([pred_depths, flows], dim=1),
                    return_dict=False,
                )[0]
                noise_latents = self.scheduler.step(noise_pred, ts, noise_latents, return_dict=False)[0]

            latents_now = noise_latents

            image = self.vae.decode(noise_latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = torch.clamp((image.float() + 1.0) / 2.0, 0, 1)  # denormalize
            gt = batch["rgbs"][:, t + 1].float()

            psnr_batch = psnr_metric(image.float(), gt.float())[batch["masks"][:, t]]
            ssim_batch = ssim_metric(image.float(), gt.float())[batch["masks"][:, t]]
            psnr_values.extend(psnr_batch.cpu().numpy())
            ssim_values.extend(ssim_batch.cpu().numpy())

            gt_flows = batch["flows"][:, t]
            flows = flows.float()

            if self.global_rank == 0 and batch_idx == 0:
                for i in range(batch_size):
                    if batch["masks"][i][t]:
                        frame = torch.cat(
                            [
                                image[i].cpu(),
                                flow_auto_to_image(flows[i]),
                                gt[i].cpu(),
                                flow_auto_to_image(gt_flows[i]),
                            ],
                            dim=-1,
                        )
                        sample_videos[i].append(frame)

            if output_dir is not None:
                for i in range(batch_size):
                    if batch["masks"][i][t]:
                        real_t = int(batch["index"][i][1]) + t + 1
                        save_image(
                            image[i],
                            f"{output_dir}/pred/{batch['index'][i][0]}_{real_t}.png",
                        )
                        save_image(
                            gt[i],
                            f"{output_dir}/gt/{batch['index'][i][0]}_{real_t}.png",
                        )
                        save_image(
                            flow_auto_to_image(flows[i]),
                            f"{output_dir}/flows_pred/{batch['index'][i][0]}_{real_t}.png",
                        )
                        save_image(
                            flow_auto_to_image(gt_flows[i]),
                            f"{output_dir}/flows_gt/{batch['index'][i][0]}_{real_t}.png",
                        )

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        return avg_psnr, avg_ssim, sample_videos

    def validation_step(self, batch, batch_idx):
        output_dir = None

        batch_size, seq_len = batch["action"].shape[0:2]
        avg_psnr, avg_ssim, sample_videos = self._inference_ar_batch(batch, batch_idx, output_dir)
        # if self.global_rank == 0 and batch_idx == 0:
        #     tb_logger: SummaryWriter = self.logger.experiment
        #     save_videos = []
        #     for i in range(batch_size):
        #         sample_video = torch.stack(sample_videos[i], dim=0)
        #         if sample_video.shape[0] < seq_len:
        #             sample_video = torch.cat([sample_video, sample_video[-1:].repeat_interleave(seq_len - sample_video.shape[0], 0)], dim=0)
        #         save_videos.append(sample_video)
        #     tb_logger.add_video(tag="validation_videos", vid_tensor=torch.stack(save_videos, dim=0), global_step=self.global_step, fps=3)

        self.log("val/psnr", avg_psnr, batch_size=batch_size, sync_dist=True, on_step=True, on_epoch=True)
        self.log("val/ssim", avg_ssim, batch_size=batch_size, sync_dist=True, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        os.makedirs(
            f"./{self.evaluation_output_path}/pred",
            exist_ok=True,
        )
        os.makedirs(f"./{self.evaluation_output_path}/gt", exist_ok=True)
        os.makedirs(
            f"./{self.evaluation_output_path}/flows_pred",
            exist_ok=True,
        )
        os.makedirs(
            f"./{self.evaluation_output_path}/flows_gt",
            exist_ok=True,
        )

        batch_size = batch["action"].shape[0]
        output_dir = f"./{self.evaluation_output_path}"
        avg_psnr, avg_ssim, _ = self._inference_ar_batch(batch, batch_idx, output_dir)

        self.log("test/psnr", avg_psnr, batch_size=batch_size, sync_dist=True, on_epoch=True)
        self.log("test/ssim", avg_ssim, batch_size=batch_size, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) + list(self.flownet.parameters()),
            lr=self.lr,
        )
        lr_scheduler = get_scheduler("constant", optimizer=optimizer)
        return [optimizer], [lr_scheduler]



def tensor_to_numpy(tensor: torch.Tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    return np.clip(tensor.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)


def tensors_to_videos(tensors: torch.Tensor):
    return np.clip(tensors.cpu().numpy() * 255, 0, 255).astype(np.uint8)