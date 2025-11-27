import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision import transforms
from piqa import PSNR, SSIM
from lpips import LPIPS
import clip


class ImagePairDataset(Dataset):
    def __init__(self, folder1, folder2):
        self.folder1 = folder1
        self.folder2 = folder2

        files1 = set(os.listdir(folder1))
        files2 = set(os.listdir(folder2))
        self.common_files = list(files1.intersection(files2))

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        img_name = self.common_files[idx]
        img1_path = os.path.join(self.folder1, img_name)
        img2_path = os.path.join(self.folder2, img_name)

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except Exception:
            print(img_name)
            return self.__getitem__(np.random.randint(0, len(self)-1))

        img1 = TF.resize(TF.to_tensor(img1), (256, 256))
        img2 = TF.resize(TF.to_tensor(img2), (256, 256))

        return img1, img2


def calculate_metrics(folder1, folder2, batch_size=256, num_workers=48):
    dataset = ImagePairDataset(folder1, folder2)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = PSNR(reduction="none").to(device)
    ssim_metric = SSIM(reduction="none").to(device)
    lpips_metric = LPIPS().to(device)
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)
    dinov2.eval()
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    psnr_values = []
    ssim_values = []
    lpips_values = []
    dinov2_values = []
    clip_values = []

    with torch.no_grad():
        for img1_batch, img2_batch in tqdm(dataloader):
            img1_batch = img1_batch.to(device)
            img2_batch = img2_batch.to(device)

            psnr_batch = psnr_metric(img1_batch, img2_batch)
            ssim_batch = ssim_metric(img1_batch, img2_batch)
            lpips_batch = lpips_metric(img1_batch, img2_batch).squeeze()

            feature1 = dinov2(transform(img1_batch))
            feature2 = dinov2(transform(img2_batch))
            dinov2_batch = torch.norm(feature1 - feature2, dim=1)

            clip_feature1 = clip_model.encode_image(transform(img1_batch))
            clip_feature2 = clip_model.encode_image(transform(img2_batch))
            clip_batch = torch.nn.functional.cosine_similarity(clip_feature1, clip_feature2)
            
            psnr_values.extend(psnr_batch.cpu().numpy())
            ssim_values.extend(ssim_batch.cpu().numpy())
            lpips_values.extend(lpips_batch.cpu().numpy())
            dinov2_values.extend(dinov2_batch.cpu().numpy())
            clip_values.extend(clip_batch.cpu().numpy())

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_dinov2 = np.mean(dinov2_values)
    avg_clip = np.mean(clip_values)

    return {"psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips, "dinov2": avg_dinov2, "clip": avg_clip}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True)
    parser.add_argument("--folder2", type=str, required=True)
    args = parser.parse_args()

    folder1 = args.folder1
    folder2 = args.folder2

    metrics = calculate_metrics(folder1, folder2)
    print(f"Results between {folder1} and {folder2}:")
    print(f"Average PSNR: {metrics['psnr']:.4f}")
    print(f"Average SSIM: {metrics['ssim']:.4f}")
    print(f"Average LPIPS: {metrics['lpips']:.4f}")
    print(f"Average DINOv2 L2: {metrics['dinov2']:.4f}")
    print(f"Average CLIP score: {metrics['clip']:.4f}")