# FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation

<p align="left">
    <a href='https://arxiv.org/'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://sharinka0715.github.io/FlowDreamer'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

This repository is the official implemetation of the paper "[FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation](https://arxiv.org/)".

![Overview](./assets/framework.png)

The code is preparing and will be released soon.

<!-- ## Installation

The code has been tested on Ubuntu 22.04, Python 3.12, PyTorch 2.5.1 with CUDA 12.4.

```shell
# The example for Anaconda installation. You can skip them and install on your own environment.
conda create -n flowdm python=3.12
conda install cuda -c nvidia/label/cuda-12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install xformers and torch-cluster. Their version should match the version of PyTorch.
pip install -U xformers==0.0.29.post1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

pip install diffusers["torch"] transformers opencv-python decord imageio scipy scikit-image lightning wandb wandb[media] ffmpeg piqa lpips pytorch_fid
pip install "moviepy<2" # for wandb save videos
```

## Models

We start to train our FlowDreamer from [Stable Diffusion 2.1 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), you need to download this model and set `--pretrained_path` to the directory of SD 2.1.

Flowdreamer needs a metric depth estimation model to perform autoregressive inference, and we choose [Depth Anything V2 for Metric Depth Estimation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) and finetune it on our training set to perform metric depth estimation.

We also provide some checkpoints used in our experiments.

| Datasets       | FlowDreamer   | Depth Estimator |
|----------------|---------------|-----------------|
| RT-1 Simpler   | [Download]() | [Download]()     |
| Language Table | [Download]() | [Download]()     |

## Data Preparation

We extract the rigid body transformation from the simulator, and calculate the 3D scene flow during training.

The structure of our dataset is as follows:
```
dataset_root
├── annotation
│   ├── test
│   │   └── 004900.json
│   ├── train
│   └── val
└── videos
    ├── test
    │   └── 004900
    │       ├── 0_depth.png
    │       ├── 1_depth.png
    │       ├── 2_depth.png
    │       ├── ...
    │       ├── intrinsics.txt
    │       ├── mask.npz
    │       ├── pose.txt
    │       ├── rgb.mp4
    │       └── seg_poses.npz
    ├── train
    └── val

```

* **RGB frames** are saved in `.mp4` format.
* **Depth maps** are saved in 16-bit `.png` format.
* **Robot actions** are saved in `.json` format.
* **Intrinsics and extrinsics** are saved in numpy `.txt` format.
* **Segmentation masks and rigid body poses** are saved in `.npz` format.

You can download our example data from [here]().

The detailed dataset information used in our paper is listed in the following table:

| Dataset name     | Height | Width | Action dim |
|------------------|--------|-------|------------|
| RT-1 Simpler     | 256    | 320   | 7          |
| Language Table   | 288    | 512   | 2          |
| VP$^2$ RoboDesk  | 320    | 320   | 5          |
| VP$^2$ Robosuite | 256    | 256   | 4          |

## Usage

To train FlowDreamer, run:

```shell
torchrun --nproc_per_node=8 main.py --dataset_dir /PATH/TO/YOUR/DATASET/ \
  --pretrained_path /PATH/TO/YOUR/SD21/ \
  --depth_est_path /PATH/TO/YOUR/DEPTH_ANYTHING_V2/ \
  --height HEIGHT --width WIDTH --action_dim ACTION_DIM
```

To evaluate FlowDreamer, run:
```shell
python main.py --dataset_dir /PATH/TO/YOUR/DATASET/ \
  --pretrained_path /PATH/TO/YOUR/SD21/ \
  --depth_est_path /PATH/TO/YOUR/DEPTH_ANYTHING_V2/ \
  --height HEIGHT --width WIDTH --action_dim ACTION_DIM \
  --evaluate --eval_length EVAL_LENGTH \
  --ckpt_path /PATH/TO/YOUR/TRAINED_CHECKPOINTS/
```

## Acknowledgement

The training code is mainly based on [huggingface/diffusers](https://github.com/huggingface/diffusers).

The depth estimator code is based on [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), and we use the `metric_depth` version.

The FID calculation code is based on [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid), and the FVD calculation code is based on [universome/stylegan-v](https://github.com/universome/stylegan-v).

## Citation

If you find this project useful, please cite our paper as:
```bibtex
```

 -->
