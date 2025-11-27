import json
import torch
from pathlib import Path

import tifffile
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class SingleViewSequentialDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train", seq_len: int = 15, mode: str = "random"):
        """
        The single-view sequential dataset classes.
        seq_len (int): n+1 observations and n actions will be returned. if 1, reduced to teacher-force.
        mode (str):
            - random: randomly choose a start frame from every episode. Affected by random seed.
            - discard: discard trajectories less than seq_len timesteps.
            - ignore: get all trajectories and ignore seq_len, which does not support batch.
            - padding: pad all-zero to the end of trajectories less than seq_len timesteps
            - history: choose all trajectories with seq_len timesteps, can have overlaps.
        """
        self.seq_len = seq_len
        assert mode in ["random", "discard", "ignore", "padding", "history"]
        self.mode = mode
        self.split = split

        self.data_dir = Path(data_dir)
        self.annotations = sorted(self.data_dir.glob(f"{split}/*/annotation.json"))

        length_cache = self.data_dir / f"length_{split}.json"

        if length_cache.exists():
            print("Use length cache from ", length_cache)
            with open(length_cache, "r") as fp:
                frames = [e for e in json.load(fp)]
        else:
            print(f"Length cache doesn't exist in {length_cache}, creating...")
            frames = []
            for ann in self.annotations:
                with open(ann) as fp:
                    annotations = json.load(fp)
                ann_id = ann.parent.name
                frames.append([ann_id, len(annotations["actions"])])
            with open(length_cache, "w") as fp:
                json.dump(frames, fp)

        if self.mode == "random":
            self.frames = [e for e in frames if e[1] >= self.seq_len]
        else:
            self.frames = []
            for ann, num_frames in frames:
                if self.mode == "history":
                    for start_idx in range(num_frames):
                        if start_idx + seq_len <= num_frames:
                            self.frames.append([ann, start_idx, min(seq_len, num_frames - start_idx)])
                else:
                    for start_idx in range(0, num_frames, seq_len):
                        if self.mode == "discard":
                            if start_idx + seq_len <= num_frames:
                                self.frames.append([ann, start_idx, min(seq_len, num_frames - start_idx)])
                        elif self.mode == "ignore" or self.mode == "padding":
                            self.frames.append([ann, start_idx, min(seq_len, num_frames - start_idx)])

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if self.mode == "random":
            ann_id, num_frames = self.frames[index]
            frame_id = torch.randint(0, num_frames - self.seq_len + 1, ())
            seq_len = self.seq_len
        else:
            ann_id, frame_id, seq_len = self.frames[index]

        # annotation
        with open(self.data_dir / self.split / ann_id / "annotation.json", "r") as fp:
            annotations = json.load(fp)
        action = torch.tensor(annotations["actions"][frame_id : frame_id + seq_len])
        cam_intr = torch.tensor(annotations["camera_intrinsics"])
        cam_pose = torch.tensor(annotations["camera_poses"])

        # rgb
        rgbs = torch.from_numpy(self._load_video(
            self.data_dir / self.split / ann_id / "rgb.mp4", list(range(frame_id, frame_id + seq_len + 1))
        )).float().permute(0, 3, 1, 2) / 255.0

        # depth
        depths = tifffile.imread(self.data_dir / self.split / ann_id / "depth.tiff")[frame_id : frame_id + seq_len + 1]
        depths[depths == 65535] = 0
        depths = torch.from_numpy(depths).float() / 1000.0

        # flow
        flow = tifffile.imread(self.data_dir / self.split / ann_id / "flow.tiff")[frame_id : frame_id + seq_len]
        flow = torch.from_numpy(flow).float().permute(0, 3, 1, 2)

        if self.mode == "padding" and seq_len < self.seq_len:
            masks = torch.zeros((self.seq_len,)).bool()
            masks[:seq_len] = True

            pad_action = torch.zeros_like(action[0]).unsqueeze(0).repeat_interleave(self.seq_len - seq_len, 0)
            action = torch.cat([action, pad_action], dim=0)
            pad_rgbs = torch.zeros_like(rgbs[0:1]).repeat_interleave(self.seq_len - seq_len, 0)
            rgbs = torch.cat([rgbs, pad_rgbs], dim=0)
            pad_depths = torch.zeros_like(depths[0:1]).repeat_interleave(self.seq_len - seq_len, 0)
            depths = torch.cat([depths, pad_depths], dim=0)
            pad_flow = torch.zeros_like(flow[0:1]).repeat_interleave(self.seq_len - seq_len, 0)
            flow = torch.cat([flow, pad_flow], dim=0)
        else:
            masks = torch.ones((seq_len,)).bool()

        return {
            # obs
            "rgbs": rgbs,
            "depths": depths,
            "flows": flow,
            # annotation
            "action": action,
            "masks": masks,
            "index": [ann_id, frame_id],
            "cam_pose": cam_pose,
            "cam_intr": cam_intr,
        }

    @staticmethod
    def collate_fn(batch):
        ret_batch = {}
        for key in batch[0]:
            if key not in ["index"]:
                ret_batch[key] = torch.stack([e[key] for e in batch], 0)
            else:
                ret_batch[key] = [e[key] for e in batch]
        return ret_batch
