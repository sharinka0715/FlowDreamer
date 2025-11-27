import torch
from torch_cluster import nearest


def rigid_transform_batch(xyz, transform):
    """Apply the rigid transform (B, D+1, D+1) to an (B, N, D) pointcloud."""
    xyz_h = torch.cat([xyz, torch.ones((xyz.shape[0], xyz.shape[1], 1), device=xyz.device, dtype=xyz.dtype)], dim=-1)
    # xyz_t_h = (transform @ xyz_h.T).T
    xyz_t_h = torch.einsum("bnd,bhd->bnh", xyz_h, transform)
    return xyz_t_h[..., :-1]


def single_view_unprojection_batch(depths, features, pose, intrinsics):
    """
    Parameters:
    - depths: [B, H, W]
    - features: [B, H, W, feature_dim]
    - pose: [N, 4, 4] (camera-to-world)
    - intrinsics: [N, 3, 3]

    Returns:
    - xyz: [B, N, 3] (N = H * W)
    - feat: [B, N, feature_dim]

    convention: RDF (same as COLMAP)
    """
    B, H, W = depths.shape
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    points_image = (
        torch.stack([u.flatten(), v.flatten()], dim=-1)
        .to(device=depths.device, dtype=depths.dtype)
        .unsqueeze(0)
        .repeat_interleave(B, dim=0)
    )  # [B, H * W, 2]
    points_camera = rigid_transform_batch(points_image, torch.inverse(intrinsics))  # [B, H * W, 3]
    points_camera = torch.cat(
        [
            points_camera,
            torch.ones((points_camera.shape[0], points_camera.shape[1], 1), device=depths.device, dtype=depths.dtype),
        ],
        dim=-1,
    ) * depths.reshape(
        B, H * W, 1
    )  # [B, H * W, 4]
    points_world = rigid_transform_batch(points_camera, pose)  # [B, H * W, 3]
    features = features.reshape(B, H * W, -1)

    return points_world, features


def get_scene_flow_3d_batch(pcd: torch.Tensor, body_data: torch.Tensor):
    """Calculate scene flow from old_pcd to next pcd according to body/link transformation.
    body_id and mask_id are uint8
    Parameters:
    - pcd: [B, N, 3]
    - body_data: [B, 2, Z, 4, 4]

    Returns:
    - new_pcd: [B, N, 3]
    - scene_flow: [B, N, 3]
    """
    new_pcd = pcd.clone()
    body_transforms = torch.einsum("bzij,bzjk->bzik", body_data[:, 1], body_data[:, 0].inverse())
    indices = new_pcd[..., 6].long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)
    body_index = torch.gather(body_transforms, dim=1, index=indices).to(
        device=new_pcd.device, dtype=new_pcd.dtype
    )  # [B, Z, 4, 4] -> [B, N, 4, 4]

    xyz_h = torch.cat(
        [
            new_pcd[..., :3],
            torch.ones((new_pcd.shape[0], new_pcd.shape[1], 1), device=new_pcd.device, dtype=new_pcd.dtype),
        ],
        dim=-1,
    )
    new_pcd[..., :3] = torch.einsum("bnd,bnhd->bnh", xyz_h, body_index)[..., :3]
    scene_flow = new_pcd[..., :3] - pcd[..., :3]

    return new_pcd, scene_flow


def get_optical_and_camera_flow_batch(
    xyz: torch.Tensor,
    flow: torch.Tensor,
    depth_map: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
):
    """
    Parameters:
    - xyz: [B, N, 3]
    - flow: [B, N, 3]
    - depth_map: [B, H, W]
    - camera_intrinsics: [B, 3, 3]
    - camera_poses: [B, 4, 4]

    Returns:
    - optical_flow: [B, H, W, 2]
    - camera_flow: [B, H, W, 3]
    """
    # Image grid
    B, height, width = depth_map.shape
    u, v = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")
    points_image = (
        torch.stack((u.flatten(), v.flatten()), dim=-1)
        .to(device=xyz.device, dtype=xyz.dtype)
        .unsqueeze(0)
        .repeat_interleave(B, dim=0)
    )
    # [B, H * W, 2]
    points_camera = rigid_transform_batch(points_image, camera_intrinsics.inverse())
    points_camera = torch.cat(
        [points_camera, torch.ones((*points_camera.shape[:-1], 1), device=xyz.device)], dim=-1
    ) * depth_map.reshape(*points_camera.shape[:-1], 1)
    # Transform 3D points to the world coordinate system using the camera pose (c2w)
    points_world = rigid_transform_batch(points_camera, camera_pose)

    flow_grids = []
    for i in range(B):
        flow_grid = flow[i, nearest(points_world[i], xyz[i])]
        flow_grids.append(flow_grid)
    flow_grid = torch.stack(flow_grids, dim=0)

    points_world_warped = points_world + flow_grid
    points_camera_warped = rigid_transform_batch(points_world_warped, camera_pose.inverse())
    points_image_warped = torch.einsum("bnd,bhd->bnh", points_camera_warped, camera_intrinsics)
    points_image_warped = points_image_warped[..., :2] / points_image_warped[..., 2:]

    optical_flow = (points_image_warped - points_image).reshape(-1, height, width, 2)
    optical_flow[..., 0] /= width
    optical_flow[..., 1] /= height
    camera_flow = (points_camera_warped - points_camera).reshape(-1, height, width, 3)

    return optical_flow, camera_flow
