import sys

sys.path.append('..')

import argparse
import os
import copy

import torch
import joblib
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from PIL import Image

from data_io import colmap_helper
from utils import debug_utils, ray_utils
from cameras import camera_pose
from geometry.basics import Translation, Rotation
from geometry import transformations
from geometry import rotation

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed


def read_4d_humans(images_dir,raw_smpl):

    vibe_estimates = {
        'verts': [],
        'joints3d': [],
        'joints2d_img_coord': [],
        'pose': [],
        'betas': [],
    }

    hmr_result = joblib.load(raw_smpl)
    for item in hmr_result.items():
        picture = item[1]

        vibe_estimates['joints3d'].append(np.array(picture['3d_joints'][0][0]))
        vibe_estimates['joints2d_img_coord'].append(np.array(picture['2d_joints'][0][0]))


        vibe_estimates['betas'].append(picture['smpl'][0]['betas'])
        # 处理bodypose
        body_pose = []
        for pose in picture['smpl'][0]['body_pose']:
            body_pose.append(rotation.rotation_matrix_to_angle_axis(torch.tensor(pose)))
        vibe_estimates['pose'].append(body_pose)

    vibe_estimates['joints2d_img_coord'] = np.array(vibe_estimates['joints2d_img_coord'])
    vibe_estimates['joints3d'] = np.array(vibe_estimates['joints3d'])
    vibe_estimates['pose'] = np.array(vibe_estimates['pose'])
    return vibe_estimates


def solve_translation(p3d, p2d, mvp):
    p3d = torch.from_numpy(p3d.copy()).float()
    p2d = torch.from_numpy(p2d.copy()).float()
    mvp = torch.from_numpy(mvp.copy()).float()
    translation = torch.zeros_like(p3d[0:1, 0:3], requires_grad=True)
    optim_list = [
        {"params": translation, "lr": 1e-3},
    ]
    optim = torch.optim.Adam(optim_list)

    total_iters = 1000
    for i in tqdm(range(total_iters), total=total_iters):
        xyzw = torch.cat([p3d[:, 0:3] + translation, torch.ones_like(p3d[:, 0:1])], axis=1)
        camera_points = torch.matmul(mvp, xyzw.T).T
        image_points = camera_points / camera_points[:, 2:3]
        image_points = image_points[:, :2]
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(image_points, p2d)
        loss.backward()
        optim.step()
    print('loss', loss, 'translation', translation)
    return translation.clone().detach().cpu().numpy()


def solve_scale(joints_world, cap, plane_model):
    cam_center = cap.cam_pose.camera_center_in_world
    a, b, c, d = plane_model
    scales = []
    for j in joints_world:
        jx, jy, jz = j
        right = -(a * cam_center[0] + b * cam_center[1] + c * cam_center[2] + d)
        coe = a * (jx - cam_center[0]) + b * (jy - cam_center[1]) + c * (jz - cam_center[2])
        s = right / coe
        if s > 0:
            scales.append(s)
    return min(scales)


def solve_transformation(j3d, j2d, plane_model, colmap_cap, smpl_cap):
    mvp = np.matmul(smpl_cap.intrinsic_matrix, smpl_cap.extrinsic_matrix)
    trans = solve_translation(j3d, j2d, mvp)
    smpl_cap.cam_pose.camera_center_in_world -= trans[0]
    joints_world = (ray_utils.to_homogeneous(
        j3d) @ smpl_cap.cam_pose.world_to_camera.T @ colmap_cap.cam_pose.camera_to_world.T)[:, :3]
    scale = solve_scale(joints_world, colmap_cap, plane_model)

    transf = smpl_cap.cam_pose.world_to_camera.T * scale
    transf[3, 3] = 1
    transf = transf @ colmap_cap.cam_pose.camera_to_world_3x4.T

    transl = transf[3]

    rot = smpl_cap.cam_pose.world_to_camera.T
    rot[3, 3] = 1
    rot = rot @ colmap_cap.cam_pose.camera_to_world.T
    rot = rot[:3, :3]
    # print(rot)
    # print(transl)
    # transf是alignment矩阵，scale是缩放倍数，transl是平移,rot是旋转
    return transf, scale, transl, rot


def make_smpl_opt(path, scales, transl, rotations):
    hmr_result = joblib.load(path)
    body_pose = []
    global_orient = []
    bbox = []
    betas = np.zeros((1, 10))
    for item, rot in zip(hmr_result.items(), rotations):
        pic = item[1]
        # 处理body_pose
        tmpbdps = []
        for pose in pic['smpl'][0]['body_pose']:
            tmpbdps.extend(rotation.rotation_matrix_to_angle_axis(torch.tensor(pose)))
        body_pose.append(tmpbdps)
        # 处理bbox
        tmpbbox = pic['bbox'][0]
        tmpbbox[2] = tmpbbox[0] + tmpbbox[2]
        tmpbbox[3] = tmpbbox[1] + tmpbbox[3]
        bbox.append(tmpbbox)
        # 处理global_orient
        tmpgo = pic['smpl'][0]['global_orient'][0]  # 已经是旋转矩阵了
        tmpgo = rot.T @ np.array(tmpgo)
        tmpgo = torch.tensor(tmpgo)
        tmpgo = rotation.rotation_matrix_to_angle_axis(tmpgo)  # 转成轴角
        global_orient.append(np.array(tmpgo))

        # 处理betas
        betas = betas + np.array(pic['smpl'][0]['betas'])
    betas = np.divide(betas, len(body_pose))
    betas = np.tile(betas, (len(body_pose, ), 1))

    np.savez('smpl_optimized_aligned_scale.npz',
             global_orient=global_orient,
             scale=scales,
             transl=transl,
             body_pose=body_pose,
             bbox=bbox,
             betas=betas
             )
    print('save smpl opt align.npz')
    return


def main(opt):
    scene = colmap_helper.ColmapAsciiReader.read_scene(
        opt.scene_dir,
        opt.images_dir,
        order='video'
    )
    raw_smpl = read_4d_humans(opt.images_dir, opt.raw_smpl)

    print(len(raw_smpl['pose']),len(scene.captures))
    assert len(raw_smpl['pose']) == len(scene.captures)

    
    # estimate the ground
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(scene.point_cloud[:, 3:] / 255)

    plane_model, inliers = pcd.segment_plane(0.02, 3, 1000)
    pcd.points = o3d.utility.Vector3dVector(scene.point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(scene.point_cloud[:, 3:] / 255)
    inliers = np.abs(np.sum(np.multiply(scene.point_cloud[:, :3], plane_model[:3]), axis=1) + plane_model[3]) < 0.02
    inliers = list(np.where(inliers)[0])
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])

    # solve the alignment
    alignments = {}
    transls = []
    rotations = []
    scales = []
    for i, cap in tqdm(enumerate(scene.captures), total=len(scene.captures)):
        pts_3d = raw_smpl['joints3d'][i]
        pts_2d = raw_smpl['joints2d_img_coord'][i]
        _, R_rod, t, inl = cv2.solvePnPRansac(pts_3d, pts_2d, cap.pinhole_cam.intrinsic_matrix, np.zeros(4),
                                              flags=cv2.SOLVEPNP_EPNP)
        t = t.astype(np.float32)[:, 0]
        R, _ = cv2.Rodrigues(R_rod)
        quat = transformations.quaternion_from_matrix(R).astype(np.float32)

        smpl_cap = copy.deepcopy(cap)
        smpl_cam_pose = camera_pose.CameraPose(Translation(t), Rotation(quat))
        smpl_cap.cam_pose = smpl_cam_pose

        # refine the translation and solve the scale
        # return transf,scale,transl,rot
        transf, scale, transl, rot = solve_transformation(
            raw_smpl['joints3d'][i],
            raw_smpl['joints2d_img_coord'][i],
            plane_model,
            cap,
            smpl_cap
        )
        scales.append(scale)
        transls.append(transl)
        rotations.append(rot)
        alignments[os.path.basename(cap.image_path)] = transf
    save_path = os.path.abspath(os.path.join(opt.scene_dir, '../alignments.npy'))
    np.save(save_path, alignments)
    print(f'alignment matrix saved at: {save_path}')
    make_smpl_opt(opt.raw_smpl, scales, transls, rotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, default=None, required=True)
    parser.add_argument('--images_dir', type=str, default=None, required=True)
    parser.add_argument('--raw_smpl', type=str, default=None, required=True)
    opt = parser.parse_args()
    main(opt)
