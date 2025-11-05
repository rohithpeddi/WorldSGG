import torch
import numpy as np

from datasets.preprocess.human.prompt_hmr.vis.traj import fit_floor_height
from datasets.preprocess.human.prompt_hmr.utils.one_euro_filter import smooth_one_euro
from datasets.preprocess.human.prompt_hmr.utils.rotation_conversions import (
    rotation_about_x, 
    rotation_about_y, 
    matrix_to_axis_angle,
)
from ..data_config import COLORS_TEXT_PATH


def transform_smpl_params(root_orient, transl, R, t, smpl_t_pose_pelvis):
    '''
    Input:
        root_orient: [B, 3, 3]
        transl: [B, 3]
        R: [B, 3, 3]
        t: [B, 3]
        smpl_t_pose_pelvis: [3]
    
    Output: 
        root_orient: [B, 3, 3]
        transl: [B, 3]
    '''
    assert smpl_t_pose_pelvis.shape == (3,)
    smpl_t_pose_pelvis = smpl_t_pose_pelvis[None, : , None]
    transl = transl.unsqueeze(-1).float()
    t = t.unsqueeze(-1)

    transl = R @ (smpl_t_pose_pelvis + transl) + t - smpl_t_pose_pelvis
    root_orient = R @ root_orient
    transl = transl.squeeze()
    return root_orient, transl


def world_hps_estimation(cfg, results, smplx, device):
    # ------------------------------------------------------------------
    # 1. base tensors -> device
    # ------------------------------------------------------------------
    colors = np.loadtxt(COLORS_TEXT_PATH) / 255.0
    colors = torch.from_numpy(colors).float().to(device)

    locations = []

    pred_cam = results['camera']
    img_focal = pred_cam['img_focal']      # not a tensor, OK to stay as-is
    img_center = pred_cam['img_center']    # not a tensor, OK to stay as-is

    # make camera tensors on device
    pred_cam_R = torch.tensor(pred_cam['pred_cam_R'], dtype=torch.float32, device=device)
    pred_cam_T = torch.tensor(pred_cam['pred_cam_T'], dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 2. optional smoothing (returns numpy) -> back to torch on device
    # ------------------------------------------------------------------
    if cfg.smooth_cam:
        min_cutoff = 0.001
        beta = 0.1
        pred_cam_T_np = smooth_one_euro(pred_cam_T.detach().cpu().numpy(), min_cutoff, beta)
        pred_cam_R_np = smooth_one_euro(pred_cam_R.detach().cpu().numpy(), min_cutoff, beta, is_rot=True)

        pred_cam_T = torch.from_numpy(pred_cam_T_np).float().to(device)
        pred_cam_R = torch.from_numpy(pred_cam_R_np).float().to(device)

    # ------------------------------------------------------------------
    # 3. build cam_wc on device
    # ------------------------------------------------------------------
    cam_wc = torch.eye(4, device=device).unsqueeze(0).repeat(len(pred_cam_R), 1, 1)
    cam_wc[:, :3, :3] = pred_cam_R
    cam_wc[:, :3, 3] = pred_cam_T

    # ------------------------------------------------------------------
    # 4. rectification (make sure rotation mats are on device)
    # ------------------------------------------------------------------
    if cfg.use_spec_calib:
        spec_calib = results['spec_calib']['first_frame']

        # assume rotation_about_x/y return torch OR numpy; normalize to torch on device
        Rpitch = rotation_about_x(-spec_calib['pitch'])
        if not torch.is_tensor(Rpitch):
            Rpitch = torch.from_numpy(Rpitch)
        Rpitch = Rpitch.to(device)[None, :3, :3]

        Rroll = rotation_about_y(spec_calib['roll'])
        if not torch.is_tensor(Rroll):
            Rroll = torch.from_numpy(Rroll)
        Rroll = Rroll.to(device)[None, :3, :3]

        cam_wc[:, :3, :3] = Rpitch @ Rroll @ cam_wc[:, :3, :3]
        cam_wc[:, :3, 3] = (Rpitch @ Rroll @ cam_wc[:, :3, 3].unsqueeze(-1)).squeeze(-1)

    elif cfg.use_floor_rectify:
        # TODO: implement, but make sure it's on device when you do
        pass

    Rwc = cam_wc[:, :3, :3]     # (B,3,3) on device
    Twc = cam_wc[:, :3, 3]      # (B,3)   on device

    # ------------------------------------------------------------------
    # 5. transform SMPL people
    # ------------------------------------------------------------------
    for k, v in results['people'].items():
        pred_smpl = v['smplx_cam']   # numpy dict

        # move all predicted smpl pieces to device
        for k_, v_ in pred_smpl.items():
            pred_smpl[k_] = torch.from_numpy(v_).to(device)

        pred_rotmat = pred_smpl['rotmat'].clone()   # (T, 55, 3, 3) typically
        pred_shape = pred_smpl['shape'].clone()
        pred_trans = pred_smpl['trans'].clone()

        frame = torch.from_numpy(v['frames']).long().to(device)

        # use mean shape across frames
        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        cam_r = Rwc[frame]  # (T,3,3) on device
        cam_t = Twc[frame]  # (T,3)   on device

        # pelvis in t-pose on device
        smpl_t_pose_pelvis = smplx(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 21 * 3, device=device),
            betas=mean_shape,
        ).joints[0, 0]  # (3,) on device

        root_orient = pred_rotmat[:, 0]   # (T,3,3) on device

        # transform_smpl_params must return tensors on device; if not, we enforce it
        root_orient, pred_trans = transform_smpl_params(
            root_orient, pred_trans.squeeze(), cam_r, cam_t, smpl_t_pose_pelvis
        )
        root_orient = root_orient.to(device)
        pred_trans = pred_trans.to(device)
        pred_rotmat[:, 0] = root_orient
        pred_trans = pred_trans.squeeze()

        # axis-angle conversion stays on device
        pred_pose_aa = matrix_to_axis_angle(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 55 * 3)
        B = pred_pose_aa.shape[0]

        pred = smplx(
            global_orient=pred_pose_aa[:, :3],
            body_pose=pred_pose_aa[:, 3:66],
            left_hand_pose=pred_pose_aa[:, 75:120],
            right_hand_pose=pred_pose_aa[:, 120:],
            betas=pred_shape,
            transl=pred_trans,
            jaw_pose=torch.zeros(B, 3, device=device),
            leye_pose=torch.zeros(B, 3, device=device),
            reye_pose=torch.zeros(B, 3, device=device),
            expression=torch.zeros(B, 10, device=device),
        )

        pred_vert_w = pred.vertices                 # (T, #verts, 3) on device
        pred_j3d_w = pred.joints[:, :22]            # (T, 22, 3) on device
        locations.append(pred_j3d_w[:, 0])          # list of (T,3) on device

        results['people'][k]['smplx_world'] = {
            'verts': pred_vert_w,
            'joints': pred_j3d_w,
            'rotmat': pred_rotmat.clone(),
            'shape': pred_shape,
            'trans': pred_trans.clone(),
            't_pose_pelvis': smpl_t_pose_pelvis,
        }

    # ------------------------------------------------------------------
    # 6. global flip & floor fitting on device
    # ------------------------------------------------------------------
    R = (torch.tensor([[-1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -1]], dtype=torch.float32, device=device)
         @ torch.tensor([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]], dtype=torch.float32, device=device))

    all_pred_vert_w = torch.cat([v['smplx_world']['verts'] for v in results['people'].values()], dim=0)
    # pred_vert_gr: (B*?, V, 3)
    pred_vert_gr = torch.einsum('ij,bnj->bni', R, all_pred_vert_w).detach()

    # fit_floor_height may return scalar/np/tensor; normalize to tensor on device
    offset = fit_floor_height(pred_vert_gr, 'ransac', 'y')
    offset = torch.as_tensor(offset, dtype=pred_vert_gr.dtype, device=device)

    # ------------------------------------------------------------------
    # 7. longest tracks -> device
    # ------------------------------------------------------------------
    locations = sorted(locations, key=len, reverse=True)[:2]
    locations = torch.cat(locations, dim=0)              # (N,3) on device
    locations = torch.einsum('ij,bj->bi', R, locations) - offset  # on device

    # remove verts to save memory
    for k, v in results['people'].items():
        del results['people'][k]['smplx_world']['verts']

    # ------------------------------------------------------------------
    # 8. viewing camera transforms on device
    # ------------------------------------------------------------------
    Rwc = torch.einsum('ij,bjk->bik', R, Rwc)        # (B,3,3) on device
    Twc = torch.einsum('ij,bj->bi', R, Twc) - offset  # (B,3)    on device
    Rcw = Rwc.mT                                      # (B,3,3) on device
    Tcw = -torch.einsum('bij,bj->bi', Rcw, Twc)       # (B,3)   on device

    # collect locs + cameras to compute viz params
    locations = torch.cat([locations, Tcw], dim=0)    # (N+cams,3) on device
    cx, cz = (locations.max(0).values + locations.min(0).values)[[0, 2]] / 2.0
    sx, sz = (locations.max(0).values - locations.min(0).values)[[0, 2]]
    scale = max(sx.item(), sz.item())

    # ------------------------------------------------------------------
    # 9. reapply global transform to each person's stored smplx_world
    # ------------------------------------------------------------------
    for k, v in results['people'].items():
        pose_rotmat = results['people'][k]['smplx_world']['rotmat']  # on device
        trans = results['people'][k]['smplx_world']['trans']         # on device
        shape = results['people'][k]['smplx_world']['shape']         # on device
        t_pose_pelvis = results['people'][k]['smplx_world']['t_pose_pelvis']  # on device

        root_orient, trans = transform_smpl_params(
            root_orient=pose_rotmat[:, 0],
            transl=trans,
            R=R,
            t=-offset,
            smpl_t_pose_pelvis=t_pose_pelvis,
        )
        root_orient = root_orient.to(device)
        trans = trans.to(device)

        pose_rotmat[:, 0] = root_orient
        trans = trans.squeeze()

        smplx_pose = matrix_to_axis_angle(pose_rotmat.reshape(-1, 3, 3)).reshape(-1, 55 * 3)
        smplx_trans = trans
        smplx_betas = shape

        results['people'][k]['smplx_world'] = {
            'pose': smplx_pose,     # on device
            'shape': smplx_betas,   # on device
            'trans': smplx_trans,   # on device
        }

    # ------------------------------------------------------------------
    # 10. store camera in results (must go to CPU before .numpy())
    # ------------------------------------------------------------------
    results['camera_world'] = {
        'pred_cam_R': Rwc.detach().cpu().numpy(),
        'pred_cam_T': Twc.detach().cpu().numpy(),
        'Rwc': Rwc.detach().cpu().numpy(),
        'Twc': Twc.detach().cpu().numpy(),
        'Rcw': Rcw.detach().cpu().numpy(),
        'Tcw': Tcw.detach().cpu().numpy(),
        'img_focal': img_focal,
        'img_center': img_center,
        'viz_scale': scale,
        'viz_center': [cx.item(), 0, cz.item()],
    }

    return results
