# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Rotations of the MS-Human-700 skeleton against the clip it was retargeted from, for the 17 SOMA joints that have a body on the muscle skeleton.

  MSH700_XML=.../MS-Human-700.xml pixi run python msh_fk.py WALK.qpos100.npz CLIP.f32 [FPS]

The retarget fits positions only (17 sites), so a joint's rotation on the muscle skeleton is whatever the fit left it. Each joint's world rotation is
compared in the clip's frame: W_model = C . W_soma . K_j, C the fixed frame change (model x forward, y left, z up; SOMA x left, y up, z forward) and K_j a constant
rest-pose offset per joint, fitted on the first half of the frames and measured on the second half (frames the fit never saw). Prints the geodesic error in degrees per joint."""
import importlib.util
import os
import pathlib
import sys

import mujoco
import numpy as np

SOMA = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"]
PARENT = {"Spine1": "Hips", "Spine2": "Spine1", "Chest": "Spine2", "Neck1": "Chest", "Neck2": "Neck1", "Head": "Neck2", "Jaw": "Head", "LeftEye": "Head",
          "RightEye": "Head", "LeftShoulder": "Chest", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
          "LeftHandThumbEnd": "LeftHand", "LeftHandMiddleEnd": "LeftHand", "RightShoulder": "Chest", "RightArm": "RightShoulder", "RightForeArm": "RightArm",
          "RightHand": "RightForeArm", "RightHandThumbEnd": "RightHand", "RightHandMiddleEnd": "RightHand", "LeftLeg": "Hips", "LeftShin": "LeftLeg",
          "LeftFoot": "LeftShin", "LeftToeBase": "LeftFoot", "RightLeg": "Hips", "RightShin": "RightLeg", "RightFoot": "RightShin", "RightToeBase": "RightFoot"}
BODY = {"Hips": "pelvis", "Spine1": "lumbar1", "Head": "head_neck",
        "LeftArm": "humerus_l", "LeftForeArm": "ulna_l", "LeftHand": "hand_l", "LeftLeg": "femur_l", "LeftShin": "tibia_l", "LeftFoot": "talus_l", "LeftToeBase": "toes_l",
        "RightArm": "humerus_r", "RightForeArm": "ulna_r", "RightHand": "hand_r", "RightLeg": "femur_r", "RightShin": "tibia_r", "RightFoot": "talus_r", "RightToeBase": "toes_r"}
C = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # model = C . soma (a cyclic permutation, det +1)


def soma_world(rot, transpose):
    """rot (F, 30, 3, 3) local -> world per joint, parents first (the list is in parent order)."""
    L = np.swapaxes(rot, -1, -2) if transpose else rot
    W = np.zeros_like(L)
    for j, n in enumerate(SOMA):
        W[:, j] = L[:, j] if n == "Hips" else W[:, SOMA.index(PARENT[n])] @ L[:, j]
    return W


def polar_mean(M):  # nearest rotation to the sum of rotations M (N, 3, 3)
    u, _, vt = np.linalg.svd(M.sum(0))
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def geo_deg(A, B):
    c = (np.trace(np.swapaxes(A, -1, -2) @ B, axis1=-2, axis2=-1) - 1) / 2
    s = np.linalg.norm(np.stack([(np.swapaxes(A, -1, -2) @ B)[..., 2, 1] - (np.swapaxes(A, -1, -2) @ B)[..., 1, 2],
                                 (np.swapaxes(A, -1, -2) @ B)[..., 0, 2] - (np.swapaxes(A, -1, -2) @ B)[..., 2, 0],
                                 (np.swapaxes(A, -1, -2) @ B)[..., 1, 0] - (np.swapaxes(A, -1, -2) @ B)[..., 0, 1]], -1), axis=-1) / 2
    return np.degrees(np.arctan2(s, c))


def model_world(npz, n_frames):
    """World rotation of each mapped body at the clip's frames."""
    here = pathlib.Path(__file__).resolve().parent
    rp = pathlib.Path(os.environ.get("MSH_ROOT_PY", pathlib.Path.home() / "3-interactor/musclemimic/musclemimic/environments/humanoids/ms_human_700_root.py"))
    s = importlib.util.spec_from_file_location("ms_human_700_root", rp)
    root = importlib.util.module_from_spec(s)
    s.loader.exec_module(root)
    sites = {"pelvis": "pelvis_mimic", "lumbar1": "upper_body_mimic", "head_neck": "head_mimic", "humerus_l": "left_shoulder_mimic", "ulna_l": "left_elbow_mimic",
             "hand_l": "left_hand_mimic", "humerus_r": "right_shoulder_mimic", "ulna_r": "right_elbow_mimic", "hand_r": "right_hand_mimic", "femur_l": "left_hip_mimic",
             "tibia_l": "left_knee_mimic", "talus_l": "left_ankle_mimic", "toes_l": "left_toes_mimic", "femur_r": "right_hip_mimic", "tibia_r": "right_knee_mimic",
             "talus_r": "right_ankle_mimic", "toes_r": "right_toes_mimic"}
    spec = mujoco.MjSpec.from_file(os.environ["MSH700_XML"])
    root.prepare_spec(spec, sites)
    m = spec.compile()
    d = mujoco.MjData(m)
    q = np.load(npz)["qpos"]
    idx = np.round(np.linspace(0, len(q) - 1, n_frames)).astype(int)
    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for n, b in BODY.items()}
    out = {n: np.zeros((n_frames, 3, 3)) for n in BODY}
    for f, i in enumerate(idx):
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        for n, b in ids.items():
            out[n][f] = d.xmat[b].reshape(3, 3)
    return out


def main():
    npz, f32 = sys.argv[1], sys.argv[2]
    clip = np.fromfile(f32, dtype=np.float32).reshape(-1, 30, 12)
    rot = clip[:, :, :9].reshape(-1, 30, 3, 3).astype(float)
    n = len(rot)
    mw = model_world(npz, n)
    h = n // 2
    best = None
    for transpose in (False, True):  # which reading of the nine numbers matches the model: chosen on the first half only
        W = soma_world(rot, transpose)
        tot = 0.0
        for nme in BODY:
            j = SOMA.index(nme)
            A = np.einsum("ij,fjk->fik", C, W[:, j])
            K = polar_mean(np.swapaxes(A[:h], -1, -2) @ mw[nme][:h])
            tot += geo_deg(A[:h] @ K, mw[nme][:h]).mean()
        if best is None or tot < best[0]:
            best = (tot, transpose)
    transpose = best[1]
    print(f"frames {n}; row-major reading chosen on the first half: {'transposed' if transpose else 'as stored'}")
    W = soma_world(rot, transpose)
    print(f"{'joint':14s} {'fit half (deg)':>15s} {'unseen half mean':>17s} {'p99':>7s}   | control: wrong joint")
    errs = []
    for nme in BODY:
        j = SOMA.index(nme)
        A = np.einsum("ij,fjk->fik", C, W[:, j])
        K = polar_mean(np.swapaxes(A[:h], -1, -2) @ mw[nme][:h])
        e_fit = geo_deg(A[:h] @ K, mw[nme][:h]).mean()
        e = geo_deg(A[h:] @ K, mw[nme][h:])
        # control: the same offset against another joint's body must be much worse
        k2 = [k for k in BODY if k != nme][(list(BODY).index(nme) + 5) % (len(BODY) - 1)]
        ec = geo_deg(A[h:] @ K, mw[k2][h:]).mean()
        errs.append(e.mean())
        print(f"{nme:14s} {e_fit:15.2f} {e.mean():17.2f} {np.percentile(e, 99):7.2f}   | vs {k2}: {ec:6.1f}")
    print(f"mean over the 17 joints on the unseen half: {np.mean(errs):.2f} deg")


if __name__ == "__main__":
    main()
