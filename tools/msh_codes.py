# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Packet codes from the retargeted MS-Human-700 clip, for the 17 SOMA joints that have a body on the muscle skeleton (the clip's own rotation where the parent has none).

  uv run --python 3.11 --with mujoco==3.4.0 --with numpy --with scipy python msh_codes.py MS-Human-700.xml MM_DIR CLIP.qpos.npy CLIPNAME

Offsets per joint are fitted on the first half of the frames, the error and the dead-reckoning cost are measured on the second half. Same prerequisites as msh_fk_retarget.py.
"""
import sys, os, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/src/garment-proxy-retarget/prototype")); sys.path.insert(0, os.path.expanduser("~/src/interactor-packet-residual-guest/tools")); sys.path.insert(0, os.path.expanduser("~/src/interactor-packet-residual-guest/train"))
import retarget_msh700 as R, msh_fk as M, pose_codes as pc, rot_frontier as rf
xml, mdir, qf, clipname = sys.argv[1:5]
names = open(mdir + "/skeleton.txt").read().split(); parents = [int(x) for x in open(mdir + "/parents.txt").read().split()]
clip = R.load_clip(mdir, clipname); P, Rw = R.soma_fk(clip, parents)
Rm = R.C @ R.yaw_matrix(Rw[0, 0]); Wm = np.einsum("ij,fnjk->fnik", Rm, Rw)
rig = R.Rig(xml); q = np.load(qf); n = len(q)
bm = {"Hips": "pelvis", "Spine1": "lumbar1", "Head": "head_neck", "LeftArm": "humerus_l", "LeftForeArm": "ulna_l", "LeftHand": "hand_l", "LeftLeg": "femur_l", "LeftShin": "tibia_l", "LeftFoot": "talus_l", "LeftToeBase": "toes_l",
      "RightArm": "humerus_r", "RightForeArm": "ulna_r", "RightHand": "hand_r", "RightLeg": "femur_r", "RightShin": "tibia_r", "RightFoot": "talus_r", "RightToeBase": "toes_r"}
ids = {k: rig.m.body(b).id for k, b in bm.items()}; MR = {k: np.zeros((n, 3, 3)) for k in bm}
for f in range(n):
    rig.d.qpos[:] = q[f]; mujoco.mj_kinematics(rig.m, rig.d)
    for k, b in ids.items(): MR[k][f] = rig.d.xmat[b].reshape(3, 3)
h = n // 2
Wnew = Wm.copy()                                       # model-space world rotation of every SOMA joint: the muscle one where there is a body, the clip's elsewhere
for k in bm:
    j = names.index(k); A = Wm[:, j]
    K = M.polar_mean(np.swapaxes(A[:h], -1, -2) @ MR[k][:h]); Wnew[:, j] = MR[k] @ K.T
Rw2 = np.einsum("ji,fnjk->fnik", Rm, Wnew)             # back to SOMA world
def local(Wd):
    L = np.zeros_like(Wd)
    for j in range(30):
        L[:, j] = Wd[:, j] if parents[j] < 0 else np.swapaxes(Wd[:, parents[j]], -1, -2) @ Wd[:, j]
    return L
L_clip, L_new = local(Rw), local(Rw2)
cj = [names.index(k) for k in bm]
def codes_of(L):
    return np.stack([rf.codes(L[:, j]) for j in range(30)], 1)
Cc, Cn = codes_of(L_clip), codes_of(L_new)
print(f"{clipname}: {n} frames at 30 fps; first half fits the offsets, second half is measured")
print(f"{'joint':14s} {'local rotation error deg':>25s} {'swing err':>10s} {'twist err':>10s}")
tot = []
for k in bm:
    j = names.index(k); E = np.swapaxes(L_clip[h:, j], -1, -2) @ L_new[h:, j]
    e = M.geo_deg(np.broadcast_to(np.eye(3), E.shape), E); c = np.array([pc.codes_of_matrix(tuple(x.reshape(-1))) for x in E[::9]], float)
    sw = np.hypot(c[:, 0], c[:, 1]) * 180 / 32767; tw = np.abs(((c[:, 2] + 32768) % 65536) - 32768) * 360 / 65536
    tot.append(e.mean()); print(f"{k:14s} {e.mean():25.1f} {sw.mean():10.1f} {tw.mean():10.1f}")
print("mean:", round(float(np.mean(tot)), 1), "deg")
# cost of dead reckoning on the unseen half, 17 mapped joints, clip codes vs muscle-derived codes (proxy bits per frame)
def bits(C):
    c2, c1, c0 = C[h:-2], C[h + 1:-1], C[h + 2:]
    a = np.abs(rf.wrap16(c0 - rf.p_codes(c2, c1)))[:, cj]; return rf.bits_of(a).sum(-1).mean(), np.mean(a == 0) * 100, np.mean(a > 126) * 100
for nm, C in (("clip codes", Cc), ("muscle-derived codes", Cn)):
    b, z, e = bits(C.astype(np.int64)); print(f"dead reckoning, {nm:22s}: {b:6.0f} bits per joint per frame (3 codes), 17 joints | exact zero {z:4.1f}% | escape {e:4.1f}%")
np.save(os.environ["TMPDIR"] + f"/msh_codes_{clipname}.npy", Cn)
