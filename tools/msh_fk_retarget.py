# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Joint rotations of the retargeted MS-Human-700 clip against the SOMA clip, using the retarget script's own Rig, soma_fk and targets.

  uv run --python 3.11 --with mujoco==3.4.0 --with numpy --with scipy python msh_fk_retarget.py MS-Human-700.xml MM_DIR WALK.qpos.npy

Needs ~/src/garment-proxy-retarget/prototype (branch dev/retarget-msh700) and tools/msh_fk.py. Prints the site-to-site segment direction error, then the per-joint world rotation error on the unseen half.
"""
import sys, os, importlib.util, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/src/garment-proxy-retarget/prototype"))
import retarget_msh700 as R
sys.path.insert(0, os.path.expanduser("~/src/interactor-packet-residual-guest/tools"))
import msh_fk as M
xml, mdir, qf = sys.argv[1:4]
names = open(mdir + "/skeleton.txt").read().split(); parents = [int(x) for x in open(mdir + "/parents.txt").read().split()]
clip = R.load_clip(mdir, "walk_1"); P, Rw = R.soma_fk(clip, parents)
Rm = R.C @ R.yaw_matrix(Rw[0, 0])
Wm = np.einsum("ij,fnjk->fnik", Rm, Rw)                       # clip world rotations in the model frame
Pm = np.einsum("ij,fnj->fni", Rm, P)
rig = R.Rig(xml); q = np.load(qf); n = len(q)
S = R.soma_sites(Pm, names)                                    # clip sites, model frame
T = R.build_targets(S, R.segment_scales(rig, S)) if True else None
track = rig.site_track(q[::5]); T5 = T[::5]
print("fit's own site error, mean over sites (m):", np.linalg.norm(track - T5, axis=-1).mean().round(4))
ix = {n: i for i, n in enumerate(R.SITE_NAMES)}
# segment directions: site to site along the tree, model against clip target, in the world model frame (heading removed by the fit)
pairs = [(k, par) for k, (_, _, _, par) in enumerate(R.SITES) if par is not None]
print(f"{'segment':26s} {'direction error deg (mean / p99)':>34s}")
tot = []
for k, (nme, _, sp, par) in enumerate(R.SITES):
    if par is None: continue
    a = ix[par]; dm = track[:, k] - track[:, a]; dt = T5[:, k] - T5[:, a]
    u = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((u(dm) * u(dt)).sum(-1), -1, 1))); tot.append(ang.mean())
    print(f"{par+'->'+nme:26s} {ang.mean():10.1f} / {np.percentile(ang, 99):6.1f}")
print("mean:", round(float(np.mean(tot)), 1), "deg")
# body rotations on the stock model with the same q
bm = {"Hips": "pelvis", "Spine1": "lumbar1", "Head": "head_neck", "LeftArm": "humerus_l", "LeftForeArm": "ulna_l", "LeftHand": "hand_l", "LeftLeg": "femur_l", "LeftShin": "tibia_l", "LeftFoot": "talus_l", "LeftToeBase": "toes_l",
      "RightArm": "humerus_r", "RightForeArm": "ulna_r", "RightHand": "hand_r", "RightLeg": "femur_r", "RightShin": "tibia_r", "RightFoot": "talus_r", "RightToeBase": "toes_r"}
ids = {k: rig.m.body(b).id for k, b in bm.items()}
MR = {k: np.zeros((len(q[::5]), 3, 3)) for k in bm}
for f, qq in enumerate(q[::5]):
    rig.d.qpos[:] = qq; mujoco.mj_kinematics(rig.m, rig.d)
    for k, b in ids.items(): MR[k][f] = rig.d.xmat[b].reshape(3, 3)
h = len(q[::5]) // 2; out = []
print(f"\n{'joint':14s} {'rotation error (unseen half) deg':>34s}")
for k in bm:
    A = Wm[::5, names.index(k)]
    K = M.polar_mean(np.swapaxes(A[:h], -1, -2) @ MR[k][:h]); e = M.geo_deg(A[h:] @ K, MR[k][h:]); out.append(e.mean())
    print(f"{k:14s} {e.mean():10.1f}")
print("mean:", round(float(np.mean(out)), 1), "deg")
