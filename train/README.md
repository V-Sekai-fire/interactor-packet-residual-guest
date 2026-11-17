<!--
SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee

SPDX-License-Identifier: MIT
-->

# Predictor training scripts

Research scripts for the learned predictor. Each reads `$TMPDIR/vae_testcodes.npy` and the dataset's train split, and prints wire bytes per tick with every packet decoded back exactly. Results are in `docs/design.md`.

- `rung3*.py`: MLP rungs (float, 16-bit fixed, ternary QAT). The flat-window MLP is blocklisted; these are the record of why.
- `distill_gdn.py`, `gdn3.py`: Gated DeltaNet-2 student, float with an integer step, L1 and soft-miss loss, dead-reckoning skip, 100-tick windows.
- `gdn_t.py`: the same with ternary q, k, v and head, and the state-path gates left higher precision. Not yet measured.
- `learning_curve_gdn.py`: bytes against number of training clips.

Imports are by file name, so run from this directory.
