# Shell-reduced surrogate for frost-season tunnel lining internal forces — release package

Companion to: 'A shell-reduced surrogate for frost-season thermal internal
forces in cold-region tunnel linings across section geometries'.

- data/: reference profiles (N/M/Q + strain/temperature labels, 100 sections,
  200 landmark-aligned stations; profiles_NM.npz, tprofiles_T.npz) and the
  four section parameters per design (ct_params.npy). Full field dataset
  (u1/u2/t npz, ~500 MB) deposited separately on Zenodo.
- code/: unroll parametrization (unroll_ct.py, ct_outline.py), profile
  fitting/integration (shell_profiles.py), capacity check (nm_envelope.py),
  training (_stage3_shell_train.py --nobc --nopca = primary config,
  _stage3_T_train.py), 2D baseline (_stage2B_ff2.py), classical per-station
  regressors on the same reduced targets (gp_pce_baselines.py: Gaussian
  process + cubic PCE, the recommended within-family deployment), evaluation
  chain (_t2_shell_eval.py, _t3_genie_NM.py, _c2_stats.py, _c2_nm_check.py).
- models/: 3-seed weights of the primary configuration (profile heads + T heads).

Environment: python>=3.10, numpy, scipy, scikit-learn, torch. See appendix A of
the paper for all hyperparameters. To reproduce the headline table:
  python _t2_shell_eval.py && python _t3_genie_NM.py && python _c2_nm_check.py
GP/PCE rows: python gp_pce_baselines.py
