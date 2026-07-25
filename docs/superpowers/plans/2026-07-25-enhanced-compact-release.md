# Enhanced Compact Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a public `v1.0.0` compact package that contains portable code and non-confidential analysis data, verifies the package internally, and aligns only the English and Chinese Data Availability statements with the actual release scope.

**Architecture:** Keep the existing portable headline evaluator and training entry points. Add a curated `data/analysis/` layer plus a standalone statistics/figure command that reads only released files. Add machine-checkable release-integrity metadata, while writing detailed reproducibility observations only to a git-ignored local audit directory.

**Tech Stack:** Python 3.10+, NumPy, SciPy, PyTorch, Matplotlib, unittest, LaTeX, Git/GitHub, Zenodo.

## Global Constraints

- Work on public-package branch `jianghongyue/public-release-v1`; never force-push or rewrite history.
- Do not publish `REPRODUCIBILITY.md`, `reproduction_report.json`, or any claim-by-claim coverage matrix.
- Do not include `_t3_plotdata.npz`, `_t3_genie_result.json`, surveyed coordinates/outlines, CAD, Abaqus input/output, Genieshan truth/prediction profiles, or any file derived from those response data.
- The public words `Genieshan`, `R_top ~ 6.6 m`, and `R_inv_ratio ~ 1.47` are allowed.
- Do not include the approximately 500 MB full-field synthetic arrays or full-field 2D-baseline predictions.
- Public numeric archives must load with `numpy.load(..., allow_pickle=False)` and contain no object arrays, non-finite numeric values, internal absolute paths, or unexpected test-index mismatches.
- Detailed reproduction results go only under `.private_release_audit/`, which must be git-ignored.
- The manuscript edit boundary is only `/home/jiang/paper_tust_tunnel_surrogate/main.tex` Data availability and `/home/jiang/paper_tust_tunnel_surrogate_zh/main.tex` 数据可用性.
- Do not change manuscript abstracts, highlights, methods, results, discussions, conclusions, `numbers.tex`, figures, captions, tables, or bibliography.
- Do not invent a Zenodo DOI. Keep the verified GitHub URL until Zenodo assigns a real DOI.
- Existing unrelated dirty changes in all three repositories belong to the user; stage only files named by the current task.

---

### Task 1: Curated analysis data and portable statistics command

**Files:**
- Create: `data/analysis/primary_test_predictions.npz`
- Create: `data/analysis/baseline_per_case.json`
- Create: `data/analysis/ablation_predictions/<configuration>/seed_<0-2>.npz`
- Create: `data/analysis/split_predictions/split_<1-5>/{mechanics,temperature}_seed_<0-2>.npz`
- Create: `data/analysis/learning_curves/seed_<0-2>.json`
- Create: `data/analysis/expected_statistics.json`
- Create: `code/reproduce_statistics.py`
- Create: `tests/test_analysis_release.py`
- Modify: `requirements.txt`
- Include and review existing uncommitted portable-core files under `code/`, `tests/`, `README.md`, and `requirements.txt` in this task's commit.

**Interfaces:**
- Consumes: released `data/profiles_NM.npz`, `data/tprofiles_T.npz`, `data/ct_params.npy`, and the curated files listed above.
- Produces: `validate_analysis_tree(data_dir: Path) -> list[str]`, `compute_statistics(data_dir: Path) -> dict[str, object]`, `write_source_figures(data_dir: Path, output_dir: Path) -> list[Path]`, and CLI options `--data-dir`, `--output-dir`, and `--output-json`.
- CLI must not read `/home/jiang`, `/mnt/d`, environment-specific fallback paths, or manuscript files.

- [ ] **Step 1: Write the failing release-analysis tests**

Create `tests/test_analysis_release.py` with real-data assertions equivalent to:

```python
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reproduce_statistics import compute_statistics, validate_analysis_tree, write_source_figures

class AnalysisReleaseTest(unittest.TestCase):
    def test_all_released_analysis_files_have_safe_schema(self):
        self.assertEqual(validate_analysis_tree(ROOT / "data"), [])

    def test_baseline_and_ablation_values_round_to_paper_values(self):
        result = compute_statistics(ROOT / "data")
        self.assertAlmostEqual(result["baseline"]["N_median_pct"], 8.41, places=2)
        self.assertAlmostEqual(result["baseline"]["M_median_pct"], 53.35, places=2)
        self.assertAlmostEqual(result["baseline"]["M_wilcoxon_p"], 3.62396240234375e-05)
        expected = {
            "primary_k32": (9.7, 0.5, 3.2, 0.2, 12.6, 1.4),
            "parameters_pca_k32": (13.6, 0.7, 4.1, 0.5, 19.3, 2.1),
            "parameters_pca_k16": (14.0, 1.1, 4.3, 0.2, 20.0, 1.9),
            "pca_only_k32": (21.6, 0.6, 7.7, 0.9, 50.4, 6.4),
        }
        for name, want in expected.items():
            got = result["ablations"][name]
            observed = tuple(round(got[k], 1) for k in (
                "slope_mean_pct", "slope_std_pct", "N_mean_pct",
                "N_std_pct", "M_mean_pct", "M_std_pct"))
            self.assertEqual(observed, want)

    def test_alternative_split_ranges_round_to_paper_values(self):
        ranges = compute_statistics(ROOT / "data")["alternative_splits"]["ranges"]
        self.assertEqual(round(ranges["M_min_pct"], 1), 6.6)
        self.assertEqual(round(ranges["M_max_pct"], 1), 9.7)
        self.assertEqual(round(ranges["N_min_pct"], 1), 2.0)
        self.assertEqual(round(ranges["N_max_pct"], 1), 2.6)
        self.assertEqual(round(ranges["sigma_profile_min_pct"], 1), 2.5)
        self.assertEqual(round(ranges["sigma_profile_max_pct"], 1), 3.5)
        self.assertEqual(round(ranges["sigma_peak_max_pct"], 1), 2.8)
        self.assertEqual(round(ranges["cracking_extent_max_pp"], 1), 1.0)

    def test_four_source_figures_are_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_figures(ROOT / "data", Path(tmp))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/home/jiang/miniforge3/envs/lddmm/bin/python -m unittest tests.test_analysis_release -v
```

Expected: import failure because `code/reproduce_statistics.py` does not yet exist.

- [ ] **Step 3: Curate only the approved non-confidential inputs**

Use these exact source mappings, renaming public files and keys descriptively while preserving numeric values:

```text
/home/jiang/_t2_plotdata.npz -> data/analysis/primary_test_predictions.npz
/home/jiang/_t1_bff_NM_result.json -> data/analysis/baseline_per_case.json
/home/jiang/_stage3_shell_pred_ncs{0,1,2}_nobc_nopca.npz -> ablation_predictions/primary_k32/seed_{0,1,2}.npz
/home/jiang/_stage3_shell_pred_s{0,1,2}_nobc.npz -> ablation_predictions/parameters_pca_k32/seed_{0,1,2}.npz
/home/jiang/_stage3_shell_pred_k16s{0,1,2}_nobc.npz -> ablation_predictions/parameters_pca_k16/seed_{0,1,2}.npz
/home/jiang/_stage3_shell_pred_nps{0,1,2}_nobc_noparams.npz -> ablation_predictions/pca_only_k32/seed_{0,1,2}.npz
/home/jiang/_stage3_shell_pred_e4{a,b,c}_nobc_sp{1..5}_nopca.npz -> split_predictions/split_{1..5}/mechanics_seed_{0,1,2}.npz
/home/jiang/_stage3_T_pred_e4{a,b,c}_nobc_sp{1..5}_nopca.npz -> split_predictions/split_{1..5}/temperature_seed_{0,1,2}.npz
/home/jiang/_t4_lc_shell_s{0,1,2}.json -> learning_curves/seed_{0,1,2}.json
```

Do not copy any source whose name contains `t3`, `genie`, `genieshan`, `L8full`, `node`, `elem`, `odb`, or `abaqus`. Load every source NPZ with `allow_pickle=False`, validate arrays before writing, and write only numeric/bool arrays.

- [ ] **Step 4: Implement the minimal statistics command**

Implement the same relative-L2 definition used by `reproduce_headline.py`. Compute 20,000-sample median bootstrap intervals with `numpy.random.RandomState(0)`. Compute paired baseline comparisons using `scipy.stats.wilcoxon`. For ablations, compute per-seed medians for `slope_ss`, `N`, and direct `M`, followed by population mean and population standard deviation (`ddof=0`) across three seeds.

For each alternative split, assert that all three mechanics files and all three temperature files have the same `test_indices`; ensemble predictions, derive bending moment from `slope_ss`, `enn_slope`, and `T_slope` using the constants in `materials.py`, then compute direct-`N`, strain-route `M`, tensile-stress profile, tensile-stress peak, and C40 cracking-extent metrics against released reference profiles. Aggregate the five split minima/maxima.

Generate exactly four PNG files in the requested output directory:

```text
error_distributions.png
selected_profiles.png
stationwise_errors.png
learning_curve.png
```

The CLI JSON output contains computed statistics only and may be directed to `.private_release_audit/`; it is not a tracked release artifact.

- [ ] **Step 5: Run RED-to-GREEN verification**

Run:

```bash
/home/jiang/miniforge3/envs/lddmm/bin/python -m unittest tests.test_analysis_release -v
/home/jiang/miniforge3/envs/lddmm/bin/python -m unittest discover -s tests -v
/home/jiang/miniforge3/envs/lddmm/bin/python code/reproduce_statistics.py --output-dir .private_release_audit/figures --output-json .private_release_audit/statistics.json
```

Expected: every test passes; the command writes one internal JSON and four internal PNGs.

- [ ] **Step 6: Commit Task 1**

Stage only the portable core, curated public data, analysis command, tests, and requirements. Do not stage `.private_release_audit/` or planning documents.

```bash
git add README.md requirements.txt code data models tests
git commit -m "feat: add compact analysis release data"
```

---

### Task 2: Release metadata, documentation, and integrity checks

**Files:**
- Create: `code/verify_release.py`
- Create: `tests/test_release_integrity.py`
- Create: `CITATION.cff`
- Create: `DATA_DICTIONARY.md`
- Create: `LICENSE-DATA`
- Create: `MANIFEST.tsv`
- Create: `SHA256SUMS`
- Create: `.gitignore`
- Modify: `README.md`
- Modify: `requirements.txt` only if PyYAML is needed for CFF validation.

**Interfaces:**
- Produces: `audit_release(root: Path) -> list[str]` and `write_checksums(root: Path, output_path: Path) -> None`.
- `audit_release` validates schemas, checksums, manifest paths/sizes, required CFF fields, forbidden filenames/extensions, private absolute paths, and credential-like text patterns without modifying files.

- [ ] **Step 1: Write failing integrity tests**

Create tests that call the real verifier and independently recalculate every line of `SHA256SUMS` with `hashlib.sha256`. Require `CITATION.cff` fields `cff-version`, `message`, `title`, `version: 1.0.0`, `repository-code`, `license: MIT`, and authors in this order: Hongyue Jiang, Shunyuan Zhang, Peizhong Yu, Fan Wang, Zhenggui Hu. Require `.private_release_audit/` to be ignored.

- [ ] **Step 2: Run the integrity test and verify RED**

```bash
/home/jiang/miniforge3/envs/lddmm/bin/python -m unittest tests.test_release_integrity -v
```

Expected: import failure because `code/verify_release.py` is absent.

- [ ] **Step 3: Implement metadata and verifier**

Use repository URL `https://github.com/jiang22514/tunnel-shell-surrogate`. Do not add ORCIDs or a DOI. `CITATION.cff` describes software release `1.0.0`. `LICENSE-DATA` applies CC BY 4.0 to public synthetic and processed numeric data; MIT remains the code/model-checkpoint licence.

`DATA_DICTIONARY.md` documents every public array/JSON field, shape, dtype, units, test split, and provenance class. `MANIFEST.tsv` columns are `path`, `bytes`, `sha256`, `category`, and `description`. `SHA256SUMS` covers regular files under `data/` and `models/` in lexical path order.

The README must state the exact public contents, the two reproduction commands, data/code licences, the omitted 500 MB full-field arrays, and the Genieshan restriction. It must not publish a result-by-result coverage table and must not claim complete reproduction.

- [ ] **Step 4: Run GREEN verification and regenerate deterministic metadata**

```bash
/home/jiang/miniforge3/envs/lddmm/bin/python -m unittest tests.test_release_integrity -v
/home/jiang/miniforge3/envs/lddmm/bin/python -m unittest discover -s tests -v
/home/jiang/miniforge3/envs/lddmm/bin/python code/verify_release.py
```

Expected: zero integrity findings and all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add .gitignore README.md requirements.txt code/verify_release.py tests/test_release_integrity.py CITATION.cff DATA_DICTIONARY.md LICENSE-DATA MANIFEST.tsv SHA256SUMS
git commit -m "docs: prepare v1.0.0 release metadata"
```

---

### Task 3: Align English and Chinese Data Availability statements only

**Files:**
- Modify: `/home/jiang/paper_tust_tunnel_surrogate/main.tex`
- Modify: `/home/jiang/paper_tust_tunnel_surrogate_zh/main.tex`

**Interfaces:**
- Produces concise availability statements that map public files to GitHub, identify omitted full-field arrays, and describe the third-party confidentiality route for Genieshan data.
- Does not describe claim-by-claim reproducibility and does not change result wording.

- [ ] **Step 1: Create dedicated paper branches and record protected hashes**

Create branch `jianghongyue/public-release-v1` in each paper repository if absent. Record SHA-256 hashes for every tracked file except `main.tex`, `main.pdf`, `.git/`, and build artefacts (`*.aux`, `*.log`, `*.out`, `*.blg`, `*.bbl`, `*.fls`, `*.fdb_latexmk`, `*.xdv`). Keep the snapshots outside the repositories.

- [ ] **Step 2: Replace only the English Data availability paragraph**

Use this exact provisional text until a real Zenodo DOI exists:

```latex
The portable code, trained model weights, synthetic section parameters,
processed 200-station reference profiles, and non-confidential analysis source
data are available in the public GitHub repository
(\url{https://github.com/jiang22514/tunnel-shell-surrogate}). A versioned
release will be archived on Zenodo, and the DOI will be added to the final
manuscript when assigned. The approximately 500\,MB full-field synthetic
arrays used for the extended two-dimensional baseline are not included in the
compact archive and may be obtained from the corresponding author upon
reasonable request. The surveyed cross-section and associated finite-element
data from the Genieshan tunnel are not publicly available because of
confidentiality restrictions imposed by the data owner. Requests for these
restricted data may be directed to the corresponding author and are subject
to approval by the data owner.
```

- [ ] **Step 3: Replace only the Chinese 数据可用性 paragraph**

Use this exact counterpart:

```latex
可移植代码、训练模型权重、合成断面参数、处理后的 200 站点参考剖线以及
非机密分析源数据已存放于公开 GitHub 仓库
（\url{https://github.com/jiang22514/tunnel-shell-surrogate}）。该仓库的版本化
发布包将存档至 Zenodo，获得 DOI 后将在论文终稿中补充。用于扩展二维基线的
约 500\,MB 合成全场数组未包含在精简发布包中，可在合理请求下向通讯作者
获取。格聂山隧道的实测断面及相关有限元数据受数据所有方保密要求限制，
不予公开；相关受限数据的申请可提交给通讯作者，并须经数据所有方批准。
```

- [ ] **Step 4: Verify the edit boundary**

Recompute the protected hashes and require an exact match. Inspect `git diff -- main.tex` in each repository and confirm that only the availability paragraph changed. Do not rebuild or overwrite the existing PDFs at this stage.

- [ ] **Step 5: Commit only the two statement files in their own repositories**

```bash
git add main.tex
git commit -m "docs: align data availability with public release"
```

Leave every pre-existing unrelated dirty file unstaged.

---

### Task 4: Full internal audit and publishable branch preparation

**Files:**
- Modify generated deterministic `MANIFEST.tsv` and `SHA256SUMS` only if their verified inputs changed.
- Create internal only: `.private_release_audit/statistics.json`
- Create internal only: `.private_release_audit/figures/*.png`
- Create internal only: `.private_release_audit/summary.md`

**Interfaces:**
- Produces a user-facing internal summary and a verified Git branch suitable for a draft pull request.
- Does not publish a GitHub release or create a Zenodo record without the user's Zenodo integration step.

- [ ] **Step 1: Run fresh code, analysis, and smoke verification**

```bash
/home/jiang/miniforge3/envs/lddmm/bin/python -m unittest discover -s tests -v
/home/jiang/miniforge3/envs/lddmm/bin/python code/reproduce_headline.py --device cpu --output-json .private_release_audit/headline.json
/home/jiang/miniforge3/envs/lddmm/bin/python code/reproduce_statistics.py --output-dir .private_release_audit/figures --output-json .private_release_audit/statistics.json
/home/jiang/miniforge3/envs/lddmm/bin/python code/train_mechanics.py --device cpu --epochs 1 --output .private_release_audit/mechanics_smoke.pt
/home/jiang/miniforge3/envs/lddmm/bin/python code/train_temperature.py --device cpu --epochs 1 --output .private_release_audit/temperature_smoke.pt
/home/jiang/miniforge3/envs/lddmm/bin/python code/verify_release.py
```

Expected headline metrics: N 3.19%, M 9.45%, tensile-stress profile 3.40%, tensile-stress peak 3.71%, and cracking extent 1.00 percentage point.

- [ ] **Step 2: Audit the exact tracked archive**

Create a temporary `git archive` from the final public-package commit. Scan filenames and extracted content for forbidden confidential names/extensions, `/home/jiang`, `/mnt/d`, common token/password patterns, unexpected files above 100 MB, and NumPy object arrays. Confirm `.private_release_audit/` and `docs/superpowers/` are absent from the intended tagged snapshot.

- [ ] **Step 3: Write the private summary**

Write `.private_release_audit/summary.md` with headline values; baseline, bootstrap, Wilcoxon, ablation, split, and learning-curve results; generated-figure status; omitted 500 MB arrays; and Genieshan exclusion. Do not stage this file.

- [ ] **Step 4: Prepare the public branch for review**

Remove planning files under `docs/superpowers/` from the intended release snapshot in a dedicated commit while preserving their history. Re-run the exact archive audit after that commit.

- [ ] **Step 5: Push and open a draft pull request**

Push `jianghongyue/public-release-v1` to `jiang22514/tunnel-shell-surrogate` without force. Open a draft pull request against `main` summarising public scope, tests, confidentiality exclusions, and the pending Zenodo connection. Stop before merging or publishing `v1.0.0`; the user must first inspect the PR and enable the repository in Zenodo.