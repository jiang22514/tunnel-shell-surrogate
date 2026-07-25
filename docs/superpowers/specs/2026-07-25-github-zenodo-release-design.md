# GitHub-Zenodo v1.0.0 Release Design

## Goal

Publish an enhanced compact, citable reproducibility package for the
manuscript *A shell-reduced surrogate for frost-season thermal internal
forces in cold-region tunnel linings across section geometries*.

The release must:

- reproduce the primary 20-section test-set results;
- provide small, nonconfidential source data for the reported baseline,
  statistical, ablation, alternative-split, and learning-curve analyses;
- exclude confidential Genieshan survey and finite-element data;
- measure internally how much of the paper is reproduced, without publishing
  a claim-by-claim coverage matrix or requiring complete paper-wide
  reproduction; and
- revise only the English and Chinese Data Availability/code-release
  statements so that they match the verified public scope.

The abstract, highlights, methods, results, discussion, conclusions, numerical
macros, figures, captions, and tables must not be changed.

## Selected publication route

Use the existing public GitHub repository,
`jiang22514/tunnel-shell-surrogate`, as the version-controlled source and
connect it to Zenodo before publishing GitHub release `v1.0.0`. Zenodo will
archive the tagged GitHub snapshot and assign a persistent DOI.

This route was selected over:

1. a manual Zenodo upload, which would require manual synchronization for
   every later version; and
2. separate software and dataset records for the compact materials, which
   would add citation and versioning overhead to a package below 20 MB.

The approximately 500 MB synthetic full-field arrays remain outside this
software release. If they are later deposited, they will use a separate
Zenodo dataset record.

The existing Git history will be retained. The Genieshan name and the
approximate geometry values already present in comments are not confidential,
and the history contains no surveyed outline, CAD file, Abaqus input/output
file, or full-field Genieshan result.

## Release scope

The `v1.0.0` tagged snapshot will include:

- portable evaluation and training code under `code/`;
- the 100 synthetic section-parameter rows;
- processed 200-station mechanics and temperature profiles;
- three mechanics checkpoints and three temperature checkpoints;
- the synthetic 20-section primary test predictions and error arrays;
- per-section 2D-baseline resultant errors, without raw full-field arrays;
- three-seed prediction arrays for the four reported ablation configurations;
- three-seed mechanics and temperature prediction arrays for each of the five
  reported alternative splits;
- three-seed learning-curve result files;
- a portable analysis entry point that recomputes the nonconfidential
  statistics and source figures;
- regression tests and the Python dependency list;
- the MIT software licence;
- a CC BY 4.0 notice for the released synthetic and processed data;
- `CITATION.cff`;
- a data dictionary and file manifest;
- SHA-256 checksums for the released data and model files; and
- an updated README with exact reproduction commands, expected results,
  release scope, licences, and confidentiality statement.

The curated analysis inputs will use descriptive public names:

- `data/analysis/primary_test_predictions.npz`;
- `data/analysis/baseline_per_case.json`;
- `data/analysis/ablation_predictions/`;
- `data/analysis/split_predictions/`;
- `data/analysis/learning_curves/`; and
- `data/analysis/expected_statistics.json`.

They will contain numeric arrays or JSON only. Internal experiment names and
absolute author-workspace paths will not be retained in public filenames or
metadata.

The analysis entry point, `code/reproduce_statistics.py`, will attempt to
recompute:

- the primary headline metrics and their per-section values;
- bootstrap confidence intervals of the medians;
- the paired profile-operator versus 2D-baseline bending comparison and
  Wilcoxon test;
- the reported four-row ablation table;
- the five alternative-split ranges;
- the learning-curve source table; and
- the nonconfidential source figures for the test distributions, selected
  profiles, station-wise errors, and learning curve.

The release will not include:

- surveyed Genieshan outlines, coordinates, CAD files, fitted point clouds,
  or raw measurements;
- Genieshan Abaqus models, meshes, boundary conditions, input files, output
  databases, CSV exports, truth profiles, prediction profiles, or other
  derived response data;
- the approximately 500 MB synthetic full-field arrays used to train and
  evaluate the extended 2D baseline;
- full-field 2D-baseline prediction arrays that cannot be interpreted without
  the omitted full-field truth;
- local author-workspace scripts that require private absolute paths; or
- unreviewed temporary outputs, caches, and manuscript build artefacts.

The public Genieshan name and the two approximate geometry values
`R_top ~ 6.6 m` and `R_inv_ratio ~ 1.47` may remain.

## Reproducibility boundary

The enhanced compact package is intended to support independent recomputation
of the reported nonconfidential metrics, statistical comparisons, ablation
summaries, alternative-split ranges, learning-curve values, and selected
source figures.

It does not support end-to-end retraining of the 2D full-field baseline,
because the approximately 500 MB input fields are not in this release. It also
does not reproduce any Genieshan result, because even small truth or
prediction profiles derived from the third-party survey are treated as
restricted unless the data owner authorizes redistribution.

The README and data dictionary must describe the released files and their
provenance accurately. The paper Data Availability statements will identify
the public materials and restricted Genieshan data without a claim-by-claim
reproducibility discussion. None of these files may claim that the compact
package reproduces the raw Abaqus-to-field pipeline or all results in the
manuscript.

## Internal reproducibility audit

The public package will be tested from its own repository root using only
released files and declared dependencies. No fallback path may read files from
`/home/jiang`, `/mnt/d`, or another author-workspace directory.

For the author's internal release check, every requested paper item will
receive one of three statuses:

- `reproduced`: the released command recomputes the reported value from
  released numeric inputs within its declared tolerance;
- `partially reproduced`: the released data verify the reported downstream
  statistic or figure source values but do not reproduce upstream training or
  finite-element generation; or
- `not included`: the necessary data are confidential, deliberately omitted,
  or unavailable in portable form.

The audit summary and machine-readable observations will be written only to a
git-ignored local audit directory. They will be given to the author directly
and will not be included in the public repository, GitHub release, Zenodo
record, README, or manuscript.

The required checks are:

1. Re-run the five primary headline quantities and compare them with
   `numbers.tex` using a tolerance of 0.01 percentage points.
2. Recompute bootstrap intervals, the paired Wilcoxon result, and per-section
   baseline comparisons from the released source data.
3. Recompute the four ablation rows from the released three-seed prediction
   arrays.
4. Verify that, for each of the five alternative splits, the three mechanics
   files and three temperature files use the same 20 test indices before
   ensembling and recomputing the reported ranges.
5. Recompute the learning-curve mean and standard deviation from the released
   three-seed JSON files.
6. Attempt the four selected nonconfidential source figures and record which
   panels are exactly regenerated, numerically supported but visually
   non-identical, or unsupported.
7. Load every new NPZ file with `allow_pickle=False`, validate its schema, and
   reject object arrays, unexpected keys, nonfinite values, or inconsistent
   test indices.

Complete reproduction is not a release gate. The gate is honest
classification: any mismatch must be investigated and then reported as
partial or not reproduced unless the released input or portable analysis code
is demonstrably wrong. Paper result values must not be edited to make a test
pass.

## Repository files and metadata

The release preparation will add or revise the following public-facing files:

- `README.md`: supported workflows, expected metrics, data scope, licences,
  enhanced statistical reproduction, and restricted-data statement.
- `CITATION.cff`: release title, version, authors in manuscript order,
  repository URL, software licence, and preferred citation message. ORCID
  fields will be omitted because none are verified.
- `DATA_DICTIONARY.md`: every public array and JSON field, shape, type, unit,
  meaning, split, provenance class, and relation to manuscript outputs.
- `LICENSE-DATA`: CC BY 4.0 terms for the released synthetic and processed
  arrays. The existing MIT licence continues to cover code and model
  checkpoints.
- `SHA256SUMS`: checksums for files under `data/` and `models/`.
- `.gitignore`: generated result files, figures, caches, and local environment
  files.

Only `CITATION.cff` will be used for Zenodo metadata. A `.zenodo.json` file
will not be added because no Zenodo-specific grant, community, or controlled
access fields are needed, and Zenodo ignores `CITATION.cff` when both files
are present.

The internal files under `docs/superpowers/` are planning records rather than
release documentation. They will be removed from the final tagged snapshot
after review, while their commits remain available in Git history.

## Manuscript edit boundary

Only the following manuscript content may be revised:

- the `Data availability` section in
  `/home/jiang/paper_tust_tunnel_surrogate/main.tex`; and
- the corresponding `数据可用性` section in
  `/home/jiang/paper_tust_tunnel_surrogate_zh/main.tex`.

The revisions will be made after the reproducibility audit so they describe
observed public-package capability rather than intended capability. They will:

- list the actual public code, processed profiles, weights, and analysis source
  data without enumerating a reproducibility coverage matrix;
- state that the 500 MB full-field arrays and end-to-end 2D-baseline
  retraining are not part of the compact archive;
- state that the Genieshan survey and finite-element data are not publicly
  redistributable because of data-owner confidentiality restrictions, and
  that access requests are subject to data-owner approval;
- retain the public GitHub repository URL; and
- add the real Zenodo DOI only after it is assigned.

No abstract, highlight, method, result, discussion, conclusion, numerical
macro, figure, caption, table, or bibliography content may be edited. The
English and Chinese PDFs may be rebuilt solely to reflect the permitted Data
Availability changes.

Before and after manuscript edits, hashes of the protected section files,
`numbers.tex`, figure assets, and table-producing sources will be compared.
Any protected-file change blocks the manuscript commit.

## Git and release flow

1. Work on branch `jianghongyue/public-release-v1`; do not force-push or
   rewrite existing history.
2. Stage only the confirmed public-package files.
3. Run the full verification, reproducibility, and confidentiality audit.
4. Present the internal reproducibility audit summary to the user before
   publishing; do not add it to the public package.
5. Commit the release preparation and push the branch.
6. Open a draft pull request against GitHub `main` for final inspection.
7. Merge only after the diff and exact release archive have been reviewed.
8. The user signs in to Zenodo, connects the GitHub account, synchronizes
   repositories, and enables `jiang22514/tunnel-shell-surrogate`.
9. Create a draft GitHub release for tag `v1.0.0`, inspect its title, notes,
   tag target, and source archive, then publish it.
10. Confirm that Zenodo created the record and that both the version DOI and
    landing page resolve.
11. Update only the permitted English and Chinese Data Availability sections
    with the verified scope and assigned DOI, then rebuild the two PDFs.

## Verification and confidentiality gate

Before any push:

- run all unit tests;
- run the headline and enhanced-statistics reproduction commands;
- produce and inspect the git-ignored internal audit outputs;
- generate the four selected nonconfidential source figures and verify that
  every supported panel is backed by a released numeric field;
- run one-epoch CPU smoke tests for both training entry points;
- scan the tracked release tree for absolute author paths, common secret
  patterns, and confidential Genieshan file types or filenames;
- confirm that no file copied from the author workspace is a Genieshan result,
  including `_t3_plotdata.npz`, `_t3_genie_result.json`, survey coordinates,
  FEM exports, or derived truth/prediction profiles;
- inspect the keys, shapes, types, and `allow_pickle=False` readability of all
  NumPy archives;
- verify that no individual release file or reachable Git object resembles
  the omitted full-field dataset;
- validate YAML syntax and required fields in `CITATION.cff`;
- regenerate and verify `SHA256SUMS`; and
- build a temporary `git archive` from the final commit and repeat the
  filename, schema, and confidentiality scan against that exact archive.

Any unexpected confidential field-data file, private path, credential-like
string, failed package test, checksum mismatch, or false reproducibility claim
blocks the push. A documented partial or unsupported paper item does not block
the release.

## Failure handling and user checkpoints

- If a new analysis file disagrees with the paper, do not change the paper
  result; investigate the file/configuration mapping and report the remaining
  discrepancy.
- If GitHub CLI authentication is unavailable, stop and ask the user to run
  `gh auth login`.
- If Zenodo does not list the repository, stop after the GitHub pull request
  and ask the user to reconnect GitHub and use **Sync now** in Zenodo.
- Do not publish the GitHub release until Zenodo integration is enabled.
- Do not upload confidential files to Zenodo as either public or restricted
  content; third-party storage would itself require the data owner's approval.
- Ask the user to approve the internal audit summary, draft pull request, and
  final GitHub release publication.
