# Cleanup validation — 2026-11-05

Branch: `refactor/project-cleanup`, based on `cb240b7`.

* 137 CPU tests passed with `python -B -m unittest discover -s tests -t . -v`.
* Seeded and controlled-unseeded fragmentation pixels match the frozen historical
  implementation across scan/random outlines, grid/uniform noise and split noise.
* Both point-training scopes pass a CPU composition test: baseline, backward,
  optimizer update, held-out evaluation, preview and checkpoint restoration.
* All 18 active Python command entry points accept `--help`.
* All shell/Slurm scripts pass `bash -n`.
* Ruff source checks and formatting pass; the frozen reference is excluded.
* The package wheel builds with `core`, including its `data_loading` subpackage.
* The offline pilot still prepares ten distinct original image/mask pairs.

The previous suite had 139 tests. Four tests specific to the rejected concept
trainer were removed; the shared training-split test was retained. Two meaningful
checks were added: full trainer composition and seeded scientific equivalence.

Historical sources/assets were archived locally to
`outputs/archive/pre-cleanup-cb240b7.zip`; every archived entry was verified by
SHA-256 before its original was removed. A manifest is beside the ZIP. The archive
also contains two obsolete refactor reports. Tracked historical sources can be
recovered from the base commit; the ZIP is ignored and is not pushed to GitHub.

This is a structural cleanup. GPU timing, native SAM3 backward/memory fit and
fine-tuning accuracy have not been rerun on Vulcan for this branch. Existing
evaluation selection and training-loss policies are preserved. The GPU-training
loop retains its pre-update native-forward check and first-update gradient audit.

The existing deletion of `docs/presentation_outline.md` and local changes to
`scripts/prepare_coco_validation.py` and `scripts/vulcan_throughput.sbatch` remain
separate from this cleanup commit. Shared-import changes to the preparation
command are committed without including the pre-existing selection-policy edits.
