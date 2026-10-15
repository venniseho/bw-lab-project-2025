# Refactor validation

2026-10-15, Windows CPU, isolated `.venv` with system site packages. Command:

```sh
python -B -m unittest discover -s tests -v
```

Result: **93 tests passed**, no skips, 12.503 seconds on the final full run.
31 stimulus/dataset/label/architecture tests are new; 62 existing regression
tests are reused. The earlier 83-test suite also passed after relocation into
the canonical library.

| Suite | Tests |
| --- | ---: |
| COCO selection | 14 |
| Fence functional | 10 |
| Prompt/target preparation | 12 |
| SAM3 smoke, mocked model | 10 |
| Scene proposals and native-loader regression | 16 |
| New stimulus transforms | 13 |
| New datasets / CPU preparation / model boundary | 6 |
| New contact-sheet labels / archive rebuild | 2 |
| Library/experiment composition and architecture | 10 |
| Total | 93 |

The frozen fragmenter Git blob hash is checked in the suite. All five archive
source files were separately compared byte-for-byte to `27f9bd6`. Four root
historical Python files were compared to main and remain identical. New module
syntax was parsed using Python's 3.12 grammar; a real Python 3.12 runtime was not
available locally. Whitespace checks exclude the frozen archive sources, which
retain historical trailing whitespace and mixed line endings by design;
`.gitattributes` prevents newline normalization of those exact copies. New and
imported active code is checked with `git diff --check`.

Environment: Python 3.13.5, torch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.2.6,
Pillow 12.0.0, OpenCV 4.12.0. This is local test evidence, not a deployment lockfile.
The existing Vulcan Python 3.12 environment remains unchanged.

One earlier custom test launcher from stdin failed Windows worker spawning
because `<stdin>` cannot be imported by child processes. The documented module
invocation passed, including `num_workers=1`. A label test initially had an
incorrect hand-calculated rectangle area (4,800 instead of 5,400); correcting the
expected value passed against the measured mask. Neither required algorithm changes.
pycocotools emits a NumPy copy-keyword deprecation warning; its real polygon/RLE
test passes. No warnings were suppressed.

The contact-sheet layout was visually inspected on an explicitly synthetic
preview. This is a rendering check, not an inferred SAM3 result. Existing local
natural-image sheets lack the complete archive needed for faithful rebuilding.

No GPU jobs, native SAM3 changes, training, or new quantitative experiments were
run. Software regression success does not validate scientific assumptions.

## Canonical library and experiment validation

The pinned reference's complete library/experiment layout, model base, trainer,
README and launcher were reviewed in addition to the six original files. The
actual implementation now lives under `bwlab/`, and Config/run wiring under
`experiments/sam3_fragmentation/`; previous paths are compatibility imports.

The new integration suite exercises named train/val/test loaders, source-image
split leakage rejection, independent severity expansion with unique sample
keys, CPU preparation of both arms, the shared Runner with the native SAM3 fake,
the full experiment `run()` with mocked CUDA/model loading, login-node rejection,
relative config paths and the library's one-way dependency boundary. It checks
actual exported candidate files, contact sheets, original masks and result
manifests. This is not a new quantitative segmentation experiment.

Editable package installation succeeded with
`python -m pip install --no-deps --no-build-isolation -e .` in the local venv.
Imports from outside the worktree resolved to that installed library. No
dependencies were installed or upgraded for this structural revision. New
library and experiment files all parse with Python 3.12 grammar, and the five
frozen archive files remain byte-identical.

The experiment CLI's CPU `prepare` command was also run on the preserved fence
demo pair. It generated clean/light/medium/heavy inputs, separate targets and
config/manifest files under ignored `outputs/review/structured_fence/`. No model
was loaded. Both the old smoke CLI and new experiment expose working `--help`.
