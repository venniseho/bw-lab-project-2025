# Refactor validation

2026-10-15, Windows CPU, isolated `.venv` with system site packages. Command:

```sh
python -B -m unittest discover -s tests -v
```

Result: **83 tests passed**, no skips, 9.520 seconds on the final full run.
21 stimulus/dataset/label tests are new; 62 existing regression tests are reused.

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
| Total | 83 |

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
