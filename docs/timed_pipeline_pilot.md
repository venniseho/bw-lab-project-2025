# Timed ten-image pipeline pilot

The pilot uses ten distinct archived source images and one existing per-instance
mask per image. `scripts/prepare_pipeline_pilot.py` chooses sorted image names
with `_ann*.png` masks and the first annotation filename for each. It does not
rank by area, shape or predicted performance, download data, duplicate a source,
or substitute union masks. The archived images/masks may reflect historical
selection filters; this is an integration/runtime pilot, not an unbiased test
set or a new segmentation-accuracy evaluation. Missing/empty/misaligned inputs
fail explicitly. The two arms use the exact same manifest.

Ten sources produce 40 fence inputs (clean/light/medium/heavy) and ten contour
inputs, with nine independent grid queries per input. Three candidates per query
would produce 1,350 raw masks, plus overlays and contact sheets. All raw masks
remain archived; output costs are part of the runtime measurements.

## Files and timing boundaries

Each arm saves `timing.json` at its root:

- `dataset_setup_seconds`: manifest/data-module construction, before lazy decoding.
- `model_load_seconds`: native model loading, CUDA transfer and provenance hashes,
  completed with a CUDA synchronization. There is one model load per arm.
- `pipeline_seconds`: DataLoader startup through evaluator finalization.
- `run_total_seconds`: run-function wall time, excluding Python process imports,
  scheduler queue time and final timing-file serialization.
- GPU name, Python/PyTorch versions, input count and CUDA synchronization flag.

`predictions/timings.csv` and each record in `predictions/results.json` contain:

| Field (seconds) | Boundary |
| --- | --- |
| `source_load_seconds` | Source dataset lookup, decoding and validation, repeated per condition. |
| `input_copy_seconds` | Copy original sample before the condition transform. |
| `transform_seconds` | Transform call, including its own copying/validation and historical-module setup. |
| `batch_wait_seconds` | Time waiting for next batch, including iterator startup on the first batch; charged to the first sample only. |
| `image_embedding_seconds` | Native image encoding with CUDA synchronization. |
| `point_query_seconds` | Sum of nine native query wall times with synchronization around each query. |
| `collection_seconds` | Embedding, queries, validation and mask packing together. |
| `inference_wall_seconds` | Whole adapter prediction, including CPU work/transfers. |
| `selection_seconds` | Existing recorded retention policy. |
| `export_seconds` | Scene images/masks/contact sheet, scene metadata and target NPZ writes; excludes updating the aggregate results manifest. |

These intervals overlap. **Do not sum all columns.** In particular, batch wait
includes source loading/transforms when `num_workers=0`, and inference wall
contains embedding/query/collection intervals. Use `pipeline_seconds` for
end-to-end pipeline wall time. Export also includes visualization work, not
just disk throughput.

The results report totals, median and p95 grouped by condition, both including
and excluding the first inference of each arm. The first-call flag is explicit
in the CSV. There is no extra warm-up inference. Ten observations per condition
and one contour realization per source are descriptive checks, not a stable
performance benchmark. No claim about segmentation robustness follows from timing.

## Run on Vulcan

Fetch branch `refactor/prompt-overlays-and-function-cleanup` and use a clean
checkout. Load the existing modules/environment:

```bash
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/env/bin/activate
python -B -m unittest discover -s tests

export PILOT_ROOT
PILOT_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/pilot10.XXXXXX)
python scripts/prepare_pipeline_pilot.py --count 10 --output-dir "$PILOT_ROOT/pilot"
```

Inspect the chosen manifest and retain the printed workspace path. In the same
shell, validate the scheduler request and submit only after CPU checks pass:

```bash
sbatch --test-only --output="$PILOT_ROOT/slurm-%j.out" \
  scripts/vulcan_pipeline_pilot.sbatch "$PWD" "$PILOT_ROOT/pilot"
JOB_ID=$(sbatch --parsable --output="$PILOT_ROOT/slurm-%j.out" \
  scripts/vulcan_pipeline_pilot.sbatch "$PWD" "$PILOT_ROOT/pilot")
```

The script requests the existing account/partition, one GPU, four CPUs, 32 GB,
and a 30-minute limit. It reuses the native SAM3 environment/checkpoint and runs
both arms sequentially; it does not submit additional jobs or install packages.
The allocation limit is not a promised runtime. Each arm must use a fresh
output directory; preserve a failed run and make a new pilot directory to retry.

After completion:

```bash
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,ExitCode
tail -n 100 "$PILOT_ROOT/slurm-${JOB_ID}.out"
python scripts/summarize_pipeline_pilot.py --pilot-dir "$PILOT_ROOT/pilot"
```

The summary checks completed states, ten unique sources per arm, 40/10 records,
and matching source sets before displaying timing summaries. Review the saved
stimuli/targets and contact sheets too. Local testing of preparation and mocked
inference does not substitute for this real 50-input GPU run.
