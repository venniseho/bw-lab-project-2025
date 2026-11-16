#!/bin/bash
# Copy only completed Stage 1 runs; preserve scratch originals and active runs.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: archive_point_stage1.sh plan-root archive-directory' >&2; exit 2; }
root=$(realpath "$1")
archive=$(realpath -m "$2")
[[ "$archive" != "$root" && "$archive" != "$root/"* ]] || {
  echo 'Archive destination must be outside the scratch run root' >&2; exit 2;
}
command -v rsync >/dev/null || { echo 'rsync is required for resumable copies' >&2; exit 2; }
conditions=(fragmentation fence-low fence-medium fence-high
  fragmentation-warmup-cosine fence-low-warmup-cosine
  fence-medium-warmup-cosine fence-high-warmup-cosine)

[[ -d "$root/prepared" ]] || { echo "Missing prepared labels: $root/prepared" >&2; exit 2; }
for condition in "${conditions[@]}"; do
  run="$root/training-$condition"
  for required in complete.json checkpoint.pt checkpoint-best.pt; do
    [[ -f "$run/$required" ]] || { echo "Missing completed output: $run/$required" >&2; exit 2; }
  done
done

mkdir -p "$archive"
rsync -a --info=progress2 "$root/prepared/" "$archive/prepared/"
for condition in "${conditions[@]}"; do
  echo "Archiving completed run: $condition"
  rsync -a --info=progress2 "$root/training-$condition/" "$archive/training-$condition/"
  for logfile in "$root/$condition-"[0-9]*.out; do
    if [[ -f "$logfile" ]]; then rsync -a "$logfile" "$archive/"; fi
  done
done
if [[ -d "$root/report-stage1" ]]; then
  rsync -a --info=progress2 "$root/report-stage1/" "$archive/report-stage1/"
fi
echo "Archive finished: $archive"
