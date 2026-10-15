"""Check paired pilot outputs and print separated wall-clock timing summaries."""
import argparse
import json
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--expect-images", type=int, default=10)
    args = parser.parse_args(argv)
    source_sets = []
    for arm, conditions in (("fence", 4), ("fragmentation", 1)):
        directory = args.pilot_dir / f"{arm}_sam3"
        report = json.loads((directory/"predictions/results.json").read_text())
        timing = json.loads((directory/"timing.json").read_text())
        assert report["status"] == "complete", f"{arm} is incomplete"
        rows = report["records"]
        sources = {row["source_key"] for row in rows}
        assert len(sources) == args.expect_images, (arm, len(sources))
        assert len(rows) == args.expect_images*conditions, (arm, len(rows))
        source_sets.append(sources)
        print(f"\n{arm}: {len(sources)} sources, {len(rows)} inputs, "
              f"{sum(r['raw_proposals'] for r in rows)} raw proposals")
        print(f"Model load: {timing['model_load_seconds']:.3f}s; "
              f"pipeline: {timing['pipeline_seconds']:.3f}s; "
              f"run total: {timing['run_total_seconds']:.3f}s")
        for name, summary in report["timing_summary_seconds"].items():
            if name == "all":
                continue
            print(f"  {name} (median / p95 ms):")
            for field in ("transform_seconds", "inference_wall_seconds", "export_seconds"):
                stats = summary[field]
                print(f"    {field}: {1000*stats['median']:.2f} / {1000*stats['p95']:.2f}")
    assert source_sets[0] == source_sets[1], "The two arms used different source sets"
    print("\nBoth arms used the same source images. Counts and statuses passed.")


if __name__ == "__main__":
    main()
