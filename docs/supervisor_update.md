# Supervisor update draft

I have a working end-to-end pipeline for testing SAM3 on fragmented and
occluded inputs. It follows the organization of your mlvbench code: reusable
datasets, stimulus transforms, model adapters and tasks are separate from the
experiment configuration and run function. I preserved both the historical
fragmented-contour method with noise and the synthetic fence method, and kept
the original target separate from the visible target after adding a fence.

All 93 software tests passed on Vulcan with Python 3.12.4. A real GPU smoke test
then processed one source image in five conditions: clean, light/medium/heavy
fence occlusion, and fragmented contours. Both arms completed successfully on
an L40S GPU in a 68-second job, producing 27 proposals per condition. This
validates the pipeline integration, not segmentation accuracy or robustness.
The workflow uses nine independent positive-point grid queries, rather than
prompt-free segmentation.

The validated refactor is merged into main. I am now making the visualizations
easier to interpret by showing prompt dots on the input and candidate overlays,
and shortening functions so each has one responsibility. Next I want to fix a
small paired experiment protocol: select source images, keep prompt and mask
selection rules explicit, and evaluate visible segmentation separately from
recovery of the pre-fence target. Oracle best-mask IoU will be reported only as
an upper bound. Ordinary COCO masks alone will not establish genuine amodal
completion behind pre-existing natural occluders.

## Evidence and next code steps

- Merged baseline: https://github.com/venniseho/bw-lab-project-2025/pull/3
- Architecture: [library/experiment correspondence](architecture.md).
- Existing Vulcan job: `1762018`, completed with exit code `0:0`, total 68 s;
  fence step 34 s and contour step 27 s. Times include model loading/export and
  are not isolated inference benchmarks.
- Original copied results remain unchanged. New contact sheets are regenerated
  from saved masks and metadata on CPU; no new SAM3 inference is needed.
- Review the overlay/function-cleanup follow-up separately, run its CPU suite
  on Vulcan, and merge it after review. No quantitative experiment has been
  completed yet.
- Keep further infrastructure proportional to the experiment: add metrics,
  scheduling or caching only when a defined protocol needs them. Preserve
  compatibility imports until downstream scripts have migrated; do not remove
  historical source or copy the reference's full training framework.
