# Plan from the 2 November meeting

## Plot and preserve the current baseline

- Plot batch-size throughput for both paths from the existing CSVs. Record the
  run revision, GPU, worker count, evaluated inputs and selection policy.
- Existing 512-source numbers used revision 5ca6426, two workers, four requested
  CPU cores, one noncrowd instance per source, and preview export disabled.
  Fence processed 2,048 inputs; fragmentation processed 512. These results
  precede packed GPU input preparation, revision 8f665a9. Keep them separate.

## Next optimization experiment

- Request 12 CPU cores per GPU; start with eight loader workers. Compare two,
  eight and twelve workers while holding image batch size, sources, code,
  transforms, warm-up and output settings fixed. Workers are loader subprocesses;
  they do not need to equal CPU cores, and more is not automatically faster.
- First validate the packed-preparation change on the GPU. Benchmark it separately
  from worker-count changes so each experiment changes one thing.
- Track evaluation throughput, data wait, GPU utilization, CPU utilization,
  host RAM and GPU memory. Existing logs contain GPU monitoring; W&B integration
  is still to be implemented. Keep offline/local logging available on Vulcan.
- Correlate utilization oscillations with data-wait measurements. Oscillation
  alone does not establish a loader problem.
- Compare resource use per input rather than assuming that similar overall
  rates imply identical fence and fragmentation workloads.
- Smaller source crops do not necessarily reduce encoder cost: the current
  native processor resizes inputs to a fixed model resolution. Separate source
  cropping from changing actual encoder resolution; either changes the experiment
  and must preserve aligned masks and prompt coordinates. Retain full-image eval.

## Training reference, then targeted experiments

Reference supplied in the meeting:
https://github.com/Applied-AI-Research-Lab/The-SAM2-to-SAM3-Gap-in-the-Segment-Anything-Model-Family/blob/main/docs/training_guide.md

1. Pin a reference repository revision, inspect the executable training code,
   and identify a specific reported metric, checkpoint, dataset split and recipe
   to reproduce. Run its documented example before changing the data.
2. Check task compatibility: the guide describes text/concept-driven SAM3
   training, while the current BWLab evaluation uses independent point prompts
   through the interactive predictor. Confirm which modules and losses apply;
   a text-detector training example is not automatically an interactive decoder
   fine-tuning implementation. Treat advertised runtimes as claims to measure.
3. Record training wall time, steps/epochs, VRAM, trainable parameter names/counts,
   loss and held-out segmentation metrics. First verify loss reduction on a tiny
   training subset; reserve COCO validation for evaluation, not training.
4. Plug the existing transforms into training in memory, changing one factor at
   a time. Keep clean and transformed held-out evaluation conditions.
5. Distinguish three possible ablations explicitly:
   - decoder-only: freeze the encoder, train the mask decoder;
   - encoder-only: freeze the decoder, train the encoder;
   - full model: train both, subject to the selected architecture.
   The notes mention both freezing and fine-tuning the decoder; these are different
   experiments. Specify the intended comparison before allocating long jobs.
6. Equal gains from full-model and frozen-decoder training would suggest the
   frozen decoder is not limiting that particular setup; it would not prove the
   decoder is universally sufficient. Failed training also requires checking
   optimization, prompts, labels and data before concluding architecture failure.
7. After reproducing the reference, write a small explicit training loop with
   optimizer/zero_grad/backward/step, validation, checkpointing and resource logs.

These are next steps, not completed training or W&B integration.
