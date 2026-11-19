# Weekly update — 19 November 2026

## Completed / results reviewed this week

- Compared **eight decoder-only runs**: fragmentation and low/medium/high fences, each with fixed LR or warmup/cosine decay. Fixed LR achieved higher final validation IoU in all four comparisons with this schedule and seed. [Learning-rate charts and examples](../experiments/decoder-learningrate-half/index.html)
- Completed **three matched fence runs that ignore bars in the loss**, including focal, Dice and the quality-score IoU target. Reported both full-target and visible-region IoU. Final visible IoU improved, while full-target IoU decreased:

  | Fence | Visible IoU: full-loss → visible-loss training | Full IoU: full-loss → visible-loss training |
  |---|---:|---:|
  | Low | 0.7049 → 0.7068 | 0.6970 → 0.6929 |
  | Medium | 0.6717 → 0.6868 | 0.6318 → 0.5878 |
  | High | 0.6134 → 0.6435 | 0.5300 → 0.3832 |

- Completed **nine cross-condition comparisons** using final checkpoints, with the same transformed target and prompt for pretrained and fine-tuned models. All nine mean IoU changes were negative:
  - Fragmentation-trained decoder → fences: full IoU fell from **0.6065 / 0.4453 / 0.2337** to **0.0005 / 0.0020 / 0.0037** for low/medium/high fences.
  - Fence-trained decoders → fragmentation: full IoU **0.061–0.082**, versus **0.144** pretrained; visible-loss training also failed to improve transfer over the pretrained model.
  - These results suggest specialization to the training transformation. The near-zero fence predictions need visual inspection and own-condition checkpoint replay before attributing a mechanism. [Paired charts, tables and 36 contact sheets](../experiments/decoder-crosscondition-half/index.html)
- Downloaded Stage 2 summaries, timings, metadata and previews to the laptop. Checkpoints remain on Vulcan. [Low](../experiments/decoder-visible-loss/training-fence-low-visible-loss/summary.csv), [medium](../experiments/decoder-visible-loss/training-fence-medium-visible-loss/summary.csv), [high](../experiments/decoder-visible-loss/training-fence-high-visible-loss/summary.csv) fence logs are the evidence for the loss comparison above.

## Protocol and runtime

- **1008 × 1008**, batch size **1**, one epoch; **4,215,109 mask-decoder parameters** trainable. Image and prompt encoders frozen. One seeded target mask and visible foreground point per image; transforms generated in memory.
- Official **COCO train2017 → val2017**, with no independent test-set result. Training counts: fragmentation **58,633**, low **58,616**, medium **58,215**, high **56,209**. Validation counts: **4,952 / 4,952 / 4,910 / 4,739**, respectively. Fully hidden fence targets are excluded.
- Validation at **step 0**, every **5,000 updates**, and the final step: **13 passes/run**. Latest and best checkpoints saved; periodic checkpoints every **1,000 updates**. Comparisons above use final checkpoints, not independently selected best steps.
- AdamW: fixed LR **1e-5**, weight decay **0.01**, gradient clipping **1.0**. The LR comparison used 5% warmup from **1e-6 → 1e-5**, then cosine decay to **1e-6**.
- Each job requested **one L40S GPU, 12 CPU cores, 64 GB RAM and eight workers**. Downloaded Stage 2 logs show **10.90–10.99 training images/s**, **2.71–2.82 hours/run** including validation/checkpointing, and **4.13 GB peak GPU allocation**. Transfer evaluation measured **15.37–16.10 images/s**, about five minutes per validation pass. [Transfer timing CSV](../experiments/decoder-crosscondition-half/timing.csv)
- Full IoU measures overlap with the original target, including pixels behind artificial bars. Visible IoU ignores bars in both prediction and target. They are identical for our current fragmentation transform. Improved visible IoU is not evidence of improved completion behind the fence.

## Next steps

- Inspect transfer failures and replay final checkpoints on their own-condition validation sets to confirm saved final metrics.
- Plot the newly downloaded full-loss versus visible-loss learning curves and review matched before/after examples.
- Evaluate clean images to check retained ability; a clean-trained control remains outstanding.
- Archive Stage 2 checkpoints and transfer results to project storage; local reports and W&B do not archive model weights.
- If comparing full point-path versus decoder-only training directly, match the data budget and eligibility rules: the earlier full-path runs used twice as many training images.

[Detailed experiment statistics and provenance](../experiments/decoder-crosscondition-half/experiment_summary.md)
