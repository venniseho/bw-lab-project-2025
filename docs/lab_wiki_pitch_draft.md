# Lead

A fence can split one animal into many visible pieces. We want to understand
when SAM3 groups those pieces into a coherent object and when it merges them
with the fence or treats them separately.

# Pitch

Connect a practical occlusion problem to controlled contour-fragmentation
research. Compare natural images, synthetic fences, and sparse fragmented
outlines with matched noise, then investigate improvements supported by those
measurements. This is an additive wiki draft for review, not a replacement of
the existing wiki.

# Research question

How does fragmentation and occlusion affect object segmentation in SAM3, and can we improve its robustness?

# Background

Prior work in this project generated fragmented COCO object contours with
background segments and explored SAM prompting and proposal ranking. The
historical report provides motivation; its strongest causal interpretations
need further controlled evidence. Fences extend the investigation to partial
occlusion in natural images while retaining controlled geometry.

# Why?

Robust segmentation should remain useful when object evidence is interrupted
by fences or foliage. Controlled corruption can reveal which failures reflect
loss of visible evidence, proposal selection, or difficulty grouping parts.

# Why us?

The project combines an existing fragmented-contour implementation, prior
experiments, a working native SAM3 installation on Vulcan, and lab expertise
in perceptual grouping. These provide a practical basis for carefully paired
tests without changing several components at once.

# Approach

Use Dataset → transforms → SAM3 → evaluator → results. Keep the original
fragmented-outline algorithm with noise as one arm and deterministic synthetic
fences as another. Compare matched source images across clean/light/medium/heavy
occlusion. Retain source, visible and occlusion masks separately. Record prompt
and mask-selection policies, with oracle best-mask scores explicitly labeled
upper bounds. Later, probe vision representations and mask decoding.

# Status

As of 2026-10-15, native SAM3 already works on Vulcan. Previous automatic grid
prompting produced qualitative proposals on seven natural examples in 49 s.
Some donkey proposals detect animal parts or combine animal and fence; a shadowed
portrait tests illumination rather than physical occlusion. Synthetic fence
generation already worked on CPU. The current branch prepares a non-destructive
refactor with both stimulus methods, independent transforms, tests and clearer
contact-sheet captions. New controlled quantitative SAM3 experiments and
encoder/decoder investigations have not been completed.

# Next step

Review the preservation inventory and proposed changes to main, validate the
refactor, then merge only after explicit approval. Branch from updated main
to fix the quantitative protocol, select source-image splits and targets, and
request an approved SLURM smoke run on matched clean/occluded inputs.

# Links

- Repository: https://github.com/venniseho/bw-lab-project-2025
- Working branch: `sam3-fragmentation-v2`
- Known working SAM3 commit: `b8aa1e860004da4cb305559f989fc6ed98612b87`
- Research log: https://docs.google.com/document/d/1Jh1bZ4EN473fMY0mR8LZzR4dBOX_XixAOIBk6ac92lc
- Prior report: https://docs.google.com/document/d/1mQrYGKPKZ44tVxhy-GtNIQdZE1FF7ZA91ec0WD_aSHA
- Pinned architecture reference: https://github.com/mtangemann/mlvbench/tree/c204a8185fb662b3b1edacfb7a98f7b36fbd9a70
- Native SAM3: https://github.com/facebookresearch/sam3
- Installed Meta SAM3 revision: `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`
- Related robustness code: https://github.com/hendrycks/robustness
