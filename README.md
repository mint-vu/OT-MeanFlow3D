# OT-MeanFlow3D

This is the supplementary code release for the paper "OT-MeanFlow3D: Bridging Optimal Transport and Meanflow for Efficient 3D Point Cloud Generation". It contains the full pipeline for training and evaluating:

1. **Shape generation** on ShapeNet chair / car / airplane: point-cloud
   autoencoder (AE) → latent context generator → MeanFlow shape generator →
   evaluation.
2. **Shape completion** on ShapeNet chair / car / airplane, from single-view
   partial point clouds (GenRe-style depth renders)

Both pipelines can run with either **vanilla MeanFlow (MF)** or
**improved MeanFlow (iMF)**, selected with the `--mf_variant` flag.

## Codebase structure

```
shape_gen/                         Training / evaluation entry points
  train_pointcloud_AE_new_data.py    Stage 1: point-cloud autoencoder
  train_latent_generator.py          Stage 2: latent context generator
  meanflow_shape_generation_AE.py    Stage 3: shape generator
  shape_generation_eval.py           Stage 4: shape-generation evaluation
  meanflow_shape_completion.py       Shape completion (joint w/ encoder)
  shape_completion_eval.py           Shape-completion evaluation
  convert_cam_params.py              GenRe-style camera-parameter prep
  pointnet_exp_models.py             Shared network architectures
  pvcnn_completion.py                PVCNN2-based point-cloud encoders
  datasets/                          ShapeNet point-cloud / single-view loaders
  modules/                           PVCNN building blocks (CUDA ops)
  metrics/                           Chamfer distance, EMD, generation metrics

meanflow/                          MeanFlow model implementations
  models/meanflow_imf.py             Improved MeanFlow (iMF)
  models/meanflow_mf.py              Vanilla MeanFlow (MF)
  models/time_sampler.py             (t, r) timestep sampling
  models/ema.py                      EMA network tracking
  models/optimal_transport.py        OT pairing (exact / sinkhorn / LOT)
```

`meanflow_imf.py` and `meanflow_mf.py` expose the same `MeanFlow` class
interface (same constructor, `forward_with_loss`, `sample`), differing only
in how the loss target is constructed. Every training/eval script picks
between them via `--mf_variant {mf,imf}` (default `imf`).

## Requirements

- Python 3.10, PyTorch (CUDA build).
- `numpy`, `matplotlib`, `tqdm`, `scipy`, `scikit-learn`, `pot` (Python
  Optimal Transport, imported as `ot`), `open3d`.
- The Chamfer distance (`shape_gen/metrics/ChamferDistancePytorch`) and EMD
  (`shape_gen/metrics/PyTorchEMD`) CUDA extensions, and the PVCNN ops
  (`shape_gen/modules/functional`), are JIT-compiled on first import via
  `torch.utils.cpp_extension.load` / `CUDAExtension` — no separate build step is required beyond having a CUDA toolchain available.
- ShapeNet point clouds (`ShapeNetCore.v2.PC15k`) for generation, which can
  be downloaded from [PointFlow](https://github.com/stevenygd/PointFlow).
- For completion, the ShapeNet single-view depth renders and camera
  parameters provided by [GenRe](https://github.com/xiumingzhang/GenRe-ShapeHD),
  following [PVD](https://github.com/alexzhou907/PVD).

Point the scripts at your local ShapeNet data either with `--dataroot_pc`
or by setting the `DATAROOT_PC` environment variable; it otherwise defaults
to `./data/ShapeNetCore.v2.PC15k/`.

## Running the shape-generation pipeline

All commands below are run from `shape_gen/`. Multi-GPU / distributed
scripts are launched with `torchrun`; `--category` is one of
`chair`, `car`, `airplane`.

```bash
# Point-cloud autoencoder
python train_pointcloud_AE_new_data.py --category chair --arch deepset

# Latent context generator
torchrun --nproc_per_node=<N> train_latent_generator.py \
    --category chair --mf_variant imf --use_ot --ot_method exact

# Shape generator
torchrun --nproc_per_node=<N> meanflow_shape_generation_AE.py \
    --category chair --mf_variant imf --use_ot --ot_method exact

# Evaluation (one step for the context generator and one for the shape generator, as in the paper)
torchrun --nproc_per_node=1 shape_generation_eval.py \
    --category chair --mf_variant imf --context_num_steps 1 --model_num_steps 1
```

All generation results in the paper use a single sampling step for both the
context generator (`--context_num_steps`) and the shape generator
(`--model_num_steps`), and both default to `1`.

**Sampling time.** The sampling times reported in the paper are measured with a batch size of 1 (one 2,048-point shape per call) and averaged over 50 runs on a single NVIDIA RTX A6000, following PSF. Each timed run includes drawing the noise, the one-step context generator and the one-step shape generator.

Stage 1 should be run with `--arch deepset` (the default).
Stages 2-4 read/write checkpoints under `./latent_gen_models/` and
`./pointnet_models/`, with filenames that encode the OT setting and
`--mf_variant` so MF and iMF runs never collide.

## Running the shape-completion pipeline

```bash
# Prepare GenRe-style camera parameters (once per dataset)
python convert_cam_params.py --dataroot GenReData/ \
    --mitsuba_xml_root GenReData/genre-xml_v2

# Train the completion model jointly with its encoder
torchrun --nproc_per_node=<N> meanflow_shape_completion.py \
    --category chair --mf_variant imf --use_ot --ot_method exact

# Evaluation
torchrun --nproc_per_node=1 shape_completion_eval.py \
    --category chair --mf_variant imf
```

Completion checkpoints are written under `./pointnet_models/`, likewise
named with `--mf_variant` folded in.

## Acknowledgements

Our code builds on the following projects, and we thank the authors
for their contributions:

- [PVD](https://github.com/alexzhou907/PVD/tree/main): data loading,
  evaluation metrics, and the PVCNN-based modules.
- [MeanFlow](https://github.com/Gsunshine/meanflow): the MeanFlow formulation.
- [MeanFlow PyTorch](https://github.com/Gsunshine/py-meanflow): the PyTorch
  MeanFlow implementation.
