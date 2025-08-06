# 3D SliceGAN (slice_gan_3d)

A 3D Slice-based Generative Adversarial Network that generates volumetric engineering designs by learning 2D slice distributions and assembling them into coherent 3D volumes.

## Overview

This implementation uses the SliceGAN approach to generate 3D engineering designs by training on 2D cross-sectional slices extracted from 3D volumes. The model learns by evaluating individual 2D slices that are formed from generated 3D volumetric structures.

## Architecture

### Components
- **Generator2D**: 2D CNN that generates individual cross-sectional slices from noise + conditions + slice position
- **Discriminator2D**: 2D CNN that classifies real vs. generated slices with condition and position awareness
- **Slice disassembly**: Stacks generated 2D slices from complete 3D volumes

### Key Features
- **Memory Efficient**: Uses 2D operations instead of memory-intensive 3D convolutions
- **Flexible Resolution**: Can generate different slice counts and resolutions
- **WGAN-GP Training**: Stable adversarial training with gradient penalty

## Quick Start

### Basic Training
```bash
# Train with default parameters
python slice_gan_3d.py --problem_id heatconduction3d --n_epochs 300

# Save model for later evaluation
python slice_gan_3d.py --save_model --track

# Memory-optimized training (can run on smaller GPUs)
python slice_gan_3d.py --batch_size 16 --n_slices 32
```

### Evaluation
```bash
# Evaluate trained model using EngiBench metrics
python evaluate_slice_gan_3d.py --seed 1 --n_samples 50

# Batch evaluation across multiple seeds
for seed in {1..5}; do
    python evaluate_slice_gan_3d.py --seed $seed --n_samples 100
done
```

### Hyperparameter Sweep
```bash
# Run Bayesian optimization sweep
wandb sweep sweep_slice_gan_3d.yaml
wandb agent <sweep_id>
```
!! Change the path of slice_gan_3d.py to your project path

## Key Parameters

### Core Training
- `--n_epochs 300`: Number of training epochs
- `--batch_size 32`: Batch size (can be larger due to 2D operations)
- `--lr_gen 0.0002`: Generator learning rate
- `--lr_disc 0.0002`: Discriminator learning rate

### Architecture
- `--latent_dim 64`: Dimensionality of noise vector

### SliceGAN Specific
- `--slice_sampling_ratio 0.3`: proportion of slices used for training


### References
Original SliceGAN: https://arxiv.org/abs/2102.07708