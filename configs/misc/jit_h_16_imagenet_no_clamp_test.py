# JiT-H/16 ADM evaluation without the low-noise denominator clamp.
# Keep the checkpoint, 50-step Heun sampler, guidance, and 50K samples
# identical to the README's 1.90 FID baseline.
_base_ = ['./jit_h_16_imagenet_test.py']

name = 'jit_h_16_imagenet_no_clamp_test'
work_dir = f'work_dirs/{name}'

# u_hat = (x_t - x0_hat) / sigma, rather than / max(sigma, 0.04).
# The inherited sampler evaluates only positive sigmas and ends with an
# Euler step to zero, so disabling this clamp does not divide by zero.
model = dict(diffusion=dict(sigma_min=0.0))
