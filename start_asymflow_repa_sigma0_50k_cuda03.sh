WORK_DIR="experiments/asymflow_h_16_r8_repa_imagenet_test"

torchrun --standalone --nnodes=1 --nproc_per_node=4 tools/test.py \
  configs/asymflow/asymflow_h_16_r8_repa_imagenet_test.py \
  --launcher pytorch --diff_seed \
  --cfg-options \
  model.diffusion.denoising.sigma_min=0.0 \
  work_dir="$WORK_DIR" \
  evaluation.0.viz_dir="$WORK_DIR/images" \
  evaluation.0.viz_num=None

