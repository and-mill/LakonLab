# ImageNet WebDataset input

All ImageNet configs under `configs/` use one dataset setting:

```bash
export IMAGENET_ROOT=/path/to/timm_imagenet_1k_wds
```

This works for both training and testing. Without this variable the default is
`data/timm_imagenet_1k_wds`. For `tools/test.py`, a command-line override is also
available:

```bash
torchrun --standalone --nproc_per_node=4 tools/test.py \
  configs/asymflow/asymflow_h_16_r8_repa_imagenet_test.py \
  --imagenet-root /path/to/timm_imagenet_1k_wds --launcher pytorch
```

## Generation-only evaluation

`ImageNet(test_mode=True)` generates class labels and seeded noise. It reads no
images, shards, or external class-list files, so no dataset path is required.
Class names are bundled from torchvision's ImageNet-1K categories (the
`ResNet50_Weights` metadata); numerical label IDs and noise seeds are unchanged.
An explicit `label2name_path` still overrides names.

The AsymFlow 50K configs use precomputed ADM reference features for FID. They
still require the model checkpoint, reference features, and Inception model;
these download on first use if not cached. A compute node without internet
access needs those assets cached beforehand.

## Reading real images

With `test_mode=False` and `datalist_path=None`, `ImageNet` reads local,
uncompressed `.tar` shards directly. Supported layouts include `train/*.tar`,
flat `train-*.tar` / `validation-*.tar` files, and a directory containing only
the desired split's shards. The dataset's `split` defaults to `train`.
Use `split='validation'` with a validation directory or flat validation shards.

Each sample must have a `.jpg`, `.jpeg`, or `.png` member and a member with the
same key containing its label: either plain integer `.cls`, or `.json` with
`label`, `class_id`, `class_idx`, or `target`. Labels must use the standard
zero-based ImageNet-1K ordering, 0 through 999. Synset-only labels and compressed
tars need conversion first; labels are never guessed from image filenames.
Inspect a shard to confirm the dataset's actual schema if loading reports an
unsupported layout or missing labels.

The initial load scans tar headers and labels once, then caches an index of
image byte offsets. Images remain in their original shards; no extraction is
needed. The memory-mapped index preserves dataset length, indexed access, and
the existing distributed sampler. Changing a shard's path, size or modification
time invalidates the cache. Index construction is locked across processes.

The default cache is `$LAKONLAB_CACHE/imagenet_wds`, or
`~/.cache/lakonlab/imagenet_wds` if `LAKONLAB_CACHE` is unset. It must be writable;
set `shard_index_cache_dir` in the dataset config to override it. Sharing this
directory across nodes avoids repeated scans. Random image access can have
different I/O performance from sequential WebDataset streaming.

Raw-image training uses the model's existing VAE encoding path rather than
precomputed latents, adding encoding work. Existing image or latent lists remain
supported by explicitly supplying both `data_root` and `datalist_path`.

## Tests

```bash
python -m unittest discover -s tests -p 'test_imagenet_wds.py' -v
```

The tests use tiny synthetic shards, including worker processes. Full model
evaluation and the contents of your local ImageNet download require a separate
run.
