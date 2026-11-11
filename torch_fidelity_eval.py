"""Evaluate saved images with the LTH14 torch-fidelity fork and reference statistics.

Example:
    python torch_fidelity_eval.py experiments/asymflow_h_16_r8_repa_imagenet_test

The reference must be compatible with the installed torch-fidelity implementation
and its Inception feature extractor; the repository's StyleGAN reference pickle
is not a drop-in replacement.
"""

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
# ImageNet reference statistics.
FID_STATISTICS_FILE = Path('/home/mlrandwm/git/LakonLab/fid_stats/jit_in256_stats.npz')
DEFAULT_EXPERIMENT = (
    REPO_ROOT / 'experiments/asymflow_h_16_r8_repa_imagenet_test'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', type=Path, default=DEFAULT_EXPERIMENT,
                        help='Experiment root; images are read from images/default (default: %(default)s).')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of CUDA.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    experiment = args.experiment.expanduser().resolve()
    images = experiment / 'images/default'
    output = experiment / 'torch_fidelity_results.json'
    if not images.is_dir():
        parser.error(f'Image folder does not exist: {images}')
    if not FID_STATISTICS_FILE.is_file():
        parser.error(f'Reference statistics file does not exist: {FID_STATISTICS_FILE}. '
                     'Set FID_STATISTICS_FILE in this script to your reference statistics path.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    try:
        import torch_fidelity
    except ImportError as exc:
        parser.exit(1, f'Cannot import torch_fidelity: {exc}\n'
                       'Install it in your evaluation environment with pip install git+https://github.com/LTH14/torch-fidelity.git.\n')

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=str(images),
        input2=None,
        fid_statistics_file=str(FID_STATISTICS_FILE),
        cuda=not args.cpu,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=args.verbose,
        batch_size=args.batch_size,
    )
    result = json.dumps({key: float(value) for key, value in metrics_dict.items()},
                        indent=2, allow_nan=False)
    print(result)
    output.write_text(result + '\n', encoding='utf-8')
    print(f'Saved metrics to {output}')


if __name__ == '__main__':
    main()
