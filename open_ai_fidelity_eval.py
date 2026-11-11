"""Run the unchanged official OpenAI evaluator on an experiment's saved PNGs.

Setup (isolated to preserve the training environment):
    uv venv --python .venv/bin/python cache_openai_eval/venv
    uv pip install --python cache_openai_eval/venv/bin/python -r requirements-openai-eval.txt
Run:
    cache_openai_eval/venv/bin/python open_ai_fidelity_eval.py [experiment]
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'third_party/openai_guided_diffusion'
CACHE = ROOT / 'cache_openai_eval'
REFERENCE_URL = ('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/'
                 'ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz')

REFERENCE_SHA256 = 'b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_reference():
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / 'VIRTUAL_imagenet256_labeled.npz'
    if not target.exists():
        temporary = target.with_suffix('.download')
        print(f'Downloading official reference: {REFERENCE_URL}', flush=True)
        try:
            urllib.request.urlretrieve(REFERENCE_URL, temporary)
            with zipfile.ZipFile(temporary) as archive:
                if 'arr_0.npy' not in archive.namelist():
                    raise ValueError('Reference batch is missing arr_0')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(target) != REFERENCE_SHA256:
        raise ValueError(f'Official reference checksum mismatch: {target}')
    return target


def package_images(images, destination):
    """Stream an NPY member into an NPZ without holding all 50K images in RAM."""
    import numpy as np
    from PIL import Image

    temporary = destination.with_suffix('.partial')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            with archive.open('arr_0.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array_header_1_0(member, {
                    'descr': np.dtype('uint8').str, 'fortran_order': False,
                    'shape': (len(images), 256, 256, 3),
                })
                for index, path in enumerate(images):
                    with Image.open(path) as image:
                        if image.size != (256, 256):
                            raise ValueError(f'{path}: expected 256x256, got {image.size}')
                        array = np.asarray(image.convert('RGB'), dtype=np.uint8)
                        member.write(array.tobytes())
                    if (index + 1) % 1000 == 0:
                        print(f'Packed {index + 1}/{len(images)} images', flush=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', type=Path,
                        default=ROOT / 'experiments/asymflow_h_16_r8_repa_imagenet_test')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Package images and download reference without running evaluation.')
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    images_dir = experiment / 'images/default'
    images = sorted(images_dir.glob('*.png'))
    if len(images) != 50000:
        parser.error(f'Final evaluation requires exactly 50000 PNGs in {images_dir}; found {len(images)}')
    if any(not re.fullmatch(r'\d{9}\.png', path.name) for path in images):
        parser.error('Expected nine-digit PNG names, e.g. 000000000.png')
    reference = download_reference()
    samples = experiment / 'open_ai_generated_samples.npz'
    package_images(images, samples)
    evaluator = VENDOR / 'evaluator.py'
    command = [sys.executable, '-u', str(evaluator), str(reference), str(samples)]
    metadata = {
        'evaluator_repository': 'https://github.com/openai/guided-diffusion',
        'evaluator_commit': (VENDOR / 'COMMIT').read_text().strip(),
        'evaluator_sha256': sha256(evaluator),
        'reference_url': REFERENCE_URL,
        'reference_sha256': sha256(reference),
        'generated_samples_sha256': sha256(samples),
        'num_images': len(images), 'command': command,
        'package_versions': {name: version(name) for name in
                             ('tensorflow', 'numpy', 'scipy', 'Pillow', 'requests', 'tqdm')},
    }
    (experiment / 'open_ai_fidelity_provenance.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    if args.prepare_only:
        return
    log_path = experiment / 'open_ai_fidelity_eval.log'
    metrics = {}
    # Run the upstream script directly; its graph/preprocessing/math are unchanged.
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=CACHE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
            match = re.fullmatch(r'(FID|sFID|Inception Score|Precision|Recall):\s*(\S+)\s*', line)
            if match:
                metrics[match[1]] = float(match[2])
        returncode = process.wait()
    if returncode or 'FID' not in metrics:
        raise RuntimeError(f'Official evaluator failed (exit {returncode}); see {log_path}')
    metadata['metrics'] = metrics
    metadata['log'] = str(log_path)
    (experiment / 'open_ai_fidelity_results.json').write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(f"Official OpenAI FID: {metrics['FID']}")


if __name__ == '__main__':
    main()
