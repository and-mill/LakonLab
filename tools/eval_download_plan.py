"""Read-only download planning; never construct models or download payloads."""
import json
import os
from pathlib import Path


def walk(value, location='config'):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f'{location}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk(child, f'{location}.{index}')
    elif isinstance(value, str):
        yield location, value


def huggingface_entry(uri, location):
    from huggingface_hub import (
        get_hf_file_metadata, hf_hub_url, try_to_load_from_cache,
    )
    from huggingface_hub.constants import HF_HUB_CACHE

    parts = uri.removeprefix('huggingface://').split('/')
    repo_type = 'model'
    if parts[0] in ('datasets', 'spaces'):
        repo_type = {'datasets': 'dataset', 'spaces': 'space'}[parts.pop(0)]
    repo_id, filename = '/'.join(parts[:2]), '/'.join(parts[2:])
    if not filename:
        raise ValueError(f'Expected a file URI, got {uri}')
    storage = Path(HF_HUB_CACHE).expanduser().absolute() / (
        repo_type + 's--' + repo_id.replace('/', '--'))
    result = dict(source=uri, config_key=location, repo_id=repo_id,
                  repo_type=repo_type, filename=filename, cache_repository=str(storage))
    cached = try_to_load_from_cache(repo_id, filename, repo_type=repo_type)
    if isinstance(cached, str):
        snapshot = Path(cached).absolute()
        result.update(status='cached', would_download=False,
                      snapshot_path=str(snapshot), blob_path=str(snapshot.resolve()),
                      revision=snapshot.relative_to(storage).parts[1])
    else:
        # HEAD metadata only: no weights, reference data, or cache writes.
        try:
            metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename, repo_type=repo_type))
            snapshot = storage / 'snapshots' / metadata.commit_hash / filename
            result.update(status='missing', would_download=True,
                          revision=metadata.commit_hash, size_bytes=metadata.size,
                          snapshot_path=str(snapshot),
                          blob_path=str(storage / 'blobs' / metadata.etag))
        except Exception as exc:
            result.update(status='unresolved', would_download=True,
                          error=f'{type(exc).__name__}: {exc}')
    result['ref_path'] = str(storage / 'refs/main')
    result['mirror_directory'] = str(storage)
    return result


def report_downloads(cfg, args):
    entries = list(walk(cfg.to_dict()))
    if args.ckpt:
        entries.append(('args.ckpt', args.ckpt))
    downloads = []
    seen = set()
    for location, value in entries:
        if value.startswith('huggingface://') and value not in seen:
            downloads.append(huggingface_entry(value, location))
            seen.add(value)
    # InceptionMetrics supplies this URL as a constructor default.
    evaluations = cfg.get('evaluation', [])
    for evaluation in evaluations:
        if args.data and evaluation.get('data') not in args.data:
            continue
        metrics = evaluation.get('metrics', [])
        if isinstance(metrics, dict):
            metrics = [metrics]
        for metric in metrics:
            if metric.get('type') == 'InceptionMetrics':
                url = metric.get('inception_args', {}).get(
                    'inception_path', 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/'
                    'pretrained/metrics/inception-2015-12-05.pt')
                if url.startswith(('http://', 'https://')) and url not in seen:
                    cache = Path(os.environ.get('LAKONLAB_CACHE', '~/.cache/lakonlab')).expanduser()
                    target = (cache / 'downloads' / url.split('/')[-1]).absolute()
                    downloads.append(dict(source=url, status='cached' if target.is_file() else 'missing',
                                          would_download=not target.is_file(), local_path=str(target)))
                    seen.add(url)
    report = dict(
        dry_run=True, work_dir=str(Path(cfg.work_dir).expanduser().absolute()),
        experiment_directory_touched=False, downloads=downloads,
        mirror_instructions=[
            'Copy each Hugging Face mirror_directory, including blobs/, snapshots/, and refs/.',
            'Preserve symlinks or dereference them when copying; never copy broken snapshot symlinks.',
            'On the offline server set HF_HUB_CACHE to the mirrored hub directory and HF_HUB_OFFLINE=1.',
            'Copy non-Hugging-Face local_path files too; set LAKONLAB_CACHE if its root differs.',
        ],
        scope='Explicit Hugging Face file URIs plus InceptionMetrics default. '
              'Dynamic downloads in arbitrary custom models are not inferred.')
    print(json.dumps(report, indent=2))
    if any(item['status'] == 'unresolved' for item in downloads):
        raise SystemExit('Some missing files could not be resolved via metadata; see JSON errors above.')
