"""Indexed local WebDataset shards for the existing distributed sampler."""

import hashlib
import json
import os
import re
import tarfile
from pathlib import Path

import numpy as np

from lakonlab.utils import locked_cache_path
from lakonlab.utils.io_utils import LAKONLAB_CACHE


class ImageNetShardIndex:
    # Small fixed-size records; the image payload stays in the original tar.
    dtype = np.dtype([
        ('shard', '<i4'), ('offset', '<i8'), ('size', '<i8'),
        ('label', '<i4'), ('ext', 'S5'),
    ])

    def __init__(self, root, split='train', cache_dir=None):
        root = Path(root).expanduser().resolve()
        split_root = root / split
        search_root = split_root if split_root.is_dir() else root
        shards = sorted(search_root.rglob('*.tar'))
        # Also support flat directories with train-*.tar / validation-*.tar.
        if search_root == root:
            def shard_split(path):
                for part in path.relative_to(root).parts:
                    match = re.match(r'^(train|validation|val|test)(?:[-_.]|$)', part)
                    if match:
                        return match[1]
                return None
            if any(shard_split(path) is not None for path in shards):
                aliases = {'val', 'validation'} if split in ('val', 'validation') else {split}
                shards = [path for path in shards if shard_split(path) in aliases]
        if not shards:
            raise FileNotFoundError(
                f'No uncompressed .tar shards for split {split!r} under {root}. '
                'Point IMAGENET_ROOT at the WebDataset root or its split directory.')
        self.shards = [str(path) for path in shards]
        fingerprint = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                       for path in shards]
        digest = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
        cache_dir = Path(cache_dir or os.path.join(LAKONLAB_CACHE, 'imagenet_wds'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = str(cache_dir / f'v1-{digest}.npy')
        with locked_cache_path(self.index_path) as (local_path, exists):
            if not exists:
                records = self._scan()
                with open(local_path, 'wb') as handle:
                    np.save(handle, np.asarray(records, dtype=self.dtype), allow_pickle=False)
        self.records = np.load(self.index_path, mmap_mode='r', allow_pickle=False)

    def _scan(self):
        records = []
        for shard_id, shard in enumerate(self.shards):
            images, labels = {}, {}
            # r: deliberately excludes compressed tars: offsets must address raw bytes.
            with tarfile.open(shard, 'r:') as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    key, ext = os.path.splitext(member.name)
                    ext = ext.lower()
                    if ext in ('.jpg', '.jpeg', '.png'):
                        if key in images:
                            raise ValueError(f'Duplicate image key {key!r} in {shard}')
                        images[key] = (member.offset_data, member.size, ext)
                    elif ext in ('.cls', '.json'):
                        payload = archive.extractfile(member).read()
                        if ext == '.json':
                            metadata = json.loads(payload)
                            payload = next((metadata[k] for k in ('label', 'class_id', 'class_idx', 'target')
                                            if k in metadata), None)
                            if payload is None:
                                continue
                        try:
                            label = int(payload)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f'Invalid label for {key!r} in {shard}') from exc
                        if not 0 <= label < 1000:
                            raise ValueError(f'Expected zero-based ImageNet label 0..999, got {label} in {shard}')
                        if key in labels and labels[key] != label:
                            raise ValueError(f'Conflicting labels for {key!r} in {shard}')
                        labels[key] = label
                    archive.members.clear()
                for key, (offset, size, ext) in images.items():
                    if key not in labels:
                        raise ValueError(f'Missing .cls or JSON label for {key!r} in {shard}')
                    records.append((shard_id, offset, size, labels[key], ext.encode()))
        if not records:
            raise ValueError('No image/label pairs found in ImageNet shards')
        return records

    def __len__(self):
        return len(self.records)

    def read(self, index):
        record = self.records[index]
        shard = self.shards[int(record['shard'])]
        offset, size = int(record['offset']), int(record['size'])
        # Each access opens its own handle, safe across forked/spawned workers.
        with open(shard, 'rb') as handle:
            handle.seek(offset)
            payload = handle.read(size)
        if len(payload) != size:
            raise OSError(f'Truncated shard {shard} at offset {offset}')
        path = f"{shard}:{offset}{record['ext'].decode()}"
        return payload, int(record['label']), path

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('records')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.records = np.load(self.index_path, mmap_mode='r', allow_pickle=False)
