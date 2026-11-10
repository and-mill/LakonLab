import io
import json
import os
import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from mmcv import Config

from lakonlab.datasets import ImageNet, build_dataset, build_dataloader
from lakonlab.datasets.imagenet_wds import ImageNetShardIndex


def write_shard(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w') as archive:
        for name, payload in samples:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


class ImageNetWebDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        buf = io.BytesIO()
        Image.new('RGB', (12, 10), (255, 0, 0)).save(buf, format='PNG')
        self.image = buf.getvalue()

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return ImageNet(data_root=str(self.root), image_size=8, random_flip=False,
                        shard_index_cache_dir=str(self.root / 'cache'), **kwargs)

    def test_generation_needs_no_files_and_preserves_seeds(self):
        with patch.dict(os.environ, {'IMAGENET_ROOT': '/missing/imagenet'}):
            dataset = ImageNet(test_mode=True, test_label_sampling='equal',
                               test_label_repeat=50, latent_size=(3, 8, 8))
        self.assertEqual(len(dataset), 50000)
        self.assertEqual(dataset[49]['labels'].item(), 0)
        self.assertEqual(dataset[50]['labels'].item(), 1)
        self.assertEqual(dataset[49999]['labels'].item(), 999)
        self.assertEqual(dataset[0]['name'].data, 'tench')
        self.assertTrue(torch.equal(dataset[0]['noise'],
                                   torch.randn((3, 8, 8), generator=torch.Generator().manual_seed(1000))))
        self.assertNotIn('images', dataset[0])

    def test_shards_labels_preprocessing_cache_and_workers(self):
        write_shard(self.root / 'train' / 'a.tar', [
            ('a.cls', b'7'), ('a.png', self.image),
            ('b.png', self.image), ('b.json', json.dumps({'label': 999}).encode()),
        ])
        write_shard(self.root / 'validation' / 'a.tar', [('x.png', self.image), ('x.cls', b'3')])
        dataset = self.dataset(return_image_bytes=True)
        self.assertEqual(len(dataset), 2)
        self.assertEqual([dataset[i]['labels'].item() for i in range(2)], [7, 999])
        self.assertEqual(dataset[0]['images'].shape, (3, 8, 8))
        self.assertTrue(torch.equal(dataset[0]['images'][0], torch.ones(8, 8)))
        self.assertEqual(dataset[0]['image_bytes'].data, self.image)
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertTrue(torch.equal(dataset[1]['images'], restored[1]['images']))
        with patch.object(ImageNetShardIndex, '_scan', side_effect=AssertionError('cache missed')):
            self.assertEqual(len(self.dataset()), 2)
        for context in ('fork', 'spawn'):
            loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                                      shuffle=False, multiprocessing_context=context, timeout=30)
            self.assertEqual([batch['labels'].item() for batch in loader], [7, 999])

    def test_flat_split_discovery_and_environment(self):
        write_shard(self.root / 'train-000.tar', [('a.png', self.image), ('a.cls', b'4')])
        write_shard(self.root / 'validation-000.tar', [('a.png', self.image), ('a.cls', b'8')])
        with patch.dict(os.environ, {'IMAGENET_ROOT': str(self.root)}):
            dataset = ImageNet(shard_index_cache_dir=str(self.root / 'cache'))
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['labels'].item(), 4)
        validation = self.dataset(split='validation')
        self.assertEqual(validation[0]['labels'].item(), 8)

    def test_invalid_label_and_missing_label(self):
        shard = self.root / 'a.tar'
        write_shard(shard, [('a.png', self.image), ('a.cls', b'1000')])
        with self.assertRaisesRegex(ValueError, 'zero-based'):
            self.dataset()
        write_shard(shard, [('a.png', self.image)])
        with self.assertRaisesRegex(ValueError, 'Missing .cls'):
            self.dataset()

    def test_legacy_image_list_and_custom_names(self):
        (self.root / 'a.png').write_bytes(self.image)
        (self.root / 'list.txt').write_text('a.png 7\n')
        (self.root / 'names.txt').write_text('7: "custom name",\n')
        dataset = self.dataset(datalist_path=str(self.root / 'list.txt'),
                               label2name_path=str(self.root / 'names.txt'))
        self.assertEqual(dataset[0]['labels'].item(), 7)
        self.assertEqual(dataset[0]['name'].data, 'custom name')

    def test_all_imagenet_configs_load_without_external_data(self):
        count = 0
        for path in Path('configs').rglob('*.py'):
            if "type='ImageNet'" not in path.read_text():
                continue
            cfg = Config.fromfile(str(path))
            for name, entry in cfg.data.items():
                if isinstance(entry, dict) and entry.get('type') == 'ImageNet':
                    self.assertIsNone(entry['data_root'], str(path))
                    self.assertIsNone(entry['datalist_path'], str(path))
                    if entry.get('test_mode'):
                        self.assertEqual(len(build_dataset(entry)), 50000)
            count += 1
        self.assertEqual(count, 18)

    def test_cli_root_override(self):
        from tools.test import override_imagenet_root
        cfg = {'val': {'type': 'ImageNet', 'datalist_path': 'old.txt'},
               'train': {'type': 'Wrapper', 'datasets': [{'type': 'ImageNet'}]},
               'other': {'type': 'ImagePrompt', 'data_root': 'prompts'}}
        override_imagenet_root(cfg, '/new/root')
        self.assertEqual(cfg['val']['data_root'], '/new/root')
        self.assertIsNone(cfg['val']['datalist_path'])
        self.assertEqual(cfg['train']['datasets'][0]['data_root'], '/new/root')
        self.assertEqual(cfg['other']['data_root'], 'prompts')


if __name__ == '__main__':
    unittest.main()
