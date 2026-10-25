import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from scripts.prepare_celeba96_full import prepare


class FullCelebaCacheTest(unittest.TestCase):
    def test_expansion_preserves_validation_and_excludes_other_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f'{i:06d}.jpg' for i in range(1, 7)]
            with zipfile.ZipFile(root / 'source.zip', 'w') as archive:
                archive.writestr('list_eval_partition.csv', 'image_id,partition\n' +
                    '\n'.join(f'{name},{part}' for name, part in zip(names, [0, 0, 0, 0, 1, 2])))
                for i, name in enumerate(names):
                    data = io.BytesIO()
                    Image.new('RGB', (178, 218), (i * 30, 20, 40)).save(data, format='JPEG')
                    archive.writestr('images/' + name, data.getvalue())
            # Distinctive cache pixels ensure preservation rather than re-decoding.
            base = np.random.default_rng(9).integers(0, 256, (3, 96, 96, 3), dtype=np.uint8)
            np.save(root / 'base.npy', base)
            meta = dict(split_mode='validation_first', crop=[0, 20, 178, 198], resolution=96,
                        resize='Pillow LANCZOS', images=dict(validation=[names[4]], train=[names[2], names[0]]),
                        array_sha256=hashlib.sha256(base.tobytes()).hexdigest())
            (root / 'base.json').write_text(json.dumps(meta))
            result = prepare(root / 'source.zip', root / 'base.npy', root / 'full.npy', expected_train=4)
            expanded = np.load(root / 'full.npy')
            np.testing.assert_array_equal(expanded[:3], base)
            self.assertEqual(expanded.shape, (5, 96, 96, 3))
            self.assertEqual(result['images']['validation'], [names[4]])
            self.assertEqual(set(result['images']['train']), set(names[:4]))
            self.assertEqual(result['array_sha256'], hashlib.sha256(expanded.tobytes()).hexdigest())
            with self.assertRaises(ValueError):
                prepare(root / 'source.zip', root / 'base.npy', root / 'full.npy', expected_train=4)
            meta['images']['train'][0] = names[5]
            (root / 'base.json').write_text(json.dumps(meta))
            with self.assertRaises(AssertionError):
                prepare(root / 'source.zip', root / 'base.npy', root / 'invalid.npy', expected_train=4)


if __name__ == '__main__':
    unittest.main()
