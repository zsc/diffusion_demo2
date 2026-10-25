"""Expand a traced CelebA cache, retaining its exact validation/training pixels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image


def array_hash(array):
    digest = hashlib.sha256()
    for start in range(0, len(array), 512):
        digest.update(array[start:start + 512].tobytes())
    return digest.hexdigest()


def prepare(source_zip, base_path, output, expected_train=162770, workers=8):
    base_path, output = Path(base_path), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite an existing dataset')
    meta = json.loads(base_path.with_suffix('.json').read_text())
    assert meta['split_mode'] == 'validation_first'
    assert meta['crop'] == [0, 20, 178, 198] and meta['resolution'] == 96
    assert meta['resize'] == 'Pillow LANCZOS'
    base = np.load(base_path, mmap_mode='r')
    val, train = meta['images']['validation'], meta['images']['train']
    assert base.dtype == np.uint8 and base.shape == (len(val) + len(train), 96, 96, 3)
    assert len(set(val + train)) == len(val + train)
    assert array_hash(base) == meta['array_sha256'], 'Base cache hash mismatch'
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        partitions = list(csv.DictReader(io.StringIO(archive.read('list_eval_partition.csv').decode())))
        official_train = {row['image_id'] for row in partitions if row['partition'] == '0'}
        official_val = {row['image_id'] for row in partitions if row['partition'] == '1'}
        assert len(official_train) == expected_train
        assert set(train) <= official_train and set(val) <= official_val
        assert not official_train.intersection(val)
        extra = sorted(official_train - set(train))
        names = train + extra
        assert len(names) == expected_train and set(names) == official_train
        members = {Path(n).name: n for n in archive.namelist() if n.lower().endswith('.jpg')}
        array = np.lib.format.open_memmap(output.with_suffix('.tmp.npy'), mode='w+',
                    dtype=np.uint8, shape=(len(val) + len(names), 96, 96, 3))
        array[:len(base)] = base

        def decode(raw):
            with Image.open(io.BytesIO(raw)) as image:
                assert image.size == (178, 218)
                return np.asarray(image.convert('RGB').crop((0, 20, 178, 198)).resize(
                    (96, 96), Image.Resampling.LANCZOS))

        source_hash = hashlib.sha256()
        with ThreadPoolExecutor(workers) as pool:
            for start in range(0, len(extra), 512):
                chunk = extra[start:start + 512]
                raw = [archive.read(members[name]) for name in chunk]
                for name, data in zip(chunk, raw):
                    source_hash.update(name.encode() + b'\0' + data)
                array[len(base) + start:len(base) + start + len(chunk)] = np.stack(list(pool.map(decode, raw)))
                if start % 5120 == 0:
                    print(json.dumps(dict(added=start + len(chunk), total_added=len(extra))), flush=True)
        array.flush()
        assert np.array_equal(array[:len(val)], base[:len(val)]), 'Validation pixels changed'
        assert np.array_equal(array[len(val):len(base)], base[len(val):]), 'Original training pixels changed'
        data_hash, val_hash = array_hash(array), array_hash(array[:len(val)])
        del array
        output.with_suffix('.tmp.npy').replace(output)
    result = {**meta, 'source_zip': str(source_zip), 'images': dict(validation=val, train=names),
              'array_sha256': data_hash, 'validation_sha256': val_hash,
              'source_selected_sha256': None,
              'new_source_selected_sha256': source_hash.hexdigest(),
              'base_source_selected_sha256': meta.get('source_selected_sha256'),
              'base_cache': str(base_path), 'base_array_sha256': meta['array_sha256'],
              'training_count': len(names), 'validation_count': len(val),
              'base_training_count': len(train), 'added_training_count': len(extra),
              'validation_byte_identical': True, 'base_training_byte_identical': True,
              'selection_seed': None, 'elapsed_seconds': time.perf_counter() - started,
              'note': 'Complete official training split; original cache is an exact prefix. Validation IDs, order and pixels unchanged.'}
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output.with_suffix('.json'))
    print(json.dumps({k: v for k, v in result.items() if k != 'images'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--base-data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    prepare(args.zip, args.base_data, args.output, workers=args.workers)
