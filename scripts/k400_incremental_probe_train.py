#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def read_urls(path: Path):
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith('#')]


def download_file(url: str, dst: Path, show_progress: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f'[download] {url} -> {dst}')
    if not show_progress or tqdm is None:
        urlretrieve(url, dst)
        return

    with tqdm(desc=f'download {dst.name}', unit='B', unit_scale=True, leave=False) as pbar:
        last = {'n': 0}

        def hook(block_count, block_size, total_size):
            if total_size and pbar.total != total_size:
                pbar.total = total_size
            downloaded = block_count * block_size
            delta = downloaded - last['n']
            if delta > 0:
                pbar.update(delta)
                last['n'] = downloaded

        urlretrieve(url, dst, reporthook=hook)


def extract_tar(tar_path: Path, extract_dir: Path, show_progress: bool = False):
    print(f'[extract] {tar_path} -> {extract_dir}')
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:*') as tf:
        members = tf.getmembers()
        if show_progress and tqdm is not None:
            for m in tqdm(members, desc=f'extract {tar_path.name}', unit='file', leave=False):
                tf.extract(m, path=extract_dir)
        else:
            tf.extractall(extract_dir)


def discover_videos(root: Path):
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def _normalize_clip_name(name: str) -> str:
    return Path(name).stem


def load_label_id_map(label_map_csv: Path):
    rows = list(csv.DictReader(label_map_csv.open('r', newline='')))
    if not rows:
        raise RuntimeError(f'No rows found in label-map CSV: {label_map_csv}')
    mapping = {}
    for r in rows:
        mapping[r['name']] = int(r['id'])
    return mapping


def load_k400_annotations(annotations_csv: Path, label_map_csv: Path | None = None):
    rows = list(csv.DictReader(annotations_csv.open('r', newline='')))
    if not rows:
        raise RuntimeError(f'No rows found in annotations CSV: {annotations_csv}')

    if label_map_csv is not None:
        label_to_idx = load_label_id_map(label_map_csv)
    else:
        labels = sorted({r['label'] for r in rows})
        label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}

    clip_to_label_idx = {}
    for r in rows:
        if r['label'] not in label_to_idx:
            raise RuntimeError(f"Label '{r['label']}' not found in label map")
        clip_name = f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}"
        clip_to_label_idx[_normalize_clip_name(clip_name)] = label_to_idx[r['label']]
    return clip_to_label_idx


def write_csv_from_extracted_with_annotations(
    extract_dir: Path,
    csv_path: Path,
    annotations_csv: Path,
    label_map_csv: Path | None = None
):
    videos = discover_videos(extract_dir)
    if not videos:
        raise RuntimeError(f'No videos found in {extract_dir}')

    clip_to_label_idx = load_k400_annotations(annotations_csv, label_map_csv=label_map_csv)
    missing = []
    rows = []
    for v in videos:
        key = _normalize_clip_name(v.name)
        lbl = clip_to_label_idx.get(key)
        if lbl is None:
            missing.append(v.name)
            continue
        rows.append((v.resolve(), lbl))

    if not rows:
        raise RuntimeError('No extracted videos matched annotation CSV clip names.')

    with csv_path.open('w') as f:
        for p, lbl in rows:
            f.write(f'{p} {lbl}\n')
    print(f'[csv] wrote {len(rows)} entries to {csv_path} using annotations')
    if missing:
        print(f'[warn] {len(missing)} files had no annotation match and were skipped')
    return len(rows), len(videos)


def run_eval(eval_config: dict, config_path: Path, devices):
    config_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))
    cmd = ['python', '-m', 'evals.main', '--fname', str(config_path), '--devices', *devices]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description='Incremental K400 probe training from tar URL batches')
    p.add_argument('--url-list', type=Path, required=True,
                   help='Path to k400_train_path.txt (one tar URL per line)')
    p.add_argument('--base-config', type=Path, default=Path('configs/evals/vitl16_k400_16x8x3.yaml'),
                   help='Base eval config to clone/override')
    p.add_argument('--work-dir', type=Path, default=Path('tmp/k400_incremental'))
    p.add_argument('--parts-per-batch', type=int, default=1,
                   help='How many tar parts to download/extract per incremental step')
    p.add_argument('--epochs-per-batch', type=int, default=1,
                   help='Probe epochs to train for each downloaded batch')
    p.add_argument('--devices', nargs='+', default=['cuda:0'])
    p.add_argument('--final-checkpoint-out', type=Path, required=True,
                   help='Where to copy final latest probe checkpoint at the end')
    p.add_argument('--show-progress', action='store_true',
                   help='Show tqdm status bars for download/extract')
    p.add_argument('--annotations-csv', type=Path, required=True,
                   help='Optional official K400 annotations CSV (label,youtube_id,time_start,time_end,...)')
    p.add_argument('--label-map-csv', type=Path, default=Path('data/datasets/k400/annotations/kinetics_400_labels.csv'),
                   help='Label-id mapping CSV with columns: id,name')
    args = p.parse_args()

    urls = read_urls(args.url_list)
    if not urls:
        raise RuntimeError(f'No URLs found in {args.url_list}')

    cfg = yaml.safe_load(args.base_config.read_text())

    if cfg.get('eval_name') != 'video_classification_frozen':
        raise RuntimeError('Base config must be for video_classification_frozen eval')

    args.work_dir.mkdir(parents=True, exist_ok=True)

    total_batches = (len(urls) + args.parts_per_batch - 1) // args.parts_per_batch
    print(f'[info] found {len(urls)} tar URLs -> {total_batches} incremental batches')

    total_processed = 0
    total_discovered = 0
    for bidx in range(total_batches):
        lo = bidx * args.parts_per_batch
        hi = min(len(urls), (bidx + 1) * args.parts_per_batch)
        batch_urls = urls[lo:hi]

        batch_dir = args.work_dir / f'batch_{bidx:04d}'
        dl_dir = batch_dir / 'downloads'
        ex_dir = batch_dir / 'extracted'
        csv_path = batch_dir / 'train.csv'
        cfg_path = batch_dir / 'eval.yaml'
        batch_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n=== batch {bidx + 1}/{total_batches} ({len(batch_urls)} parts) ===')

        for url in batch_urls:
            fname = Path(url).name
            tar_path = dl_dir / fname
            download_file(url, tar_path, show_progress=args.show_progress)
            extract_tar(tar_path, ex_dir, show_progress=args.show_progress)

        processed, discovered = write_csv_from_extracted_with_annotations(
            ex_dir, csv_path, args.annotations_csv, label_map_csv=args.label_map_csv
        )

        total_processed += processed
        total_discovered += discovered

        run_cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy
        run_cfg['resume_checkpoint'] = (bidx > 0)
        run_cfg['train_only'] = True
        run_cfg['validation_only'] = False
        run_cfg.pop('validation_checkpoint_path', None)

        run_cfg['data']['dataset_train'] = str(csv_path.resolve())
        # val path still parsed by loader even in train_only mode
        run_cfg['data']['dataset_val'] = str(csv_path.resolve())
        run_cfg['optimization']['num_epochs'] = int(args.epochs_per_batch)

        run_eval(run_cfg, cfg_path, args.devices)

        # cleanup this batch so next batch starts fresh on disk
        print(f'[cleanup] removing {batch_dir}')
        shutil.rmtree(batch_dir, ignore_errors=True)

    pretrain_folder = Path(cfg['pretrain']['folder'])
    eval_tag = cfg.get('tag', None)
    write_tag = cfg['pretrain']['write_tag']

    ckpt_dir = pretrain_folder / 'video_classification_frozen'
    if eval_tag:
        ckpt_dir = ckpt_dir / eval_tag

    latest_ckpt = ckpt_dir / f'{write_tag}-latest.pth.tar'
    if not latest_ckpt.exists():
        raise FileNotFoundError(f'Expected latest checkpoint not found: {latest_ckpt}')

    args.final_checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest_ckpt, args.final_checkpoint_out)
    print(f'[done] copied final checkpoint: {latest_ckpt} -> {args.final_checkpoint_out}')
    print(f'[done] processed videos: {total_processed}/{total_discovered}')


if __name__ == '__main__':
    main()
