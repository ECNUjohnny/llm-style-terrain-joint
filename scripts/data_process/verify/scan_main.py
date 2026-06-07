"""
扫描 data/main 下的联合数据集 (dem + rgb + txt) 并输出统计 / Scan joint dataset under data/main

用法 / Usage:
    uv run python scripts/data_process/verify/scan_main.py [data_dir]

    data_dir  默认为 data/main / defaults to data/main
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)


def scan_main(data_dir: str):
    dem_dir = os.path.join(data_dir, "dem")
    rgb_dir = os.path.join(data_dir, "rgb")
    txt_dir = os.path.join(data_dir, "txt")

    missing = []
    for label, d in [("dem", dem_dir), ("rgb", rgb_dir), ("txt", txt_dir)]:
        if not os.path.isdir(d):
            missing.append(label)
    if missing:
        print(
            f"缺少子目录 / Missing subdirectories — {', '.join(missing)}  "
            f"in {data_dir}"
        )
        return

    dem_files = sorted(glob.glob(os.path.join(dem_dir, "*.*")))
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.*")))
    txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.*")))

    print(
        f"文件数量 / File counts — "
        f"dem: {len(dem_files)},  rgb: {len(rgb_files)},  txt: {len(txt_files)}"
    )
    print("─" * 60)

    # ── 提取 basename (不含扩展名) ──
    dem_basenames = {os.path.splitext(os.path.basename(f))[0] for f in dem_files}
    rgb_basenames = {os.path.splitext(os.path.basename(f))[0] for f in rgb_files}
    txt_basenames = {os.path.splitext(os.path.basename(f))[0] for f in txt_files}

    all_basenames = dem_basenames | rgb_basenames | txt_basenames
    intersect = dem_basenames & rgb_basenames & txt_basenames

    print(
        f"去重 basename 总数 / Unique basenames: {len(all_basenames)}"
    )
    print(
        f"三目录完全匹配 / Fully matched (dem+ rgb+ txt): "
        f"{len(intersect)}"
    )

    only_dem = dem_basenames - rgb_basenames - txt_basenames
    only_rgb = rgb_basenames - dem_basenames - txt_basenames
    only_txt = txt_basenames - dem_basenames - rgb_basenames

    only_dem_rgb = (dem_basenames & rgb_basenames) - txt_basenames
    only_dem_txt = (dem_basenames & txt_basenames) - rgb_basenames
    only_rgb_txt = (rgb_basenames & txt_basenames) - dem_basenames

    if only_dem:
        print(f"  仅 dem / dem only: {len(only_dem)}  — {sorted(only_dem)[:5]}{'...' if len(only_dem) > 5 else ''}")
    if only_rgb:
        print(f"  仅 rgb / rgb only: {len(only_rgb)}  — {sorted(only_rgb)[:5]}{'...' if len(only_rgb) > 5 else ''}")
    if only_txt:
        print(f"  仅 txt / txt only: {len(only_txt)}  — {sorted(only_txt)[:5]}{'...' if len(only_txt) > 5 else ''}")
    if only_dem_rgb:
        print(f"  dem+rgb (缺txt) / dem+rgb (no txt): {len(only_dem_rgb)}")
    if only_dem_txt:
        print(f"  dem+txt (缺rgb) / dem+txt (no rgb): {len(only_dem_txt)}")
    if only_rgb_txt:
        print(f"  rgb+txt (缺dem) / rgb+txt (no dem): {len(only_rgb_txt)}")
    print("─" * 60)

    # ── dem 统计 ──
    print("DEM 高程图统计 / DEM height-map statistics:")
    dem_exts = {}
    shapes = set()
    dtypes = set()
    min_vals, max_vals, mean_vals = [], [], []

    for fpath in dem_files:
        ext = os.path.splitext(fpath)[1]
        dem_exts[ext] = dem_exts.get(ext, 0) + 1
        if ext == ".npy":
            arr = np.load(fpath)
        else:
            arr = np.array(Image.open(fpath))
        shapes.add(arr.shape)
        dtypes.add(str(arr.dtype))
        min_vals.append(float(arr.min()))
        max_vals.append(float(arr.max()))
        mean_vals.append(float(arr.mean()))

    print(f"  文件扩展名 / File extensions:{'':6s}", end="")
    for ext, cnt in sorted(dem_exts.items()):
        print(f"  {ext} ×{cnt}", end="")
    print()
    print(f"  NumPy 形状 / Array shapes:       {sorted(shapes)}")
    print(f"  数据类型 / dtypes:               {sorted(dtypes)}")
    print(f"  全局最小值 / Global min:          {min(min_vals):.6f}")
    print(f"  全局最大值 / Global max:          {max(max_vals):.6f}")
    print(f"  文件均值之均值 / Mean of means:   {np.mean(mean_vals):.6f}")
    print(f"  文件均值之中位数 / Median of means: {np.median(mean_vals):.6f}")
    print(f"  总数据量 / Total data:            {sum(os.path.getsize(f) for f in dem_files) / (1024**2):.2f} MiB")
    print()

    # ── rgb 统计 ──
    print("RGB 彩图统计 / RGB image statistics:")
    rgb_exts = {}
    sizes = {}
    modes = {}
    rgb_min, rgb_max, rgb_mean = [], [], []

    for fpath in rgb_files:
        ext = os.path.splitext(fpath)[1]
        rgb_exts[ext] = rgb_exts.get(ext, 0) + 1
        img = Image.open(fpath)
        sizes[img.size] = sizes.get(img.size, 0) + 1
        modes[img.mode] = modes.get(img.mode, 0) + 1
        arr = np.array(img)
        rgb_min.append(float(arr.min()))
        rgb_max.append(float(arr.max()))
        rgb_mean.append(float(arr.mean()))

    print(f"  文件扩展名 / File extensions:{'':6s}", end="")
    for ext, cnt in sorted(rgb_exts.items()):
        print(f"  {ext} ×{cnt}", end="")
    print()
    print("  图像尺寸 / Image sizes:")
    for sz, cnt in sorted(sizes.items()):
        print(f"    {sz[0]} × {sz[1]}  —  {cnt} 个文件")
    print("  色彩模式 / Modes:")
    for md, cnt in sorted(modes.items()):
        print(f"    {md}  —  {cnt} 个文件")
    if rgb_min:
        print(f"  全局像素最小值 / Global pixel min:  {min(rgb_min):.2f}")
        print(f"  全局像素最大值 / Global pixel max:  {max(rgb_max):.2f}")
        print(f"  文件均值之均值 / Mean of means:     {np.mean(rgb_mean):.2f}")
    print(f"  总数据量 / Total data:              {sum(os.path.getsize(f) for f in rgb_files) / (1024**2):.2f} MiB")
    print()

    # ── txt 统计 ──
    print("TXT 提示词统计 / TXT prompt statistics:")
    char_lens = []
    word_lens = []
    line_counts = []
    empty_count = 0

    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            empty_count += 1
            continue
        char_lens.append(len(text))
        word_lens.append(len(text.split()))
        line_counts.append(text.count("\n") + 1)

    print(f"  文件数量 / File count:          {len(txt_files)}")
    if empty_count:
        print(f"  空文件 / Empty files:           {empty_count}")
    if char_lens:
        print(f"  字符数 / Character count:")
        print(f"    min: {min(char_lens)},  max: {max(char_lens)},  mean: {np.mean(char_lens):.0f},  median: {np.median(char_lens):.0f}")
        print(f"  单词数 / Word count:")
        print(f"    min: {min(word_lens)},  max: {max(word_lens)},  mean: {np.mean(word_lens):.0f},  median: {np.median(word_lens):.0f}")
        print(f"  行数 / Line count:")
        print(f"    min: {min(line_counts)},  max: {max(line_counts)},  mean: {np.mean(line_counts):.0f},  median: {np.median(line_counts):.0f}")
    print(f"  总数据量 / Total data:          {sum(os.path.getsize(f) for f in txt_files) / 1024:.2f} KiB")
    print("─" * 60)

    # ── 总体概要 ──
    total_files = len(dem_files) + len(rgb_files) + len(txt_files)
    total_bytes = (
        sum(os.path.getsize(f) for f in dem_files)
        + sum(os.path.getsize(f) for f in rgb_files)
        + sum(os.path.getsize(f) for f in txt_files)
    )
    print("总体概要 / Overall summary:")
    print(f"  完整样本数 / Complete samples (dem+rgb+txt): {len(intersect)}")
    print(f"  总文件数 / Total files:                      {total_files}")
    print(f"  总数据量 / Total size:                       {total_bytes / (1024**2):.2f} MiB")
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="扫描 data/main 联合数据集并输出统计 / Scan joint dataset under data/main"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/main",
        help="data/main 目录路径 / Path to the data/main directory",
    )
    args = parser.parse_args()
    scan_main(args.data_dir)


if __name__ == "__main__":
    main()
