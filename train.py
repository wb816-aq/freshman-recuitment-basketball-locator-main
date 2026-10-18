"""在 dataset/data.yaml 上微调 YOLOv8 篮球检测器。

用法:
    python train.py
    python train.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "weights" / "basketball.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="微调 YOLOv8 篮球检测器（单类 basketball）")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集配置 data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="预训练权重（首次运行自动下载）或本地权重")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--imgsz", type=int, default=640, help="训练输入尺寸")
    p.add_argument("--batch", type=int, default=16, help="批大小，-1 为自动")
    p.add_argument("--device", default=None, help="cpu / 0 / 0,1，留空则自动选择")
    p.add_argument("--project", default=str(ROOT / "runs" / "detect"), help="训练输出根目录")
    p.add_argument("--name", default="basketball_yolov8", help="本次训练的输出子目录")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--patience", type=int, default=30, help="验证指标多少轮不提升后早停")
    p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 8, help="数据加载线程数")
    p.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    p.add_argument("--no-copy", action="store_true", help="不把 best.pt 复制到 weights/")
    return p.parse_args()


def check_dataset(data_yaml: Path) -> None:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            sys.exit(f"[train] 错误: {data_yaml} 缺少 '{split}' 字段")
        split_dir = (data_yaml.parent / rel).resolve()
        if not split_dir.is_dir():
            sys.exit(f"[train] 错误: 找不到 {split} 目录 {split_dir}")
        print(f"[train] {split}: {len(list(split_dir.glob('*')))} 个文件 ({split_dir})")
    if not cfg.get("names"):
        sys.exit(f"[train] 错误: {data_yaml} 缺少 'names' 字段")


def to_absolute_data_yaml(data_yaml: Path) -> Path:
    """生成 path 为绝对路径的副本供 ultralytics 使用。

    仓库里的 data.yaml 写的是 `path: .`，而不同 ultralytics 版本对相对 path 的解析基准
    并不一致（有的按 yaml 所在目录、有的按全局 datasets_dir），改成绝对路径可以避免
    "找不到 train/images" 这类问题。
    """
    data_yaml = data_yaml.resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    cfg["path"] = str(data_yaml.parent)
    out = data_yaml.with_name("data.absolute.yaml")
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[train] 已生成绝对路径配置: {out}")
    return out


def main() -> None:
    args = parse_args()

    if not args.data.is_file():
        sys.exit(f"[train] 错误: 找不到数据集配置 {args.data}")
    check_dataset(args.data)
    data_yaml = to_absolute_data_yaml(args.data)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[train] 错误: 未安装 ultralytics，请先执行 pip install -r requirements.txt")

    print(f"[train] 预训练权重: {args.model}")
    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume,
        exist_ok=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\n[train] 训练结束，权重目录: {save_dir}")

    if not best.is_file():
        sys.exit(f"[train] 错误: 训练未产出 {best}")

    if not args.no_copy:
        DEFAULT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, DEFAULT_WEIGHTS)
        print(f"[train] best.pt 已复制到 {DEFAULT_WEIGHTS}（video_detect.py 默认加载）")

    print("[train] 用 best.pt 在验证集上复核指标 ...")
    metrics = YOLO(str(best)).val(
        data=str(data_yaml), split="val", imgsz=args.imgsz, device=args.device, verbose=False
    )
    box = metrics.box
    print("\n===== 验证集指标 =====")
    print(f"mAP50     : {box.map50:.4f}")
    print(f"mAP50-95  : {box.map:.4f}")
    print(f"precision : {box.mp:.4f}")
    print(f"recall    : {box.mr:.4f}")
    print("=====================")


if __name__ == "__main__":
    main()
