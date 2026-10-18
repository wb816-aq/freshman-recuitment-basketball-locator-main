"""读取验收视频，逐帧检测篮球，绘制检测框 / 类别 / 置信度 / 中心点坐标 / 运动轨迹，并导出结果视频。

用法:
    python video_detect.py
    python video_detect.py --weights weights/basketball.pt --conf 0.25 --imgsz 960 --track
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "evaluation" / "basketball_dribble_evaluation.mp4"
DEFAULT_OUTPUT = ROOT / "outputs" / "result.mp4"

CLASS_NAME = "basketball"
BOX_COLOR = (0, 220, 0)          # BGR
CENTER_COLOR = (0, 215, 255)
HOLD_COLOR = (0, 140, 255)
HUD_COLOR = (30, 30, 30)
# 多目标跟踪时按 id 分配轨迹颜色
TRACK_PALETTE = [
    (0, 220, 0), (255, 120, 0), (0, 200, 255), (255, 0, 180),
    (180, 255, 0), (0, 120, 255), (255, 255, 0), (128, 0, 255),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="篮球检测 + 结果视频导出")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="输入视频")
    p.add_argument("--weights", type=Path, default=None, help="模型权重，默认自动查找")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出视频路径")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸，调大有利于小目标")
    p.add_argument("--device", default=None, help="cpu / 0，留空自动选择")
    p.add_argument("--track", action="store_true", help="启用 ByteTrack 跟踪，轨迹更稳定")
    p.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器配置")
    p.add_argument("--trail", type=int, default=60, help="轨迹长度（帧数），0 为关闭轨迹")
    p.add_argument("--hold", type=int, default=0, help="漏检时延续显示上一次检测框的帧数")
    p.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（调试用），0 为全部")
    p.add_argument("--fourcc", default=None, help="视频编码 fourcc，默认依次尝试 mp4v/avc1/XVID")
    return p.parse_args()


def find_weights(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.is_file():
            return explicit
        sys.exit(f"[detect] 错误: 找不到权重 {explicit}")
    candidate = ROOT / "weights" / "basketball.pt"
    if candidate.is_file():
        return candidate
    runs = ROOT / "runs" / "detect"
    if runs.is_dir():
        best = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if best:
            return best[0]
    sys.exit(
        "[detect] 错误: 没有找到模型权重。\n"
        "        请先运行 `python train.py`，或用 --weights 指定 .pt 文件。"
    )


def open_writer(path: Path, fps: float, size: tuple[int, int], fourcc: str | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    codecs = [fourcc] if fourcc else ["mp4v", "avc1", "XVID"]
    for code in codecs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer, code
        writer.release()
    sys.exit(f"[detect] 错误: 无法创建输出视频 {path}（试过 {codecs}）")


def put_label(img, text: str, x: int, y: int, color, scale: float = 0.5, thickness: int = 1) -> None:
    """在 (x, y) 处绘制带底色的小标签。"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = max(0, y - th - 6)
    cv2.rectangle(img, (max(0, x), top), (x + tw + 8, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_trail(frame, points: deque, color) -> None:
    """按点的新旧程度做颜色渐变地画折线。"""
    if len(points) < 2:
        return
    overlay = frame.copy()
    n = len(points)
    pts = list(points)
    for i in range(1, n):
        ratio = i / n  # 越靠近当前帧越亮
        seg_color = tuple(int(c * (0.25 + 0.75 * ratio)) for c in color)
        cv2.line(overlay, pts[i - 1], pts[i], seg_color, 2, cv2.LINE_AA)
        cv2.circle(overlay, pts[i], max(1, int(3 * ratio)), seg_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


def draw_hud(frame, lines: list[str]) -> None:
    """左上角状态面板（只用 ASCII，避免 cv2 无法渲染中文）。"""
    x, y = 12, 24
    for i, text in enumerate(lines):
        put_label(frame, text, x, y + i * 22, HUD_COLOR, scale=0.55, thickness=1)


def main() -> None:
    args = parse_args()

    if not args.source.is_file():
        sys.exit(f"[detect] 错误: 找不到输入视频 {args.source}")
    weights = find_weights(args.weights)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[detect] 错误: 未安装 ultralytics，请先执行 pip install -r requirements.txt")

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        sys.exit(f"[detect] 错误: 无法读取视频 {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_run = min(total, args.max_frames) if args.max_frames > 0 else total

    print(f"[detect] 视频: {args.source.name}  {width}x{height}  {fps:.2f} fps  共 {total} 帧")
    print(f"[detect] 权重: {weights}")
    print(f"[detect] 跟踪: {'ByteTrack' if args.track else '关闭'}  轨迹长度: {args.trail}")

    model = YOLO(str(weights))
    writer, codec = open_writer(args.output, fps, (width, height), args.fourcc)
    print(f"[detect] 输出: {args.output}  (编码 {codec})\n")

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(1, args.trail)))
    frame_idx = 0
    detected_frames = 0
    total_boxes = 0
    conf_sum = 0.0
    last = None  # (xyxy, conf, center)
    miss_streak = 0
    t_start = time.perf_counter()

    while cap.isOpened() and frame_idx < frames_to_run:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if args.track:
            results = model.track(
                frame, persist=True, tracker=args.tracker,
                conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False,
            )
        else:
            results = model.predict(
                frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False,
            )

        boxes = results[0].boxes
        n = 0 if boxes is None else len(boxes)
        best_conf = 0.0

        if n > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [0] * n
            primary = int(np.argmax(confs))  # 置信度最高的框，用作 HUD 与"上一帧"记录

            for i in range(n):
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                conf = float(confs[i])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = TRACK_PALETTE[int(ids[i]) % len(TRACK_PALETTE)] if args.track else BOX_COLOR

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAME} {conf:.2f}"
                if args.track:
                    label = f"{CLASS_NAME} #{int(ids[i])} {conf:.2f}"
                put_label(frame, label, x1, y1, color)

                cv2.circle(frame, (cx, cy), 4, CENTER_COLOR, -1, cv2.LINE_AA)
                cv2.putText(frame, f"({cx},{cy})", (x1, min(height - 6, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1, cv2.LINE_AA)

                # 开启跟踪时每个 id 一条轨迹；关闭时只跟踪置信度最高的球，避免轨迹在多个框之间跳动
                if args.trail > 0 and (args.track or i == primary):
                    trails[int(ids[i]) if args.track else 0].append((cx, cy))

            best_conf = float(confs[primary])
            conf_sum += float(confs.sum())
            total_boxes += n
            bx1, by1, bx2, by2 = (int(v) for v in xyxy[primary])
            last = ((bx1, by1, bx2, by2), best_conf, ((bx1 + bx2) // 2, (by1 + by2) // 2))

            detected_frames += 1
            miss_streak = 0
        else:
            miss_streak += 1
            # 短时遮挡/模糊时延续上一帧的框，避免结果视频里目标闪烁
            if last is not None and args.hold > 0 and miss_streak <= args.hold:
                (x1, y1, x2, y2), conf, (cx, cy) = last
                cv2.rectangle(frame, (x1, y1), (x2, y2), HOLD_COLOR, 1)
                put_label(frame, f"{CLASS_NAME} {conf:.2f} hold", x1, y1, HOLD_COLOR)

        if args.trail > 0:
            for tid, pts in trails.items():
                color = TRACK_PALETTE[tid % len(TRACK_PALETTE)] if args.track else BOX_COLOR
                draw_trail(frame, pts, color)

        elapsed = time.perf_counter() - t_start
        rate = detected_frames / frame_idx * 100
        draw_hud(frame, [
            f"frame {frame_idx}/{frames_to_run}",
            f"proc fps {frame_idx / elapsed:.1f}",
            f"balls {n}" + (f"  best conf {best_conf:.2f}" if n else "  (no detection)"),
            f"detect rate {rate:.0f}%",
        ])

        writer.write(frame)
        if frame_idx % 60 == 0 or frame_idx == frames_to_run:
            print(f"[detect] {frame_idx}/{frames_to_run} 帧  检测率 {rate:.0f}%")

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    print("\n===== 处理结果 =====")
    print(f"处理帧数    : {frame_idx}")
    print(f"检测到球的帧: {detected_frames} ({detected_frames / max(1, frame_idx) * 100:.1f}%)")
    print(f"检测框总数  : {total_boxes}")
    if total_boxes:
        print(f"平均置信度  : {conf_sum / total_boxes:.4f}")
    print(f"耗时        : {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps)")
    print(f"输出视频    : {args.output}")
    print("====================")


if __name__ == "__main__":
    main()
