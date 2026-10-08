"""YOLO block detection with deterministic HSV color classification."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Iterable

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


COLOR_LABELS = {"blue": "蓝色积木", "green": "绿色积木", None: "颜色未知"}
COLOR_BGR = {"blue": (220, 110, 35), "green": (55, 185, 75), None: (140, 140, 140)}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    color: str | None
    color_coverage: float
    coverages: dict[str, float] = field(default_factory=dict)
    observed: bool = True

    @property
    def display_label(self):
        return COLOR_LABELS[self.color]


class TargetResolutionError(ValueError):
    pass


def resolve_unique_target(detections, color):
    matches = [item for item in detections if item.color == color]
    color_name = COLOR_LABELS.get(color, color)
    if not matches:
        raise TargetResolutionError(f"未检测到可抓取的{color_name}")
    if len(matches) > 1:
        raise TargetResolutionError(
            f"检测到 {len(matches)} 个{color_name}，目标不唯一，本次不抓取")
    if not matches[0].observed:
        raise TargetResolutionError(f"{color_name}正在重新确认，本次不抓取")
    return matches[0]


DEFAULT_DETECTION_CONFIG = {
    "candidate_labels": ["block", "cube"],
    "confidence": 0.35,
    "imgsz": 1280,
    "nms_iou": 0.75,
    "max_det": 20,
    "min_box_area": 400,
    "inset_ratio": 0.10,
    "min_saturation": 70,
    "min_value": 45,
    "min_color_coverage": 0.18,
    "min_color_margin": 0.06,
    "tracking_iou_threshold": 0.30,
    "tracking_confirm_frames": 3,
    "tracking_max_missed_frames": 3,
    "tracking_smoothing_alpha": 0.35,
    "tracking_color_window": 5,
    "tracking_color_confirm_frames": 2,
    "colors": {
        "blue": {"lower": [90, 70, 45], "upper": [135, 255, 255]},
        "green": {"lower": [35, 70, 45], "upper": [85, 255, 255]},
    },
}


def _bbox_iou(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def suppress_duplicate_detections(detections):
    """Remove overlapping same-color boxes, preserving separate real targets."""
    kept = []
    for item in sorted(detections, key=lambda d: (d.observed, d.confidence), reverse=True):
        duplicate = False
        for other in kept:
            if item.color != other.color:
                continue
            a, b = item.bbox, other.bbox
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
            if _bbox_iou(a, b) >= .55 or (smaller > 0 and intersection / smaller >= .90):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


@dataclass
class _DetectionTrack:
    bbox: np.ndarray
    label: str
    confidence: float
    color_coverage: float
    coverages: dict[str, float]
    color_history: deque
    stable_color: str | None = None
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


class DetectionStabilizer:
    """Confirm and smooth detections before they reach the UI or grasp path."""

    def __init__(self, config=None):
        settings = {**DEFAULT_DETECTION_CONFIG, **(config or {})}
        self.iou_threshold = float(settings["tracking_iou_threshold"])
        self.confirm_frames = max(1, int(settings["tracking_confirm_frames"]))
        self.max_missed = max(0, int(settings["tracking_max_missed_frames"]))
        self.alpha = min(1.0, max(0.01, float(settings["tracking_smoothing_alpha"])))
        self.color_window = max(1, int(settings["tracking_color_window"]))
        self.color_confirm = max(1, int(settings["tracking_color_confirm_frames"]))
        self._tracks = []

    def _new_track(self, detection):
        history = deque([detection.color], maxlen=self.color_window)
        stable_color = detection.color if self.color_confirm <= 1 else None
        return _DetectionTrack(
            bbox=np.asarray(detection.bbox, dtype=float),
            label=detection.label,
            confidence=detection.confidence,
            color_coverage=detection.color_coverage,
            coverages=dict(detection.coverages),
            color_history=history,
            stable_color=stable_color,
            confirmed=self.confirm_frames <= 1,
        )

    def _update_color(self, track, color):
        track.color_history.append(color)
        counts = Counter(item for item in track.color_history if item is not None)
        if not counts:
            return
        winner, votes = counts.most_common(1)[0]
        if votes < self.color_confirm:
            return
        if track.stable_color is None or winner == track.stable_color:
            track.stable_color = winner
            return
        current_votes = counts.get(track.stable_color, 0)
        if votes > current_votes:
            track.stable_color = winner

    def _update_track(self, track, detection):
        alpha = self.alpha
        track.bbox = ((1.0 - alpha) * track.bbox
                      + alpha * np.asarray(detection.bbox, dtype=float))
        track.label = detection.label
        track.confidence = ((1.0 - alpha) * track.confidence
                            + alpha * detection.confidence)
        track.color_coverage = ((1.0 - alpha) * track.color_coverage
                                + alpha * detection.color_coverage)
        keys = set(track.coverages) | set(detection.coverages)
        track.coverages = {
            key: ((1.0 - alpha) * track.coverages.get(key, 0.0)
                  + alpha * detection.coverages.get(key, 0.0))
            for key in keys
        }
        track.hits += 1
        track.misses = 0
        track.confirmed = track.confirmed or track.hits >= self.confirm_frames
        self._update_color(track, detection.color)

    def update(self, detections):
        detections = suppress_duplicate_detections(detections)
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        pairs = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                score = _bbox_iou(track.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, track_index, detection_index))
        for _, track_index, detection_index in sorted(pairs, reverse=True):
            if track_index not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self._update_track(self._tracks[track_index], detections[detection_index])
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.misses += 1
            if not track.confirmed:
                track.hits = 0
        for detection_index in unmatched_detections:
            self._tracks.append(self._new_track(detections[detection_index]))
        self._tracks = [track for track in self._tracks
                        if track.misses <= self.max_missed]

        stable = []
        for track in self._tracks:
            if not track.confirmed:
                continue
            stable.append(Detection(
                bbox=tuple(int(round(value)) for value in track.bbox),
                label=track.label,
                confidence=track.confidence,
                color=track.stable_color,
                color_coverage=track.color_coverage,
                coverages=dict(track.coverages),
                observed=track.misses == 0,
            ))
        return sorted(suppress_duplicate_detections(stable), key=lambda item: (item.bbox[0], item.bbox[1]))


def _clamped_crop(image, bbox, inset_ratio=0.10):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    inset_x = int((x2 - x1) * inset_ratio)
    inset_y = int((y2 - y1) * inset_ratio)
    x1, x2 = x1 + inset_x, x2 - inset_x
    y1, y2 = y1 + inset_y, y2 - inset_y
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def classify_block_color(image, bbox, config=None):
    """Return ``(color, winning_coverage, all_coverages)`` for one box."""
    settings = {**DEFAULT_DETECTION_CONFIG, **(config or {})}
    colors = settings.get("colors", DEFAULT_DETECTION_CONFIG["colors"])
    crop = _clamped_crop(image, bbox, float(settings["inset_ratio"]))
    if crop is None or crop.size == 0:
        return None, 0.0, {name: 0.0 for name in colors}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    valid = ((hsv[:, :, 1] >= int(settings["min_saturation"]))
             & (hsv[:, :, 2] >= int(settings["min_value"])))
    denominator = int(valid.size)
    if denominator == 0:
        return None, 0.0, {name: 0.0 for name in colors}
    coverages = {}
    for name, bounds in colors.items():
        mask = cv2.inRange(
            hsv, np.asarray(bounds["lower"], dtype=np.uint8),
            np.asarray(bounds["upper"], dtype=np.uint8),
        ).astype(bool)
        coverages[name] = float(np.count_nonzero(mask & valid) / denominator)
    ranked = sorted(coverages.items(), key=lambda item: item[1], reverse=True)
    winner, coverage = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if (coverage < float(settings["min_color_coverage"])
            or coverage - runner_up < float(settings["min_color_margin"])):
        return None, coverage, coverages
    return winner, coverage, coverages


class AirbotYolo:
    def __init__(self, model_factory=YOLO, *, checkpoint=None):
        with open("configs/config_file.yaml", "r", encoding="utf-8") as file:
            config_path = yaml.safe_load(file)["Path"]
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        yolo_config = config["AirbotYolo"]
        self.verbose = yolo_config.get("verbose", False)
        self.detection_config = {
            **DEFAULT_DETECTION_CONFIG, **config.get("RealtimeDetection", {})}
        self.checkpoint = checkpoint or yolo_config["checkpoint"]
        self.model = model_factory(model=self.checkpoint)
        self.result = None
        self._inference_lock = RLock()

    def inference(self, image):
        with self._inference_lock:
            result = self.model(
                image,
                verbose=self.verbose,
                imgsz=int(self.detection_config["imgsz"]),
                conf=float(self.detection_config["confidence"]),
                iou=float(self.detection_config["nms_iou"]),
                max_det=int(self.detection_config["max_det"]),
            )
            self.result = result
            return result

    def detect_candidates(self, image) -> list[Detection]:
        detections = []
        allowed = set(self.detection_config["candidate_labels"])
        threshold = float(self.detection_config["confidence"])
        min_area = int(self.detection_config["min_box_area"])
        height, width = image.shape[:2]
        for result in self.inference(image):
            for box in result.boxes:
                label = self.model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                if label not in allowed or confidence < threshold:
                    continue
                raw = box.xyxy[0].flatten().tolist()
                x1 = max(0, min(width, int(raw[0])))
                y1 = max(0, min(height, int(raw[1])))
                x2 = max(0, min(width, int(raw[2])))
                y2 = max(0, min(height, int(raw[3])))
                if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < min_area:
                    continue
                color, coverage, coverages = classify_block_color(
                    image, (x1, y1, x2, y2), self.detection_config)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2), label=label,
                    confidence=confidence, color=color,
                    color_coverage=coverage, coverages=coverages))
        return suppress_duplicate_detections(detections)

    def get_max_conf_bbox_and_label(self, image, label_filter: Iterable[str] = ()):
        """Compatibility helper used by the legacy automatic mode."""
        ignored = set(label_filter)
        best = None
        for result in self.inference(image):
            for box in result.boxes:
                label = self.model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                if label in ignored or (best is not None and confidence <= best[2]):
                    continue
                bbox = tuple(int(value) for value in box.xyxy[0].flatten().tolist())
                best = (bbox, label, confidence)
        return best if best is not None else (None, None, 0.0)


def draw_detections(image, detections, selected_bbox=None):
    overlay = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        selected = selected_bbox == detection.bbox
        color = COLOR_BGR[detection.color]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4 if selected else 2)
        ascii_label = {"blue": "BLUE BLOCK", "green": "GREEN BLOCK",
                       None: "UNKNOWN"}[detection.color]
        state = "" if detection.observed else " HOLD"
        text = (f"{ascii_label}{state}  "
                f"{detection.confidence:.2f}/{detection.color_coverage:.0%}")
        cv2.putText(overlay, text, (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return overlay


def main():
    camera = cv2.VideoCapture(2)
    detector = AirbotYolo()
    try:
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                break
            cv2.imshow("AIRBOT block detection", draw_detections(
                frame, detector.detect_candidates(frame)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
