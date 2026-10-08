"""Safety-bounded local voice command parsing for the grasp demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import unicodedata


PHRASES = {
    "capture": ("拍照", "拍一张照", "捕获图像", "采集图像"),
    "predict": ("预测抓取", "预测抓取位姿", "计算抓取位姿"),
    "grasp": ("开始抓取", "执行抓取", "抓取选中物体", "抓取并放置"),
    "grasp_blue": ("抓取蓝色积木", "抓蓝色积木", "抓一下蓝色积木", "把蓝色积木抓起来"),
    "grasp_green": ("抓取绿色积木", "抓绿色积木", "抓一下绿色积木", "把绿色积木抓起来"),
    "open_gripper": ("打开夹爪", "张开夹爪", "松开夹爪"),
    "close_gripper": ("关闭夹爪", "闭合夹爪", "合上夹爪"),
    "observe": ("回到观察位", "返回观察位", "回到观察姿态", "移动到观察位"),
}
LABELS = {
    "capture": "手动拍照（会清除旧目标）",
    "predict": "预测手动选中物体的抓取位姿",
    "grasp": "抓取手动选中物体并放置",
    "grasp_blue": "抓取唯一的蓝色积木",
    "grasp_green": "抓取唯一的绿色积木",
    "open_gripper": "打开夹爪",
    "close_gripper": "关闭夹爪",
    "observe": "移动到观察位（会清除旧目标）",
}
ALIASES = {
    "假爪": "夹爪", "甲爪": "夹爪", "家爪": "夹爪", "夹抓": "夹爪",
    "假抓": "夹爪", "甲抓": "夹爪", "家抓": "夹爪", "夹着": "夹爪",
    "假找": "夹爪", "甲找": "夹爪", "家找": "夹爪", "夹找": "夹爪",
    "兰色": "蓝色", "篮色": "蓝色", "难色": "蓝色", "南色": "蓝色",
    "率色": "绿色", "虑色": "绿色", "旅色": "绿色", "驴色": "绿色",
    "鸡木": "积木", "积母": "积木", "积目": "积木", "机木": "积木",
    "抓去": "抓取", "抓曲": "抓取", "打来夹爪": "打开夹爪",
}
PREFIX_FILLERS = ("请帮我", "麻烦你", "麻烦", "请", "帮我", "给我", "现在")
SUFFIX_FILLERS = ("一下", "一下吧", "吧", "啊", "呢")
NEGATIONS = ("不要", "别", "不用", "不许", "不能", "取消")
REPORTED_CONTEXT = ("他说", "她说", "它说", "电视里", "视频里", "广播里", "听到",
                    "如果", "比如", "有人说", "刚才说")
ACTION_MARKERS = ("打开", "张开", "松开", "关闭", "闭合", "合上", "抓", "拍照",
                  "捕获", "采集", "预测", "计算", "回到", "返回", "移动")


@dataclass(frozen=True)
class CommandMatch:
    action: str
    transcript: str
    normalized: str
    quality: str

    @property
    def auto_executable(self):
        return self.quality in ("exact", "curated")


def _strip_wrappers(text):
    changed = True
    while changed:
        changed = False
        for prefix in PREFIX_FILLERS:
            if text.startswith(prefix):
                text, changed = text[len(prefix):], True
                break
        for suffix in SUFFIX_FILLERS:
            if text.endswith(suffix):
                text, changed = text[:-len(suffix)], True
                break
    return text


def _distance(left, right):
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def _embedded_commands(text):
    """Find non-overlapping complete whitelist phrases inside longer ASR text."""
    candidates = []
    for action, phrases in PHRASES.items():
        for phrase in phrases:
            start = text.find(phrase)
            while start >= 0:
                candidates.append((start, start + len(phrase), action, phrase))
                start = text.find(phrase, start + 1)
    selected = []
    for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if any(candidate[0] >= item[0] and candidate[1] <= item[1]
               and candidate[2] == item[2] for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected)


def match_command(text):
    transcript = unicodedata.normalize("NFKC", text).strip()
    if not transcript or "\n" in transcript:
        raise ValueError("未执行：未识别到单条有效指令。")
    if any(token in transcript for token in NEGATIONS):
        raise ValueError("未执行：识别结果包含否定或取消语义。")
    if any(token in transcript for token in REPORTED_CONTEXT):
        raise ValueError("未执行：识别结果像转述或条件句。")
    if re.search(r"[？?\"'“”《》]", transcript):
        raise ValueError("未执行：疑问或引用内容不作为机械臂指令。")
    if transcript.rstrip("。！! ").endswith(("吗", "么")) or "是不是" in transcript:
        raise ValueError("未执行：疑问句不作为机械臂指令。")
    normalized = re.sub(r"[\s，,。、！!]", "", transcript)
    alias_used = False
    for wrong, right in ALIASES.items():
        if wrong in normalized:
            normalized = normalized.replace(wrong, right)
            alias_used = True
    normalized = _strip_wrappers(normalized)
    for action, phrases in PHRASES.items():
        if normalized in phrases:
            return CommandMatch(action, transcript, normalized,
                                "curated" if alias_used else "exact")
    candidates = []
    for action, phrases in PHRASES.items():
        for phrase in phrases:
            if len(phrase) >= 4 and abs(len(normalized) - len(phrase)) <= 1:
                if _distance(normalized, phrase) == 1:
                    candidates.append((action, phrase))
    if len(candidates) == 1:
        action, phrase = candidates[0]
        return CommandMatch(action, transcript, phrase, "fuzzy")

    embedded = _embedded_commands(normalized)
    if len(embedded) == 1:
        start, end, action, phrase = embedded[0]
        outside = normalized[:start] + normalized[end:]
        if (any(marker in outside for marker in ACTION_MARKERS)
                or any(joiner in normalized for joiner in
                       ("然后", "再", "同时", "并且", "或者"))):
            raise ValueError("未执行：识别结果包含多项动作。")
        return CommandMatch(action, transcript, phrase, "extracted")
    if len(embedded) > 1:
        raise ValueError("未执行：一次只能说一条指令。")
    marker_count = sum(normalized.count(marker) for marker in ACTION_MARKERS)
    if any(joiner in normalized for joiner in ("然后", "再", "同时", "并且", "或者")) or marker_count > 1:
        raise ValueError("未执行：一次只能说一条指令。")
    raise ValueError(
        "未执行：未提取到唯一的支持指令，可说“抓取蓝色积木”"
        "“抓取绿色积木”或“打开夹爪”。")


def parse_command(text):
    return match_command(text).action


def execute_command(target, text):
    match = match_command(text)
    action = match.action
    if target.voice_is_busy():
        raise ValueError("未执行：机械臂或抓取计算正在进行。")
    if action in ("predict", "grasp") and not target.voice_has_selection():
        raise ValueError("未执行：手动抓取需要先拍照并选择目标。")
    callbacks = {
        "capture": target.capture_frame,
        "predict": target.predict_grasp_pose,
        "grasp": target.predict_and_grasp,
        "grasp_blue": lambda: target.grasp_color("blue"),
        "grasp_green": lambda: target.grasp_color("green"),
        "open_gripper": lambda: target.set_gripper_open(True),
        "close_gripper": lambda: target.set_gripper_open(False),
        "observe": target.move_to_observe,
    }
    callbacks[action]()
    return action


def main():
    parser = argparse.ArgumentParser(description="仅解析指令，不连接或控制机械臂")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    try:
        match = match_command(args.text)
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps({"text": args.text, "action": match.action,
                      "description": LABELS[match.action], "quality": match.quality},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
