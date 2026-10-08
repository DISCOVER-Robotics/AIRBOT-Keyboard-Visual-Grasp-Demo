#!/usr/bin/env python3
"""Persistent FunASR worker using newline-delimited JSON on stdin/stdout."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys
import tempfile
from threading import Event, Lock, Thread


output_lock = Lock()


def emit(event, **payload):
    with output_lock:
        print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def load_asr(model_name, device):
    with redirect_stdout(sys.stderr):
        from funasr import AutoModel
        return AutoModel(model=model_name, hub="ms", vad_model="fsmn-vad",
                         device=device, disable_update=True, disable_pbar=True)


def transcribe(model, audio_path):
    with redirect_stdout(sys.stderr):
        result = model.generate(input=audio_path, batch_size_s=60)
    if not isinstance(result, list):
        raise ValueError("语音识别返回格式异常")
    text = "".join(item.get("text", "") for item in result
                   if isinstance(item, dict)).strip()
    if not text:
        raise ValueError("未识别到语音")
    return text


class SpeechEndpoint:
    def __init__(self, threshold, silence_seconds, minimum_speech=0.2):
        self.threshold = threshold
        self.silence_seconds = silence_seconds
        self.minimum_speech = minimum_speech
        self.speech_started = False
        self.speech_duration = 0.0
        self.silence_duration = 0.0

    def feed(self, rms, duration):
        if rms >= self.threshold:
            self.speech_started = True
            self.speech_duration += duration
            self.silence_duration = 0.0
        elif self.speech_started:
            self.silence_duration += duration
        epsilon = 1e-9
        return (self.speech_duration + epsilon >= self.minimum_speech
                and self.silence_duration + epsilon >= self.silence_seconds)

    @property
    def has_speech(self):
        return (self.speech_started
                and self.speech_duration + 1e-9 >= self.minimum_speech)


class Recorder:
    def __init__(self, model):
        self.model = model
        self.cancel = Event()
        self.thread = None
        self.lock = Lock()

    def start(self, request):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                emit("error", message="录音正在进行")
                return
            self.cancel.clear()
            self.thread = Thread(target=self._record, args=(request,), daemon=True)
            self.thread.start()

    def stop(self):
        self.cancel.set()

    def _record(self, request):
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        device = request.get("device")
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        sample_rate = 16000
        block_seconds = 0.1
        block_size = int(sample_rate * block_seconds)
        max_seconds = min(10.0, max(1.0, float(request.get("max_seconds", 4.0))))
        silence_seconds = min(2.0, max(0.3, float(request.get("silence_seconds", 0.8))))
        energy_threshold = min(0.2, max(0.001, float(request.get("energy_threshold", 0.012))))
        minimum_speech = 0.2
        chunks = []
        endpoint = SpeechEndpoint(energy_threshold, silence_seconds, minimum_speech)
        elapsed = 0.0
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=sample_rate,
                                    dtype="float32")
            emit("listening")
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                blocksize=block_size, device=device) as stream:
                while elapsed < max_seconds and not self.cancel.is_set():
                    data, overflowed = stream.read(block_size)
                    if overflowed:
                        print("警告：录音输入溢出", file=sys.stderr, flush=True)
                    chunk = np.asarray(data, dtype=np.float32).copy()
                    chunks.append(chunk)
                    elapsed += block_seconds
                    rms = float(np.sqrt(np.mean(np.square(chunk))))
                    if endpoint.feed(rms, block_seconds):
                        break
            if self.cancel.is_set():
                emit("cancelled")
                return
            if not endpoint.has_speech:
                emit("no_speech")
                return
            emit("transcribing")
            audio = np.concatenate(chunks, axis=0)
            with tempfile.TemporaryDirectory(prefix="grasp-voice-") as directory:
                path = str(Path(directory) / "command.wav")
                sf.write(path, audio, sample_rate, subtype="PCM_16")
                emit("transcript", text=transcribe(self.model, path))
        except Exception as exc:
            emit("error", message=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-model", default="paraformer-zh")
    args = parser.parse_args()
    try:
        model = load_asr(args.asr_model, args.asr_device)
    except Exception as exc:
        emit("fatal", message=str(exc))
        return 1
    recorder = Recorder(model)
    emit("ready")
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if not isinstance(message, dict) or "command" not in message:
                raise ValueError("工作进程协议消息无效")
            command = message["command"]
            if command == "record":
                recorder.start(message)
            elif command == "cancel":
                recorder.stop()
            elif command == "shutdown":
                recorder.stop()
                if recorder.thread is not None:
                    recorder.thread.join(timeout=3)
                return 0
            else:
                raise ValueError(f"未知工作进程命令：{command}")
        except Exception as exc:
            emit("error", message=str(exc))
    recorder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
