#!/usr/bin/env python3
"""
Plauder: mic capture → wake word detection → local STT → emit JSON to stdout.

Usage:
    python3 plauder.py --model hey_jarvis --threshold 0.5 --cooldown 2.0
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

import faster_whisper
import numpy as np
import pyaudio

import openwakeword

# All warnings/logging -> stderr only
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger("plauder")
logger.setLevel(logging.INFO)

# Suppress onnxruntime CUDA warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

running = True

# --- State machine globals ---
_state = "IDLE"  # IDLE | LISTENING | TRANSCRIBING
_record_buffer: list[np.ndarray] = []
_speech_started = False
_silence_frames = 0
_listening_frame_count = 0
_last_emit_time: float = 0.0

# Audio constants (16 kHz, 16-bit mono)
CHUNK_SIZE = 1280
SAMPLE_RATE = 16000


def _signal_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _emit(event_dict: dict):
    """Print JSON line to stdout (flush immediately)."""
    print(json.dumps(event_dict), flush=True)


def _resolve_model_path(model_name: str) -> str:
    """Resolve short model name to full path, or return as-is if already a path."""
    known = openwakeword.models
    if model_name in known:
        return known[model_name]["model_path"]
    return model_name


def _reset_recording_state():
    """Reset all LISTENING/TRANSCRIBING state variables."""
    global _speech_started, _silence_frames, _listening_frame_count
    _speech_started = False
    _silence_frames = 0
    _listening_frame_count = 0


def _start_listening():
    """Transition from IDLE → LISTENING, reset buffer, emit event."""
    global _state, _record_buffer
    _state = "LISTENING"
    _record_buffer = []
    _reset_recording_state()
    _emit({"event": "listening"})


def _check_vad(chunk: np.ndarray, rms_threshold: float,
               silence_frame_limit: int, max_frames: int) -> bool:
    """
    Run VAD on chunk while in LISTENING state.
    Returns True if recording should stop (silence timeout or max duration).
    """
    global _speech_started, _silence_frames, _listening_frame_count

    _listening_frame_count += 1

    rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

    if not _speech_started:
        if rms > rms_threshold:
            _speech_started = True
            _silence_frames = 0
        # else: still waiting for first speech — don't count silence yet
    else:
        if rms < rms_threshold:
            _silence_frames += 1
        else:
            _silence_frames = 0

    # Silence timeout after speech started
    if _speech_started and _silence_frames >= silence_frame_limit:
        return True

    # Max recording duration
    if _listening_frame_count >= max_frames:
        return True

    return False


def _transcribe_and_emit(stt_model, stt_language):
    """
    Run transcription in background thread.
    Accesses global _record_buffer — safe because main thread is in TRANSCRIBING state.
    """
    global _state, _record_buffer, _last_emit_time

    try:
        if not _record_buffer:
            text = ""
        else:
            audio = np.concatenate(_record_buffer).astype(np.float32) / 32768.0
            lang = stt_language if stt_language else None  # None → auto-detect
            segments, info = stt_model.transcribe(audio, language=lang)
            text = " ".join(seg.text for seg in segments).strip()

        if text:
            _emit({"event": "transcription", "text": text})
        else:
            _emit({"event": "transcription_error", "message": "No speech detected"})
    except Exception as exc:
        _emit({"event": "transcription_error", "message": f"Transcription failed: {exc}"})
    finally:
        _record_buffer.clear()
        _last_emit_time = time.monotonic()
        _emit({"event": "idle"})
        _state = "IDLE"


def main():
    global _state, _last_emit_time

    parser = argparse.ArgumentParser(description="Plauder — wake word + local STT")
    parser.add_argument(
        "--model",
        default="hey_jarvis",
        help="Wake word model name or path (default: hey_jarvis)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between wake word emissions (default: 2.0)",
    )
    parser.add_argument(
        "--stt-model",
        default="tiny",
        help="Whisper model size (default: tiny)",
    )
    parser.add_argument(
        "--stt-language",
        default="",
        help="Language code for STT, empty = auto-detect (default: '')",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=0.8,
        help="Seconds of silence after speech to stop recording (default: 0.8)",
    )
    parser.add_argument(
        "--max-record-sec",
        type=float,
        default=10.0,
        help="Maximum recording duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=500.0,
        help="RMS amplitude threshold for silence detection (default: 500)",
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = _resolve_model_path(args.model)

    # --- PyAudio init (stderr suppressed to hide ALSA device spam) ---
    audio = None
    stream = None
    old_stderr = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception as exc:
        _emit({"event": "error", "message": f"PyAudio init (fd redirect) failed: {exc}"})
        sys.exit(1)

    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
    except Exception as exc:
        _emit({"event": "error", "message": f"PyAudio init failed: {exc}"})
        sys.exit(1)
    finally:
        if old_stderr is not None:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)

    # --- Load wake word model ---
    try:
        model = openwakeword.Model(wakeword_model_paths=[model_path])
    except Exception as exc:
        _emit({"event": "error", "message": f"Model load failed: {exc}"})
        sys.exit(1)

    _emit({"event": "ready"})

    # --- Load STT model (after ready) ---
    stt_model = None
    try:
        stt_model = faster_whisper.WhisperModel(
            args.stt_model, device="cpu", compute_type="int8"
        )
        logger.info("STT model '%s' loaded", args.stt_model)
    except Exception as exc:
        _emit({"event": "error", "message": f"STT model load failed: {exc}"})
        logger.warning("STT unavailable, proceeding with wake word only")

    # --- Compute derived constants ---
    silence_frame_limit = int(args.silence_timeout / (CHUNK_SIZE / SAMPLE_RATE))
    max_frames = int(args.max_record_sec * SAMPLE_RATE / CHUNK_SIZE)

    frame_count = 0

    while running:
        try:
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except Exception as exc:
            logger.error("Audio read error: %s", exc)
            break

        frame_count += 1
        chunk = np.frombuffer(raw, dtype=np.int16)

        if _state == "IDLE":
            # Wake word detection
            scores = model.predict(chunk)
            for model_name, score in scores.items():
                if score > args.threshold:
                    now = time.monotonic()
                    since_last = now - _last_emit_time
                    logger.info(
                        "DETECT frame=%d model=%s score=%.4f since_last=%.3fs "
                        "threshold=%.2f cooldown=%.1f",
                        frame_count, model_name, score, since_last,
                        args.threshold, args.cooldown,
                    )
                    if since_last >= args.cooldown:
                        _start_listening()
                        break

        elif _state == "LISTENING":
            _record_buffer.append(chunk)
            stop = _check_vad(chunk, args.rms_threshold,
                              silence_frame_limit, max_frames)
            if stop:
                _state = "TRANSCRIBING"
                if stt_model is not None:
                    threading.Thread(
                        target=_transcribe_and_emit,
                        args=(stt_model, args.stt_language),
                        daemon=True,
                    ).start()
                else:
                    _emit({"event": "transcription_error", "message": "STT model not available"})
                    _record_buffer.clear()
                    _last_emit_time = time.monotonic()
                    _state = "IDLE"
                    _emit({"event": "idle"})

        elif _state == "TRANSCRIBING":
            # Discard frames while transcription runs in background thread
            pass

    # --- Cleanup ---
    if stream is not None:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
    if audio is not None:
        try:
            audio.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
