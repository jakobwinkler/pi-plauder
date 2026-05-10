# pi-plauder

Wake word triggered STT for [pi coding agent](https://github.com/earendil-works/pi-coding-agent). Say "Hey Jarvis" (or custom wake word - <https://openwakeword.com/library>), speak, get transcribed text inserted into chat.

Uses [openWakeWord](https://github.com/dscripka/openWakeWord) for detection + [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local STT.

## Design Goals

Inspired by [pi-listen](https://github.com/codexstar69/pi-listen) and [friday](https://github.com/dantetekanem/friday), but focused on:

- **Local-only**: no cloud APIs, everything runs on your machine
- **Simple**: minimal moving parts, easy to understand and modify - KISS!

## Install

```bash
# Python deps
pip install openwakeword faster-whisper pyaudio

# pi extension, e.g.
pi install https://github.com/jakobwinkler/pi-plauder 
```

Configure in `~/.pi/agent/settings.json`:

```json
{
  "plauder": {
    "model": "hey_jarvis",
    "threshold": 0.5,
    "cooldown": 2.0,
    "stt_model": "tiny.en",
    "stt_language": "",
    "silence_timeout": 0.8,
    "max_record_sec": 10.0,
    "rms_threshold": 500,
    "autosubmit": false
  }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `hey_jarvis` | Wake word model (built-in: `alexa`, `hey_mycroft`, `hey_jarvis`, `timer`, `weather`) or path to `.onnx` file in `~/.pi/agent/plauder/` |
| `threshold` | `0.5` | Detection confidence (0-1). Lower = more sensitive |
| `cooldown` | `2.0` | Min seconds between wake word triggers |
| `stt_model` | `tiny` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, or `.en` variants) |
| `stt_language` | `""` | Language code (empty = auto-detect) |
| `silence_timeout` | `0.8` | Seconds of silence to end recording |
| `max_record_sec` | `10.0` | Max recording length |
| `rms_threshold` | `500` | RMS amplitude for speech/silence threshold |
| `autosubmit` | `false` | Submit transcription directly vs insert in editor |

## Usage

Say wake word → mic opens → speak → transcription appears.

Commands in pi:

- `/plauder help` - Show help
- `/plauder toggle` - Start/stop listener
- `/plauder autosubmit` - Toggle instant submit mode

## Custom Models

Place `.onnx` files in `~/.pi/agent/plauder/` and reference by name (without extension):

```json
{ "plauder": { "model": "zugzug" } }
```

This loads `~/.pi/agent/plauder/zugzug.onnx`.
