# Transcriber

Push-to-talk dictation that runs **entirely offline on the CPU**. Hold a hotkey,
speak, release — the audio is transcribed locally and typed into whatever window
has focus. No audio ever leaves your machine.

## How it works

The whole app is a single ~390-line script, [`transcriber.py`](transcriber.py).
It's mostly glue:

1. **Capture** mic audio while the hotkey is held (`sounddevice`).
2. **Transcribe** the buffer with a pre-trained speech model running through
   ONNX Runtime on the CPU (`onnx-asr` + `onnxruntime`).
3. **Scrub** the text — remove filler words ("um", "uh", ...) and merge dictated
   digit runs ("nine, four, eight" → "948") via plain regex, no LLM.
4. **Type** the result into the focused window (`pynput`).

The "intelligence" is the speech model — a multi-hundred-MB neural network that
is **downloaded on first run** (cached under `~/.cache/huggingface`) and is *not*
part of this repo.

## Requirements

- Python 3.9+
- A microphone
- ~1–2 GB free disk for the model (downloaded once, on first run)

Install dependencies:

```sh
pip install -r requirements.txt
```

## Usage

Hold **Ctrl + Windows** together, speak, then release. The transcription is typed
at the cursor. Press **Ctrl+C** in the console window to quit.

Run directly:

```sh
python transcriber.py
```

Or use the Windows launcher:

- [`run.bat`](run.bat) — NVIDIA Parakeet (fast, English)

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device SUBSTRING` | `USB PnP` | Substring of the input device name. Pass `''` for the system default. |
| `--no-scrub` | off | Disable filler-word scrubbing; type the raw transcription. |
| `--threads N` | half your physical cores | Max CPU threads for inference, so other apps stay responsive. |
| `--priority {below,normal}` | `below` | Process priority during transcription. |

Example:

```sh
python transcriber.py --device "" --threads 4
```

## Notes

- **First run downloads the model** and may take a minute; later runs load from cache.
- Inference is capped (half your cores, below-normal priority) so dictation doesn't
  hog the machine.
- The trigger keys and filler-word lists are defined near the top of
  [`transcriber.py`](transcriber.py) if you want to customise them.
