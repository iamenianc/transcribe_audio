"""
Push-to-talk dictation tool.

Hold the hotkey (default: Ctrl + Windows key held together), speak, and release.
The captured audio is transcribed locally (NVIDIA Parakeet via ONNX, CPU) and
the resulting text is typed into whatever window currently has focus.

Alternatively, pass ``--file recording.mp3`` to transcribe a prerecorded media
file in batch (WAV, MP3, MP4, M4A, ...); the text is written to a ``.txt`` next
to the source (no mic, no hotkeys, no typing into a window). Non-WAV formats are
decoded with ffmpeg, which must be on PATH.

Everything runs offline on the CPU. No audio ever leaves the machine.
"""

import argparse
import contextlib
import datetime
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
import onnx_asr
import psutil
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller, Key

SAMPLE_RATE = 16000          # model expects 16 kHz mono
CHANNELS = 1
# Keys that must ALL be held down at once to trigger (hold-to-talk) recording.
TRIGGER_KEYS = {Key.ctrl_l, Key.cmd}
# Keys that, tapped together, TOGGLE a long hands-free session on/off (Win +
# Left-Shift). The toggle is edge-triggered so one tap fires once (see App).
TOGGLE_KEYS = {Key.cmd, Key.shift_l}

# Long-session transcripts are auto-saved here (one file per session) so the full
# text survives even if the target window is lost or stops accepting input.
TRANSCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")

# "Pure" fillers: sounds that are never legitimate English words, so they can be
# deleted wherever they appear as standalone tokens.
PURE_FILLERS = ["um", "uhm", "uh", "erm", "mm+", "mhm"]

# "Soft" fillers: usually disfluencies, but also valid interjections
# ("Ah, says Derek", "Oh!", "Eh?"). Only removed when they sit MID-sentence
# bracketed by commas (the disfluency signature, e.g. "I went, uh, there") --
# never at the start of an utterance where they're likely intentional.
SOFT_FILLERS = ["ah", "eh", "er", "oh", "huh"]

# A run of one or more PURE filler sounds, possibly joined by hyphens/spaces
# (e.g. "um", "mm-hmm"). \b boundaries keep real words like "humble", "summer".
_FILLER_ALT = "|".join(PURE_FILLERS)
_FILLER_GROUP = rf"\b(?:{_FILLER_ALT})(?:[-\s]+(?:{_FILLER_ALT}))*\b"

# A soft filler bracketed by commas on BOTH sides, mid-sentence only.
_SOFT_ALT = "|".join(SOFT_FILLERS)
_SOFT_FILLER_RE = re.compile(rf",\s*(?:{_SOFT_ALT})\s*,", re.IGNORECASE)

# Spoken number words 0-9 -> digit, so a dictated digit sequence can be merged.
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_REPEAT_WORDS = {"double": 2, "triple": 3, "quadruple": 4}
# Words that, when they directly precede a digit run, mean the digits are a
# reference/label (e.g. "section 9 4") and should NOT be merged into one number.
_NO_COLLAPSE_AFTER = {
    "section", "sections", "clause", "clauses", "chapter", "chapters",
    "paragraph", "paragraphs", "verse", "verses", "article", "articles",
    "rule", "rules", "step", "steps", "figure", "figures", "table", "tables",
    "page", "pages", "item", "items", "version", "level", "part", "parts",
    "phase", "exhibit", "appendix", "schedule", "note", "line", "point",
}
_PRECEDING_WORD = re.compile(r"([A-Za-z]+)\s*$")
# A single spoken digit: either a literal digit or a number word.
_DIGIT_TOKEN = r"(?:\d|" + "|".join(_DIGIT_WORDS) + r")"
# "double"/"triple"/... optionally followed by a digit token.
_REPEAT_TOKEN = r"(?:" + "|".join(_REPEAT_WORDS) + r")\s+" + _DIGIT_TOKEN
# A run of 2+ digit tokens / repeat-words separated only by spaces, commas, dashes.
_SEP = r"[\s,\-]+"
_DIGIT_RUN = re.compile(
    rf"\b(?:{_REPEAT_TOKEN}|{_DIGIT_TOKEN})(?:{_SEP}(?:{_REPEAT_TOKEN}|{_DIGIT_TOKEN}))+\b",
    re.IGNORECASE,
)
# Tokenizer for pulling the pieces out of a matched run.
_RUN_PIECE = re.compile(
    rf"(?:(?P<rep>{'|'.join(_REPEAT_WORDS)})\s+)?(?P<dig>{_DIGIT_TOKEN})",
    re.IGNORECASE,
)


def _expand_digit_run(match: re.Match) -> str:
    """Turn a matched spoken digit run into one concatenated number string."""
    # Don't merge when the run is a reference/label, e.g. "section 9 4".
    preceding = _PRECEDING_WORD.search(match.string, 0, match.start())
    if preceding and preceding.group(1).lower() in _NO_COLLAPSE_AFTER:
        return match.group(0)
    out = []
    for piece in _RUN_PIECE.finditer(match.group(0)):
        word = piece.group("dig").lower()
        digit = _DIGIT_WORDS.get(word, word)  # number word -> digit, or already a digit
        count = _REPEAT_WORDS.get((piece.group("rep") or "").lower(), 1)
        out.append(digit * count)
    return "".join(out)


def collapse_digit_runs(text: str) -> str:
    """Merge a dictated run of single digits into one number.

    "9, 4, 8, 1, double 1" -> "948111". Pure text; only fires on runs of 2+
    standalone digit tokens so ordinary prose numbers ("3 cats") are untouched.
    """
    if not text:
        return text
    return _DIGIT_RUN.sub(_expand_digit_run, text)


# Ordered text-cleaning rules. Applied top to bottom. Each is (pattern, repl).
_SCRUB_RULES = [
    # Collapse all whitespace runs to single spaces first.
    (re.compile(r"\s+"), " "),
    # Delete a PURE filler run together with the comma that trails it (that comma
    # was part of the disfluency, e.g. "..., um, ..."). A comma BEFORE the filler
    # belongs to the preceding real word and is intentionally kept.
    (re.compile(rf"{_FILLER_GROUP}\s*,?", re.IGNORECASE), ""),
    # Delete SOFT fillers (ah/oh/eh/er/huh) ONLY when bracketed by commas on both
    # sides mid-sentence ("I went, uh, there"). Leaving the leading comma keeps
    # sentence-initial interjections like "Ah, says Derek" untouched.
    (_SOFT_FILLER_RE, ","),
    # --- punctuation / spacing repair left behind by deletions ---
    (re.compile(r"\s+([,.;:!?])"), r"\1"),     # no space before punctuation
    (re.compile(r"(?:,\s*)+,"), ", "),          # collapse runs of commas to one
    (re.compile(r",+([.;:!?])"), r"\1"),        # drop comma stranded before end punct
    (re.compile(r"^[\s,]+"), ""),               # strip leading space/commas
    (re.compile(r"[\s,]+$"), ""),               # strip trailing space/commas
    (re.compile(r"^(\d+)\.$"), r"\1"),          # drop trailing period if whole text is digits
    (re.compile(r"\s{2,}"), " "),               # collapse doubled spaces
]


def scrub_disfluencies(text: str) -> str:
    """Remove spoken filler sounds (um, uh, ...) from transcribed text.

    Pure text/regex; no LLM, no network. Conservative by design: PURE fillers
    (um, uhm, erm, mm, mhm) are deleted anywhere; SOFT fillers (ah, oh, eh, er,
    huh) only when bracketed by commas mid-sentence, so genuine interjections
    like "Ah, says Derek" survive. Then tidies up the resulting punctuation.
    """
    if not text:
        return text
    # Merge dictated digit sequences ("9, 4, 8, 1" -> "9481") before the comma
    # cleanup rules run, since this consumes the separators between digits.
    text = collapse_digit_runs(text)
    for pattern, repl in _SCRUB_RULES:
        text = pattern.sub(repl, text)
    text = text.strip()
    # Recapitalize the first alphabetic character, if any.
    for i, ch in enumerate(text):
        if ch.isalpha():
            text = text[:i] + ch.upper() + text[i + 1:]
            break
    return text


class Recorder:
    """Records mic audio while active, on a background sounddevice stream."""

    def __init__(self, device=None):
        self.device = device
        self._frames = []
        self._q = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self._q.put(indata.copy())

    def start(self):
        self._frames = []
        self._q = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def _drain(self):
        while not self._q.empty():
            self._frames.append(self._q.get())

    def snapshot(self):
        """Return all audio captured so far WITHOUT stopping the stream.

        Used for live transcription during a long session: callers can poll this
        repeatedly, transcribe whatever new audio has accumulated, and keep
        recording. Returns a flat float32 array (empty if nothing captured yet).
        """
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._drain()
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames, axis=0).flatten()

    def stop(self):
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._drain()
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames, axis=0).flatten()


def split_on_silence(audio, sample_rate=SAMPLE_RATE, max_chunk_s=30.0,
                     min_silence_s=0.6, silence_rms=0.005, min_chunk_s=1.0):
    """Split long audio into model-friendly chunks, cutting only in pauses.

    Pure numpy, no deps. We never cut mid-utterance: a chunk grows until it has
    exceeded ``max_chunk_s``, then we cut at the NEXT silence gap of at least
    ``min_silence_s`` (a run of low-RMS frames). A continuous monologue with no
    such gap stays a single chunk. Audio shorter than ``max_chunk_s`` is returned
    unchanged as ``[audio]`` so short clips behave exactly as before.

    Returns a list of float32 arrays (silence between kept chunks is dropped).
    """
    n = audio.size
    if n == 0:
        return []
    if n <= int(max_chunk_s * sample_rate):
        return [audio]

    # Per-frame RMS over short hops to classify silence vs. speech.
    hop = max(1, int(0.02 * sample_rate))            # 20 ms frames
    n_frames = n // hop
    if n_frames == 0:
        return [audio]
    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    is_silent = rms < silence_rms

    max_frames = int(max_chunk_s * sample_rate / hop)
    min_silence_frames = max(1, int(min_silence_s * sample_rate / hop))
    min_chunk_frames = max(1, int(min_chunk_s * sample_rate / hop))

    chunks = []
    start = 0          # frame index where the current chunk begins
    i = 0
    while i < n_frames:
        # Once the current chunk is long enough, look for a silence gap to cut on.
        if i - start >= max_frames and is_silent[i]:
            j = i
            while j < n_frames and is_silent[j]:
                j += 1
            if j - i >= min_silence_frames:
                # Cut: keep [start, i) as a chunk, skip the silence, restart after it.
                if i - start >= min_chunk_frames:
                    chunks.append(audio[start * hop : i * hop])
                start = j
                i = j
                continue
        i += 1

    # Trailing chunk (everything from the last cut to the end of the audio).
    if n - start * hop >= min_chunk_frames * hop:
        chunks.append(audio[start * hop :])

    return chunks if chunks else [audio]


def split_completed_chunks(audio, sample_rate=SAMPLE_RATE, max_chunk_s=30.0,
                           min_silence_s=0.6, silence_rms=0.005, min_chunk_s=1.0):
    """Like ``split_on_silence`` but for LIVE use: only emit chunks that are
    closed by a real silence gap, and report how much audio was consumed.

    Returns ``(chunks, consumed_samples)``. The trailing audio after the last
    silence cut is assumed to be speech still in progress, so it is NOT emitted
    and NOT counted as consumed -- the caller keeps it and re-feeds it next poll
    once the speaker pauses. This guarantees we never type a half-spoken word.
    """
    n = audio.size
    if n == 0:
        return [], 0

    hop = max(1, int(0.02 * sample_rate))            # 20 ms frames
    n_frames = n // hop
    if n_frames == 0:
        return [], 0
    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    is_silent = rms < silence_rms

    max_frames = int(max_chunk_s * sample_rate / hop)
    min_silence_frames = max(1, int(min_silence_s * sample_rate / hop))
    min_chunk_frames = max(1, int(min_chunk_s * sample_rate / hop))

    chunks = []
    start = 0            # frame index where the current (open) chunk begins
    consumed_frames = 0  # frames past the end of the last completed silence gap
    i = 0
    while i < n_frames:
        if i - start >= max_frames and is_silent[i]:
            j = i
            while j < n_frames and is_silent[j]:
                j += 1
            # Only a gap that has actually ENDED (j < n_frames) is a real, closed
            # pause. A silence run still touching the buffer end may just be a brief
            # lull before more speech, so we leave it for the next poll.
            if j - i >= min_silence_frames and j < n_frames:
                if i - start >= min_chunk_frames:
                    chunks.append(audio[start * hop : i * hop])
                start = j
                consumed_frames = j
                i = j
                continue
        i += 1

    return chunks, consumed_frames * hop


@contextlib.contextmanager
def _below_normal_priority(enabled=True):
    """Temporarily drop this process to below-normal CPU priority.

    Keeps a heavy transcription from competing with the user's foreground apps.
    Best-effort: silently no-ops if psutil can't change priority.
    """
    if not enabled:
        yield
        return
    proc = psutil.Process()
    try:
        original = proc.nice()
    except Exception:
        yield
        return
    low = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10)  # Windows class; nice 10 on POSIX
    try:
        proc.nice(low)
    except Exception:
        pass
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            proc.nice(original)


# The transcription model: NVIDIA Parakeet TDT 0.6B, int8, CPU. English-only,
# fast, ~600MB (downloaded on first run).
MODEL = {
    "model": "nemo-parakeet-tdt-0.6b-v2",
    "quantization": "int8",
    "label": "NVIDIA Parakeet TDT 0.6B (English)",
    "size": "~600MB",
}


class Transcriber:
    """Transcribes audio on CPU via onnx-asr (NVIDIA Parakeet), capped so it
    doesn't hog the machine. Output is post-processed by scrub_disfluencies()."""

    def __init__(self, scrub=True, threads=None, low_priority=True):
        cfg = MODEL
        # Default to half the physical cores so the user's other apps keep CPU.
        if threads is None:
            cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 2
            threads = max(1, cores // 2)
        self.threads = threads
        self.scrub = scrub
        self.low_priority = low_priority

        # --- CPU safeguards live in the ONNX Runtime session options ---
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads     # cap parallelism within ops
        sess_options.inter_op_num_threads = 1           # no extra cross-op threads
        # Don't busy-wait spin while idle (otherwise threads burn cycles between calls).
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        print(f"[init] Loading {cfg['label']} "
              f"(first run downloads {cfg['size']}; capped to {threads} thread(s))...")
        self.model = onnx_asr.load_model(
            cfg["model"],
            quantization=cfg["quantization"],
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        print("[init] Model ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with _below_normal_priority(self.low_priority):
            text = self.model.recognize(audio, sample_rate=SAMPLE_RATE)
        text = (text or "").strip()
        if self.scrub:
            cleaned = scrub_disfluencies(text)
            if cleaned != text:
                print(f"[scrub] raw: {text!r}")
            text = cleaned
        return text


def load_via_ffmpeg(path):
    """Decode any media file (mp3, mp4, m4a, ...) to 16 kHz mono float32.

    Shells out to ffmpeg, which must be on PATH, and asks it to output raw
    32-bit float little-endian PCM at the model's sample rate / mono on stdout.
    ffmpeg handles the demux, decode, downmix, and resample, so we just wrap the
    bytes in a numpy array. Raises a clear error if ffmpeg is missing or fails.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"ffmpeg is required to read {os.path.splitext(path)[1]} files but was not "
            "found on PATH. Install it (e.g. `winget install Gyan.FFmpeg`) or convert "
            "the file to WAV first."
        )

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", path,
        "-f", "f32le",              # raw 32-bit float little-endian
        "-acodec", "pcm_f32le",
        "-ac", "1",                 # mono
        "-ar", str(SAMPLE_RATE),    # 16 kHz
        "-",                        # write to stdout
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}:\n{err}")

    audio = np.frombuffer(proc.stdout, dtype="<f4")
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_audio(path):
    """Load any supported media file into a 16 kHz mono float32 array.

    WAV files are read with the stdlib (no deps); everything else (mp3, mp4,
    m4a, etc.) is decoded via ffmpeg.
    """
    if os.path.splitext(path)[1].lower() == ".wav":
        return load_wav(path)
    return load_via_ffmpeg(path)


def load_wav(path):
    """Read a WAV file into a 16 kHz mono float32 array for the model.

    Uses only the stdlib ``wave`` module + numpy (no extra deps, in keeping with
    the app's offline/minimal footprint). Handles 8/16/24/32-bit integer PCM,
    downmixes multi-channel to mono, normalizes to [-1, 1], and resamples to
    ``SAMPLE_RATE`` by linear interpolation.
    """
    import wave

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()          # bytes per sample
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    # Decode PCM bytes to float32 in [-1, 1] based on sample width.
    if sampwidth == 1:                         # 8-bit PCM is unsigned, centered at 128
        ints = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (ints - 128.0) / 128.0
    elif sampwidth == 2:                       # 16-bit signed
        ints = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        audio = ints / 32768.0
    elif sampwidth == 3:                       # 24-bit signed, packed 3 bytes/sample
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)  # sign-extend
        audio = ints.astype(np.float32) / 8388608.0
    elif sampwidth == 4:                       # 32-bit signed
        ints = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        audio = ints / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    # Downmix to mono by averaging channels (frames are interleaved).
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample to the model's expected rate by linear interpolation.
    if rate != SAMPLE_RATE and audio.size:
        n_out = int(round(audio.size * SAMPLE_RATE / rate))
        src = np.linspace(0.0, audio.size - 1, num=n_out, dtype=np.float64)
        audio = np.interp(src, np.arange(audio.size), audio).astype(np.float32)

    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe_file(path, transcriber, max_chunk_s=30.0, min_silence_s=0.6):
    """Transcribe a prerecorded media file and write the text to a .txt beside it.

    Accepts WAV (stdlib) plus anything ffmpeg can decode (mp3, mp4, m4a, ...).
    Reuses the same silence-aware chunking as a long mic session so arbitrarily
    long files are handled. Returns the output path.
    """
    print(f"[file] Loading {path}...")
    audio = load_audio(path)
    dur = audio.size / SAMPLE_RATE
    print(f"[file] {dur:.1f}s of audio; transcribing...")

    chunks = split_on_silence(
        audio, sample_rate=SAMPLE_RATE,
        max_chunk_s=max_chunk_s, min_silence_s=min_silence_s,
    )
    lines = []
    for n, chunk in enumerate(chunks, 1):
        text = transcriber.transcribe(chunk)
        if text:
            print(f"[file] chunk {n}/{len(chunks)}: {text}")
            lines.append(text)

    out_path = os.path.splitext(path)[0] + ".txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"[file] Wrote transcript to {out_path}")
    return out_path


class App:
    """Two ways to dictate:
      - Hold Ctrl+Win to record, release to transcribe & type (push-to-talk).
      - Tap Win+Left-Shift to start a long hands-free session; tap again to stop.
        During the session each silence-bounded chunk is transcribed and typed
        LIVE as you speak (keep focus on the target window), so a failure costs
        only one chunk rather than the whole hour.
    """

    def __init__(self, recorder, transcriber, max_chunk_s=30.0, min_silence_s=0.6):
        self.recorder = recorder
        self.transcriber = transcriber
        self.max_chunk_s = max_chunk_s
        self.min_silence_s = min_silence_s
        self.kbd = Controller()
        self._pressed = set()
        self._recording = False     # hold-to-talk active
        self._session = False       # long toggle session active
        self._toggle_armed = True   # edge-trigger guard for the toggle combo
        self._session_stop = None   # threading.Event signalling the live worker to finish
        self._worker = None         # live-transcription worker thread
        self._transcript = None     # open file for the current session's auto-saved transcript
        self._lock = threading.Lock()

    def _on_press(self, key):
        self._pressed.add(self._normalize(key))

        # --- Long-session toggle (Win+Left-Shift), edge-triggered ---
        if TOGGLE_KEYS.issubset(self._pressed):
            if self._toggle_armed:
                self._toggle_armed = False   # one physical tap fires once
                self._toggle_session()
            return

        # --- Hold-to-talk (Ctrl+Win) -- suppressed while a session is active ---
        if (not self._session and not self._recording
                and TRIGGER_KEYS.issubset(self._pressed)):
            with self._lock:
                if not self._recording:
                    self._recording = True
                    print("\n[rec] Recording... (release keys to transcribe)")
                    self.recorder.start()

    def _on_release(self, key):
        self._pressed.discard(self._normalize(key))

        # Re-arm the toggle once the combo is broken, so the next full press fires.
        if not TOGGLE_KEYS.issubset(self._pressed):
            self._toggle_armed = True

        # A long session owns the recorder; never let release end it (the toggle does).
        if self._session:
            return

        if self._recording and not TRIGGER_KEYS.issubset(self._pressed):
            with self._lock:
                if self._recording:
                    self._recording = False
                    audio = self.recorder.stop()
                    threading.Thread(target=self._handle, args=(audio,), daemon=True).start()

    def _toggle_session(self):
        with self._lock:
            if not self._session:
                # Don't start a session on top of a hold-to-talk recording.
                if self._recording:
                    return
                self._session = True
                self._open_transcript()
                print("\n[long] Long session recording (typing live as you speak)... "
                      "(tap Win+Left-Shift again to stop)")
                self.recorder.start()
                self._session_stop = threading.Event()
                self._worker = threading.Thread(target=self._live_worker, daemon=True)
                self._worker.start()
            else:
                self._session = False
                print("[long] Stopping...")
                # Signal the worker; it stops the recorder and flushes the tail.
                self._session_stop.set()

    @staticmethod
    def _normalize(key):
        # Collapse left/right variants into the generic keys used in the triggers.
        # Left-Shift is kept distinct on purpose: the long-session toggle is bound
        # specifically to Win + Left-Shift, so right shift must not stand in for it.
        if key == Key.shift_r:
            return Key.shift
        if key in (Key.cmd_l, Key.cmd_r):
            return Key.cmd
        return key

    def _type(self, text):
        # pynput's kbd.type() bursts characters as fast as the OS accepts them,
        # which makes Notepad (and other windows) drop characters -- producing
        # garbled output with whole words missing. Type one char at a time with a
        # tiny delay so the target window keeps up.
        for ch in text:
            self.kbd.type(ch)
            time.sleep(0.006)

    def _handle(self, audio):
        dur = audio.size / SAMPLE_RATE
        print(f"[rec] Captured {dur:.1f}s, transcribing...")
        text = self.transcriber.transcribe(audio)
        if not text:
            print("[out] (nothing recognised)")
            return
        print(f"[out] {text}")
        self._type(text + " ")

    def _live_worker(self):
        """Drive a long session: type each completed chunk as it's ready.

        Polls the still-running recorder, splits off only chunks that are closed
        by a real pause (never a half-spoken word), transcribes and types each,
        and tracks how much audio has been consumed. On stop it drains the final
        tail through ``split_on_silence`` so the last (open) utterance is typed.
        """
        consumed = 0  # samples already transcribed-and-typed from the buffer
        while not self._session_stop.is_set():
            self._session_stop.wait(2.0)   # poll cadence; wakes early on stop
            buf = self.recorder.snapshot()
            pending = buf[consumed:]
            chunks, used = split_completed_chunks(
                pending, sample_rate=SAMPLE_RATE,
                max_chunk_s=self.max_chunk_s, min_silence_s=self.min_silence_s,
            )
            for chunk in chunks:
                self._type_chunk(chunk)
            consumed += used

        # --- stop: close the stream and flush the remaining tail ---
        audio = self.recorder.stop()
        tail = audio[consumed:]
        print(f"[long] Stopped. Flushing final {tail.size / SAMPLE_RATE:.1f}s...")
        for chunk in split_on_silence(
            tail, sample_rate=SAMPLE_RATE,
            max_chunk_s=self.max_chunk_s, min_silence_s=self.min_silence_s,
        ):
            self._type_chunk(chunk)
        self._close_transcript()
        print("[long] Session complete.")

    def _type_chunk(self, chunk):
        """Transcribe one audio chunk and type it on its own line."""
        text = self.transcriber.transcribe(chunk)
        if not text:
            return
        print(f"[out] {text}")
        self._save_line(text)   # persist BEFORE typing, so a lost window never loses text
        self._type(text + "\n")

    def _open_transcript(self):
        """Start a fresh auto-save file for this long session.

        One file per session, named by start time. Each completed chunk is
        appended immediately (and flushed to disk) so the full transcript
        survives even if the focused window stops accepting typed input.
        Best-effort: failure to open just disables saving for this session.
        """
        try:
            os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(TRANSCRIPT_DIR, f"session-{stamp}.txt")
            self._transcript = open(path, "a", encoding="utf-8")
            print(f"[long] Saving transcript to {path}")
        except Exception as exc:
            self._transcript = None
            print(f"[long] Could not open transcript file ({exc}); continuing without auto-save.",
                  file=sys.stderr)

    def _save_line(self, text):
        """Append one transcribed line to the session file and flush to disk."""
        f = self._transcript
        if f is None:
            return
        try:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())   # force to disk so an abrupt exit can't lose it
        except Exception as exc:
            print(f"[long] Failed to write transcript ({exc}).", file=sys.stderr)

    def _close_transcript(self):
        if self._transcript is not None:
            with contextlib.suppress(Exception):
                self._transcript.close()
            self._transcript = None

    def run(self):
        print("\nReady. Hold Ctrl+Win to dictate, or tap Win+Left-Shift for a "
              "long session. Ctrl+C in this window to quit.\n")
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()


def pick_device(name_substring):
    if name_substring is None:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_substring.lower() in d["name"].lower():
            print(f"[init] Using input device [{i}] {d['name']}")
            return i
    print(f"[init] No input device matched '{name_substring}'; using system default.")
    return None


def main():
    p = argparse.ArgumentParser(description="Push-to-talk local dictation "
                                            "(NVIDIA Parakeet, ONNX, CPU).")
    p.add_argument("--file", default=None,
                   help="transcribe a prerecorded media file (WAV, MP3, MP4, M4A, ...) in "
                        "batch and write the text to a .txt next to it, then exit (no mic, "
                        "no hotkeys). Non-WAV formats require ffmpeg on PATH.")
    p.add_argument("--device", default="USB PnP",
                   help="substring of the input device name (default: 'USB PnP'). "
                        "Pass '' to use the system default.")
    p.add_argument("--no-scrub", action="store_true",
                   help="disable filler-word scrubbing (type the raw transcription).")
    p.add_argument("--threads", type=int, default=None,
                   help="max CPU threads for inference (default: half your physical cores, "
                        "so other apps stay responsive).")
    p.add_argument("--priority", choices=["below", "normal"], default="below",
                   help="process priority during transcription (default: below normal).")
    p.add_argument("--max-chunk", type=float, default=30.0,
                   help="long-session: target max seconds per chunk before cutting at the "
                        "next pause (default: 30).")
    p.add_argument("--min-silence", type=float, default=0.6,
                   help="long-session: minimum pause length (seconds) that counts as a "
                        "cut point (default: 0.6).")
    args = p.parse_args()

    transcriber = Transcriber(
        scrub=not args.no_scrub,
        threads=args.threads,
        low_priority=(args.priority == "below"),
    )

    # Batch mode: transcribe a prerecorded WAV and exit (no mic, no hotkeys).
    if args.file:
        transcribe_file(args.file, transcriber,
                        max_chunk_s=args.max_chunk, min_silence_s=args.min_silence)
        return

    device = pick_device(args.device if args.device else None)
    recorder = Recorder(device=device)
    App(recorder, transcriber,
        max_chunk_s=args.max_chunk, min_silence_s=args.min_silence).run()


if __name__ == "__main__":
    main()
