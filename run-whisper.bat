@echo off
REM Launch the push-to-talk dictation tool using Whisper small.en (CPU, balanced).
REM Alternative to run.bat (which uses NVIDIA Parakeet). Same hotkey & scrubbing.
py "%~dp0transcriber.py" --engine whisper --device "USB PnP"
pause
