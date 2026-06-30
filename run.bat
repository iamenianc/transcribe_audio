@echo off
REM Launch the push-to-talk dictation tool (NVIDIA Parakeet, CPU).
REM --threads caps CPU use (default: half your cores).
py "%~dp0transcriber.py" --device "USB PnP"
pause
