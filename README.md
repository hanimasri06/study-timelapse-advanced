# Study Timelapse Advanced

A Windows desktop app that turns a study session into a cinematic 30-second timelapse. While you study, it captures webcam frames, uses YOLOv8 to score your focus and detect study tools, tracks whatever's playing in the background, then renders an MP4 with a glassmorphism HUD. An analytics tab visualizes past sessions, with an optional on-device Phi-3 "AI coach" for feedback.

## Features

- **Live capture** — webcam frames at 1 Hz during a study session, with pause/resume and an auto-pause after ~2 minutes of detected unfocus
- **Focus & posture detection** — YOLOv8 (ONNX, running on DirectML / Ryzen NPU) scores focus per frame and flags slouching or leaning posture
- **Activity detection** — recognizes phone, laptop, and book usage via YOLOv8 object detection
- **Now-playing tracking** — polls Windows for the currently playing track during the session
- **Cinematic rendering** — dynamic pacing that slows down around key moments (music changes, pauses) and speeds through the rest, with a glassmorphism HUD and a summary card
- **Analytics dashboard** — charts and aggregates across all past sessions
- **AI coach** — optional on-device Phi-3 model generates feedback from session data

## Requirements

- Windows (uses WinRT media APIs, DirectML, Segoe UI — not cross-platform)
- A webcam
- Python 3 (installed automatically into a venv by `run.bat`)

## Setup

```bash
run.bat
```

This creates a `.venv`, installs `requirements.txt`, and launches the app. This is the only supported entry point — the app expects to be run from the repo root.

On first run, YOLOv8 weights (`yolov8n.pt`, `yolov8n-pose.pt`) are downloaded automatically and exported to ONNX. The Phi-3 AI coach model downloads on demand the first time you use it, from the Analytics tab.

## Architecture

Three modules, each owning one phase of the pipeline:

- **`main.py`** — the CustomTkinter UI shell: a *Live Session* tab (webcam preview, session controls) and an *Analytics Dashboard* tab (charts, AI coach)
- **`capture_engine.py`** — the background capture thread, focus/posture/activity detection via YOLOv8, and now-playing polling
- **`renderer.py`** — reads a session's `events.json`, applies the dynamic pacing algorithm, draws the HUD, and renders the final MP4

Session data is written to `sessions/<timestamp>/`, and rendered output to `timelapses/`. Both are gitignored — they're generated locally, not part of the repo.

## Notes

- No automated test suite. `test_media.py` and `test_thread.py` are ad-hoc smoke scripts for the WinRT media API and the YOLO/Phi-3 DirectML coexistence check.
- See [`CLAUDE.md`](CLAUDE.md) for a detailed architecture and implementation reference.
