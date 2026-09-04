# 📚⏱️ Study Timelapse Advanced

Turn your study grind into a cinematic 30-second timelapse ✨

While you study, it quietly captures webcam frames, uses YOLOv8 🧠 to score how focused you actually are and spot your study tools, tracks whatever's playing in the background 🎵, then stitches it all into a slick MP4 with a glassmorphism HUD 💎. There's also an analytics tab to flex your stats 📊, plus an optional on-device Phi-3 "AI coach" 🤖 that roasts (or hypes) your focus.

## ✨ Features

- 📸 **Live capture** — webcam frames at 1 Hz during a study session, with pause/resume and an auto-pause after ~2 minutes of you zoning out
- 🎯 **Focus & posture detection** — YOLOv8 (ONNX, running on DirectML / Ryzen NPU) scores your focus per frame and calls out slouching or leaning
- 📱💻📖 **Activity detection** — catches you on your phone, laptop, or reading a book
- 🎵 **Now-playing tracking** — logs whatever song was carrying your focus session
- 🎬 **Cinematic rendering** — dynamic pacing that slows down at the interesting moments (song changes, breaks) and speeds through the boring grind, topped off with a glassy HUD and a summary card
- 📊 **Analytics dashboard** — charts and stats across every past session
- 🤖 **AI coach** — an on-device Phi-3 model gives you feedback based on your data

## 🖥️ Requirements

- Windows (uses WinRT media APIs, DirectML, Segoe UI — not cross-platform, sorry Mac gang 🍎)
- A webcam 📷
- Python 3 (installed automatically into a venv by `run.bat`)

## 🚀 Setup

```bash
run.bat
```

That's it. It spins up a `.venv`, installs `requirements.txt`, and launches the app. Just run it from the repo root and you're good ✅.

On first run, YOLOv8 weights (`yolov8n.pt`, `yolov8n-pose.pt`) download automatically and get exported to ONNX. The Phi-3 AI coach model downloads on demand the first time you hit the button in the Analytics tab.

## 🏗️ Architecture

Three modules, each owning one phase of the pipeline:

- **`main.py`** 🖼️ — the CustomTkinter UI shell: a *Live Session* tab (webcam preview, session controls) and an *Analytics Dashboard* tab (charts, AI coach)
- **`capture_engine.py`** 🧠 — the background capture thread, focus/posture/activity detection via YOLOv8, and now-playing polling
- **`renderer.py`** 🎞️ — reads a session's `events.json`, applies the dynamic pacing algorithm, draws the HUD, and renders the final MP4

Session data lands in `sessions/<timestamp>/`, and rendered timelapses in `timelapses/`. Both are gitignored — they're generated locally on your machine, not part of the repo 🙈.

## 📝 Notes

- No automated test suite yet. `test_media.py` and `test_thread.py` are ad-hoc smoke scripts for the WinRT media API and the YOLO/Phi-3 DirectML coexistence check.
- See [`CLAUDE.md`](CLAUDE.md) for the deep-dive architecture and implementation reference 🔍.
