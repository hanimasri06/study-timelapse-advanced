# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

"Study Timelapse Advanced" — a Windows desktop app (Python + CustomTkinter) that turns a study session into a cinematic 30-second timelapse. While the user studies, it captures webcam frames at 1 Hz, uses YOLOv8 (ONNX on DirectML / Ryzen NPU) to score focus and detect study tools, polls Windows for the currently-playing track, then renders an mp4 with a glassmorphism HUD. An analytics tab visualizes past sessions and an optional on-device Phi-3 "AI coach" generates feedback.

Platform-specific: depends on Windows-only APIs (`winrt`, `DmlExecutionProvider`, Segoe UI fonts). Do not port code paths to assume cross-platform availability.

## Running the app

- `run.bat` — creates `.venv`, installs `requirements.txt`, launches `python main.py`. This is the only supported entry point.
- After first setup, you can also run `python main.py` inside the activated venv.
- There is **no test framework**. `test_media.py` and `test_thread.py` are ad-hoc smoke scripts for the WinRT media API and the YOLO+Phi-3 DirectML coexistence check — run with `python test_media.py` / `python test_thread.py`.

## Architecture

Three modules, each owning one phase of the pipeline:

1. **`main.py` — `TimelapseApp` (CTk root)**. UI shell with two tabs: *Live Session* (webcam preview + start/pause/resume/end + mini-player toplevel) and *Analytics Dashboard* (aggregates over all sessions, matplotlib charts, Phi-3 coach). Two recurring `self.after` loops: preview at 30 ms and status update at 1 s. Owns one `CaptureEngine` instance.
2. **`capture_engine.py` — `CaptureEngine`**. Background capture thread; the camera is owned by the engine during a session and by `TimelapseApp.preview_cap` otherwise (see "Camera handoff" below). Also spawns a WinRT polling thread that updates `current_song_title`/`current_song_artist` every second, and a YOLO init thread that lazy-exports `.pt → .onnx` and loads two ONNX sessions (detect + pose) on DirectML.
3. **`renderer.py` — `Renderer`**. Reads `events.json`, applies the dynamic pacing algorithm, draws the HUD onto each selected frame, writes the final mp4. Triggered from `TimelapseApp._render_video` on a worker thread after the user clicks "End & Render".

### Data on disk

- `sessions/<YYYYMMDD_HHMMSS>/frames.mp4` — H.264 video at the camera's native 2560×1440, 1 fps. Written incrementally via `cv2.VideoWriter('avc1')` during capture; falls back to `mp4v` if avc1 fails. **The mp4 is only finalized on `VideoWriter.release()`** (writes the moov atom) — if the app crashes mid-session the file may be unreadable. `events.json` survives.
- `sessions/<YYYYMMDD_HHMMSS>/events.json` — the **single source of truth** for analytics and rendering. Schema: `{ "start_time": <unix>, "events": [ {type, timestamp, [frame_index, focused, music, activity, posture]} ] }` where `type ∈ {start, frame, pause, resume, end}` and `posture ∈ {good, slouching, leaning, unknown}`. `frame_index` is the position into `frames.mp4` (0-based, monotonically increasing — paused frames are not written and not indexed). Every event write rewrites the whole file (`_log_event` in [capture_engine.py:147](capture_engine.py)).
- **Legacy sessions** (pre-mp4 migration) used `sessions/<id>/frames/frame_NNNNNN.jpg` and `event["filename"]` instead of `frame_index`. The current renderer refuses to re-render these (prints a message and returns `None`); analytics still works since it doesn't touch frame data. Pre-posture sessions also lack the `posture` field — analytics must handle missing keys.
- `timelapses/timelapse_<timestamp>.mp4` — rendered output.
- `models/phi3/directml/directml-int4-awq-block-128/` — Phi-3 model, downloaded on demand the first time the user clicks "Generate AI Coach Insight".
- `yolov8n{,-pose}.{pt,onnx}` — YOLO weights at the repo root; `.onnx` is auto-exported from `.pt` on first run.

### Focus detection logic ([capture_engine.py:184](capture_engine.py))

Two YOLOv8 ONNX models run per frame, both at 640×640:
- **Pose** → two heuristics from the same inference. Keypoint layout is `preds_pose[5 + kpt*3 + (0/1/2), best_idx]` for x/y/conf, COCO order (0=nose, 1=l_eye, 2=r_eye, 5=l_shoulder, 6=r_shoulder).
  - *Focus*: require nose/eye confidence > 0.5; `focused=True` if nose_x lies within `[min(eye_x), max(eye_x)] ± 70% of eye spread`. Wide margin is intentional for dual-monitor setups.
  - *Posture*: require nose + both shoulders > 0.5 confidence. Compute `head_above = mean(shoulder_y) - nose_y` and `shoulder_dx = |ls_x - rs_x|`. Posture is `slouching` if `head_above / shoulder_dx < 0.35` (head sinking toward shoulders), `leaning` if `shoulder_dy / shoulder_dx > 0.25` (visibly tilted shoulder line), otherwise `good`. Thresholds are rough — if user complains of false positives, tune them down.
- **Detect** → uses **hardcoded COCO class indices**: 67=cell phone (forces focused=False, activity="Distracted (Phone)"), 63=laptop, 73=book. If you swap models or classes, update these literals.

Auto-pause: 120 consecutive unfocused frames (~2 minutes at 1 fps) calls `pause_session(auto=True)`. Auto-resume requires 3 consecutive focused frames. Manual pauses do not auto-resume.

### Renderer dynamic pacing ([renderer.py:78](renderer.py))

Target output is `target_video_duration=30` s at `fps=30` (900 frames). The algorithm computes `base_step = total_frames // 900`, then walks frames using `slow_step = base_step // 4` near "key events" (music change, pause) and `fast_step = base_step * 1.5` elsewhere — so the timelapse slows down at narratively interesting moments. Pauses are mapped to selected frames and rendered as fade-out → text card ("X mins pause") → fade-in. A final 3-second summary card with total study time and focus score is appended.

### HUD ([renderer.py:200](renderer.py))

Glassmorphism is implemented in OpenCV/PIL: extract ROI at top-right → `GaussianBlur(55,55)` → darken → composite under a PIL `rounded_rectangle` alpha mask → add a 1px white border. Text is drawn via PIL with Segoe UI (fallback to Arial, then default bitmap font). The HUD is silently skipped if the frame is too small to fit it.

### AI Coach — the DirectML contention workaround ([main.py:451](main.py))

This is the most subtle piece of code in the project. YOLO holds the DirectML device for the entire session, so loading Phi-3 with DML would conflict. The workaround:
1. Read `genai_config.json` from the downloaded model.
2. Save `provider_options`, then **overwrite them to `[]`** (forces CPU).
3. Load `og.Model(...)` (now on CPU).
4. In a `finally` block, restore the original `provider_options` so a future standalone run can still use DML.

Any future change to how the LLM is loaded must preserve this save-clear-restore dance, or remove it only after also reworking how YOLO holds the DML device.

### Camera handoff

There is a single physical webcam. `TimelapseApp` opens `self.preview_cap` at startup for the live preview; `start_session` sets `is_previewing = False` and **releases** that capture so the engine's `_capture_loop` can open its own `cv2.VideoCapture(0)`. `_render_complete` re-opens the preview capture. If you add a new code path that touches the camera, follow this same release-before-open ownership pattern or both readers will fight.

## Conventions worth knowing

- All file I/O assumes the CWD is the project root (`os.path.join(os.getcwd(), "sessions" / "timelapses" / "models")`). The app must be launched from the repo root — `run.bat` does this; if you add a new entry point, preserve it.
- Threads are `daemon=True` everywhere; the app relies on process exit to clean them up rather than explicit shutdown signaling. `end_session` is the one exception — it `.join()`s the capture thread before returning.
- UI updates from worker threads happen via direct widget `.configure()` calls (no `after(0, ...)` marshalling for most paths). This works in CustomTkinter / Tk but is technically not thread-safe; if you see flicker or crashes, that's the likely cause.
- The Phi-3 imports (`onnxruntime_genai`, `huggingface_hub`) are intentionally lazy — they're heavy and only loaded when the coach button is clicked. Keep them lazy.
