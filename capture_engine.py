import cv2
import threading
import time
import os
import json
import asyncio
import hashlib
import ctypes
import copy
from datetime import datetime
import numpy as np
from chroma_feedback import ChromaFeedback
from fatigue_detector import FatigueDetector
from whoop_client import WhoopClient

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
    WINSDK_AVAILABLE = True
except ImportError:
    WINSDK_AVAILABLE = False


class CaptureEngine:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.auto_paused = False # Flag for auto-pause state
        self.thread = None
        self.stop_event = threading.Event()
        self.cap = None
        
        self.session_dir = ""
        self.frames_dir = ""
        self.assets_dir = ""
        self.events_file = ""
        self.frames_video_path = ""
        self.video_writer = None
        
        self.events = []
        self.events_lock = threading.Lock()
        self.events_write_lock = threading.Lock()
        self.events_flush_interval = 5.0
        self.last_events_flush = 0.0
        self.start_time = 0
        self.total_focus_time_sec = 0.0
        self.total_break_time_sec = 0.0
        
        self.chroma = ChromaFeedback()
        
        self.capture_interval = 1.0
        self.auto_pause_seconds = 120
        self.consecutive_slouch_seconds = 0.0
        self.distracted_time_seconds = 0.0
        self.media_paused_by_us = False
        self.capture_width = 2560
        self.capture_height = 1440
        self.capture_codec = "mp4v"
        self.privacy_mode = "Off"
        self.session_goal = ""
        
        self.current_song_title = ""
        self.current_song_artist = ""
        self.current_song_cover_bytes = b""
        self.current_song_cover_hash = ""
        self.current_posture = "unknown"  # good | slouching | leaning | unknown
        self.current_activity = "None"
        self.current_focused = False
        self.current_fatigue = {}
        self.frames_captured = 0
        self.last_inference_ms = 0.0
        self.last_loop_ms = 0.0
        self.backend_status = {
            "state": "Initializing",
            "selected": "Pending",
            "available": [],
            "benchmark_ms": None,
            "message": "Vision backend is warming up",
        }
        self.media_lock = threading.Lock()
        self.whoop_client = WhoopClient()
        self.whoop_lock = threading.Lock()
        self.current_whoop = self.whoop_client.cached_metrics()
        self.whoop_poll_seconds = 600
        self.fatigue_detector = FatigueDetector()
        self.fatigue_lock = threading.Lock()
        self.current_fatigue = self.fatigue_detector.snapshot()
        self.fatigue_pause_until = 0.0
        self.fatigue_autopause_cooldown_until = 0.0

        # State tracking
        self.unfocused_counter = 0
        self.focused_counter = 0
        
        self.ort_session_detect = None
        self.ort_session_pose = None
        self.detect_input_name = None
        self.pose_input_name = None
        if ONNX_AVAILABLE:
            threading.Thread(target=self._init_yolo_npu, daemon=True).start()
        else:
            self.backend_status.update({
                "state": "Unavailable",
                "selected": "None",
                "available": [],
                "message": "onnxruntime is not installed",
            })
        
        if WINSDK_AVAILABLE:
            self.media_thread = threading.Thread(target=self._media_polling_loop, daemon=True)
            self.media_thread.start()

        self.whoop_thread = threading.Thread(target=self._whoop_polling_loop, daemon=True)
        self.whoop_thread.start()

    def configure(self, capture_interval=None, auto_pause_seconds=None, privacy_mode=None,
                  session_goal=None, capture_resolution=None, capture_codec=None):
        if capture_interval is not None:
            self.capture_interval = max(0.25, float(capture_interval))
        if auto_pause_seconds is not None:
            self.auto_pause_seconds = max(10, int(auto_pause_seconds))
        if privacy_mode is not None:
            self.privacy_mode = privacy_mode
        if session_goal is not None:
            self.session_goal = session_goal.strip()
        if capture_codec is not None:
            self.capture_codec = capture_codec
        if capture_resolution:
            try:
                width, height = capture_resolution.lower().split("x")
                self.capture_width = int(width)
                self.capture_height = int(height)
            except ValueError:
                pass

    def get_live_snapshot(self):
        with self.whoop_lock:
            whoop_snapshot = copy.deepcopy(self.current_whoop)
        with self.fatigue_lock:
            fatigue_snapshot = copy.deepcopy(self.current_fatigue)
        if self.fatigue_pause_until:
            fatigue_snapshot["break_remaining_seconds"] = max(0, int(round(self.fatigue_pause_until - time.time())))
        return {
            "focused": self.current_focused,
            "posture": self.current_posture,
            "activity": self.current_activity,
            "frames_captured": self.frames_captured,
            "inference_ms": self.last_inference_ms,
            "loop_ms": self.last_loop_ms,
            "backend": dict(self.backend_status),
            "privacy_mode": self.privacy_mode,
            "capture_interval": self.capture_interval,
            "auto_pause_seconds": self.auto_pause_seconds,
            "goal": self.session_goal,
            "music_cover_available": bool(self.current_song_cover_hash),
            "total_focus_time_sec": self.total_focus_time_sec,
            "total_break_time_sec": self.total_break_time_sec,
            "whoop": whoop_snapshot,
            "fatigue": fatigue_snapshot,
            "capture_error": self.backend_status.get("capture_error", ""),
        }

    def refresh_whoop_now(self, force=False):
        metrics = self.whoop_client.fetch_metrics(force=force)
        with self.whoop_lock:
            self.current_whoop = copy.deepcopy(metrics)
        return metrics

    def _whoop_polling_loop(self):
        while True:
            try:
                if self.whoop_client.is_configured() and self.whoop_client.is_connected():
                    self.refresh_whoop_now(force=False)
                else:
                    with self.whoop_lock:
                        self.current_whoop = copy.deepcopy(self.whoop_client.cached_metrics())
            except Exception:
                pass
            time.sleep(max(30, int(self.whoop_poll_seconds / 4)))

    def _available_ort_providers(self):
        if not ONNX_AVAILABLE:
            return []
        try:
            return list(ort.get_available_providers())
        except Exception:
            return []

    def _preferred_provider_sets(self):
        available = self._available_ort_providers()
        self.backend_status["available"] = available

        candidates = [
            ("AMD VitisAI NPU", ["VitisAIExecutionProvider", "CPUExecutionProvider"]),
            ("DirectML", ["DmlExecutionProvider", "CPUExecutionProvider"]),
            ("Intel OpenVINO", ["OpenVINOExecutionProvider", "CPUExecutionProvider"]),
            ("Qualcomm QNN NPU", ["QNNExecutionProvider", "CPUExecutionProvider"]),
            ("CPU", ["CPUExecutionProvider"]),
        ]
        return [(label, providers) for label, providers in candidates if providers[0] in available]

    def _new_session_options(self, providers):
        options = ort.SessionOptions()
        if "DmlExecutionProvider" in providers:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options

    def _media_polling_loop(self):
        async def read_thumbnail(thumbnail):
            if not thumbnail:
                return b""
            try:
                from winrt.windows.storage.streams import DataReader
                stream = await thumbnail.open_read_async()
                size = int(getattr(stream, "size", 0) or 0)
                if size <= 0 or size > 5_000_000:
                    return b""
                reader = DataReader(stream)
                loaded = int(await reader.load_async(size))
                if loaded <= 0:
                    return b""
                data = bytearray(loaded)
                reader.read_bytes(data)
                return bytes(data)
            except Exception:
                return b""

        async def get_media_info():
            try:
                sessions = await GlobalSystemMediaTransportControlsSessionManager.request_async()
                current_session = sessions.get_current_session()
                if current_session:
                    info = await current_session.try_get_media_properties_async()
                    cover_bytes = await read_thumbnail(getattr(info, "thumbnail", None))
                    return info.title, info.artist, cover_bytes
            except Exception:
                pass
            return "", "", b""

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            title, artist, cover_bytes = loop.run_until_complete(get_media_info())
            title = title if title else ""
            artist = artist if artist else ""
            cover_hash = hashlib.sha1(cover_bytes).hexdigest()[:16] if cover_bytes else ""
            with self.media_lock:
                track_changed = title != self.current_song_title or artist != self.current_song_artist
                self.current_song_title = title
                self.current_song_artist = artist
                if cover_bytes:
                    self.current_song_cover_bytes = cover_bytes
                    self.current_song_cover_hash = cover_hash
                elif track_changed:
                    self.current_song_cover_bytes = b""
                    self.current_song_cover_hash = ""
            time.sleep(1)

    def _init_yolo_npu(self):
        try:
            from ultralytics import YOLO
            
            detect_model = "yolov8n.onnx"
            if not os.path.exists(detect_model):
                print("Downloading and exporting YOLOv8 Detect for NPU...")
                YOLO("yolov8n.pt").export(format="onnx")
                
            pose_model = "yolov8n-pose.onnx"
            if not os.path.exists(pose_model):
                print("Downloading and exporting YOLOv8 Pose for NPU...")
                YOLO("yolov8n-pose.pt").export(format="onnx")
                
            self.backend_status.update({
                "state": "Loading",
                "message": "Selecting the best available ONNX execution provider",
            })

            provider_sets = self._preferred_provider_sets()
            if not provider_sets:
                provider_sets = [("CPU", ["CPUExecutionProvider"])]

            last_error = None
            for label, providers in provider_sets:
                try:
                    print(f"Loading YOLOv8 ONNX models with {label}: {providers}")
                    options = self._new_session_options(providers)
                    self.ort_session_detect = ort.InferenceSession(detect_model, sess_options=options, providers=providers)
                    self.ort_session_pose = ort.InferenceSession(pose_model, sess_options=options, providers=providers)
                    self.detect_input_name = self.ort_session_detect.get_inputs()[0].name
                    self.pose_input_name = self.ort_session_pose.get_inputs()[0].name
                    benchmark_ms = self._benchmark_vision_backend()
                    self.backend_status.update({
                        "state": "Ready",
                        "selected": label,
                        "benchmark_ms": benchmark_ms,
                        "message": f"{label} ready for real-time vision",
                    })
                    print(f"YOLO models ready with {label} ({benchmark_ms:.1f} ms warmup run)")
                    return
                except Exception as e:
                    last_error = e
                    self.ort_session_detect = None
                    self.ort_session_pose = None
                    self.detect_input_name = None
                    self.pose_input_name = None
                    print(f"Failed to load YOLO with {label}: {e}")

            self.backend_status.update({
                "state": "Unavailable",
                "selected": "None",
                "message": f"Vision backend failed: {last_error}",
            })
        except Exception as e:
            self.backend_status.update({
                "state": "Unavailable",
                "selected": "None",
                "message": f"Vision backend failed: {e}",
            })
            print(f"Failed to load NPU YOLO models: {e}")

    def _benchmark_vision_backend(self):
        if not self.ort_session_pose or not self.ort_session_detect:
            return None
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        t0 = time.time()
        self.ort_session_pose.run(None, {self.pose_input_name: dummy})
        self.ort_session_detect.run(None, {self.detect_input_name: dummy})
        return (time.time() - t0) * 1000.0

    def start_session(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        self.is_paused = False
        self.auto_paused = False
        self.unfocused_counter = 0
        self.focused_counter = 0
        self.frames_captured = 0
        self.current_activity = "None"
        self.current_focused = False
        self.current_posture = "unknown"
        self.backend_status.pop("capture_error", None)
        self.start_time = time.time()
        self.total_focus_time_sec = 0.0
        self.total_break_time_sec = 0.0
        self.fatigue_pause_until = 0.0
        self.fatigue_autopause_cooldown_until = 0.0
        self.fatigue_detector.reset()
        self.chroma.start()
        with self.fatigue_lock:
            self.current_fatigue = self.fatigue_detector.snapshot()
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(os.getcwd(), "sessions", session_id)
        self.frames_dir = os.path.join(self.session_dir, "frames")
        self.assets_dir = os.path.join(self.session_dir, "assets")
        os.makedirs(self.session_dir, exist_ok=True)
        self.events_file = os.path.join(self.session_dir, "events.json")
        self.frames_video_path = os.path.join(self.session_dir, "frames.mp4")
        
        self.events = []
        self.last_events_flush = 0.0
        self._log_event("start")
        if self.whoop_client.is_configured() and self.whoop_client.is_connected():
            threading.Thread(target=self.refresh_whoop_now, kwargs={"force": True}, daemon=True).start()
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        return self.session_dir

    def pause_session(self, auto=False, reason="", recommended_break_seconds=0):
        if self.is_running and not self.is_paused:
            self.is_paused = True
            self.auto_paused = auto
            if auto and recommended_break_seconds:
                self.fatigue_pause_until = time.time() + recommended_break_seconds
            details = {
                "auto": auto,
                "reason": reason,
                "recommended_break_seconds": int(recommended_break_seconds or 0),
            }
            self._log_event("pause", details=details)

    def resume_session(self):
        if self.is_running and self.is_paused:
            self.is_paused = False
            self.auto_paused = False
            self.fatigue_pause_until = 0.0
            self.unfocused_counter = 0
            self.focused_counter = 0
            self._log_event("resume")

    def end_session(self):
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(8.0, self.capture_interval + 5.0))
            if self.thread.is_alive():
                self.backend_status["capture_error"] = "Camera shutdown timed out; the session may need recovery"
        self._log_event("end")
        self.chroma.close()

    def _cover_asset_extension(self, cover_bytes):
        if cover_bytes.startswith(b"\xff\xd8"):
            return ".jpg"
        if cover_bytes.startswith(b"\x89PNG"):
            return ".png"
        if cover_bytes[:4] == b"RIFF" and cover_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".img"

    def _ensure_music_cover_asset(self):
        if not self.session_dir:
            return ""

        with self.media_lock:
            cover_bytes = bytes(self.current_song_cover_bytes)
            cover_hash = self.current_song_cover_hash

        if not cover_bytes or not cover_hash:
            return ""

        try:
            os.makedirs(self.assets_dir, exist_ok=True)
            filename = f"cover_{cover_hash}{self._cover_asset_extension(cover_bytes)}"
            cover_path = os.path.join(self.assets_dir, filename)
            if not os.path.exists(cover_path):
                with open(cover_path, "wb") as f:
                    f.write(cover_bytes)
            return f"assets/{filename}"
        except OSError:
            return ""

    def _log_event(self, event_type, frame_index=None, filename=None, focused=True, music="", music_cover="", whoop=None, fatigue=None, activity="", posture="", details=None):
        if details is None and isinstance(frame_index, dict) and event_type != "frame":
            details = frame_index
            frame_index = None
        event = {
            "timestamp": time.time(),
            "type": event_type
        }
        if event_type == "frame":
            if frame_index is not None:
                event["frame_index"] = frame_index
            if filename:
                event["filename"] = filename
            event["focused"] = focused
            event["music"] = music
            if music_cover:
                event["music_cover"] = music_cover
            if whoop and whoop.get("available"):
                event["whoop"] = whoop
            if fatigue and fatigue.get("available"):
                event["fatigue"] = fatigue
            event["activity"] = activity
            event["posture"] = posture
            event["inference_ms"] = round(self.last_inference_ms, 1)
        elif event_type == "start":
            event["goal"] = self.session_goal
            event["capture_interval"] = self.capture_interval
            event["privacy_mode"] = self.privacy_mode
            event["backend"] = self.backend_status.get("selected", "Pending")
        if details:
            event.update({key: value for key, value in details.items() if value not in ("", None, 0, False)})
        with self.events_lock:
            self.events.append(event)
        force_flush = event_type != "frame" or time.time() - self.last_events_flush >= self.events_flush_interval
        if force_flush:
            self._write_events_file()

    def _write_events_file(self):
        if not self.events_file:
            return
        with self.events_write_lock:
            with self.events_lock:
                events_snapshot = copy.deepcopy(self.events)
            payload = {
                "start_time": self.start_time,
                "goal": self.session_goal,
                "settings": {
                    "capture_interval": self.capture_interval,
                    "auto_pause_seconds": self.auto_pause_seconds,
                    "privacy_mode": self.privacy_mode,
                    "capture_resolution": f"{self.capture_width}x{self.capture_height}",
                    "capture_codec": self.capture_codec,
                },
                "backend": dict(self.backend_status),
                "events": events_snapshot,
            }
            temp_path = f"{self.events_file}.tmp"
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, separators=(",", ":"))
                os.replace(temp_path, self.events_file)
                self.last_events_flush = time.time()
            except OSError as e:
                self.backend_status["capture_error"] = f"Session checkpoint failed: {e}"
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass

    def _apply_recording_privacy(self, frame):
        mode = self.privacy_mode
        if mode == "Off":
            return frame

        img = frame.copy()
        h, w = img.shape[:2]
        if mode == "Face Blur":
            x1, x2 = int(w * 0.34), int(w * 0.66)
            y1, y2 = int(h * 0.08), int(h * 0.42)
            roi = img[y1:y2, x1:x2]
            if roi.size:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (81, 81), 0)
            return img

        if mode == "Background Blur":
            blurred = cv2.GaussianBlur(img, (71, 71), 0)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w // 2, int(h * 0.52)), (int(w * 0.28), int(h * 0.42)), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (61, 61), 0)[:, :, None] / 255.0
            return (img * mask + blurred * (1 - mask)).astype(np.uint8)

        if mode == "Low Detail":
            tiny = cv2.resize(img, (max(8, w // 10), max(8, h // 10)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)

        return frame

    def _should_auto_pause_for_fatigue(self, fatigue):
        if self.is_paused or not fatigue or not fatigue.get("break_recommended"):
            return False
        if time.time() < self.fatigue_autopause_cooldown_until:
            return False
        score = int(fatigue.get("score", 0) or 0)
        high_seconds = float(fatigue.get("high_fatigue_seconds", 0) or 0)
        eye_seconds = float(fatigue.get("eye_fade_seconds", 0) or 0)
        slouch_seconds = float(fatigue.get("slouch_seconds", 0) or 0)
        return score >= 78 or high_seconds >= 35 or eye_seconds >= 18 or slouch_seconds >= 150

    def _capture_loop(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 2560x1440 = HP 5MP cam's native sensor mode. 1080p is center-cropped on this device.
        # MJPG must come before resolution — YUY2 won't deliver high-res and forces driver crop.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.cap.isOpened() or actual_w <= 0 or actual_h <= 0:
            self.backend_status["capture_error"] = "Camera could not be opened"
            self.is_running = False
            self.stop_event.set()
            return

        # Single-file session storage as H.264. avc1 emits noisy stderr about libopenh264
        # on this install but still encodes successfully (verified by test_codec.py — ~half
        # the size of mp4v). Falls back to mp4v if avc1 truly can't initialize.
        # Current default is mp4v to avoid OpenH264 DLL/version errors in normal runs.
        preferred_codec = self.capture_codec or "mp4v"
        source_fps = max(0.5, min(4.0, 1.0 / max(0.25, self.capture_interval)))
        self.video_writer = cv2.VideoWriter(self.frames_video_path,
                                             cv2.VideoWriter_fourcc(*preferred_codec), source_fps,
                                             (actual_w, actual_h))
        if not self.video_writer.isOpened() and preferred_codec != "mp4v":
            print(f"[Capture] {preferred_codec} writer failed to open, falling back to mp4v")
            self.video_writer = cv2.VideoWriter(self.frames_video_path,
                                                 cv2.VideoWriter_fourcc(*'mp4v'), source_fps,
                                                 (actual_w, actual_h))
        if not self.video_writer.isOpened():
            print("[Capture] mp4v writer failed to open, falling back to JPEG frames")
            os.makedirs(self.frames_dir, exist_ok=True)
            self.video_writer = None

        frame_count = 0

        while self.is_running and not self.stop_event.is_set():
            start_loop = time.time()

            ret, frame = self.cap.read()
            if not ret:
                if self.stop_event.wait(0.5):
                    break
                continue

            vision_ready = bool(
                self.ort_session_pose
                and self.ort_session_detect
                and self.pose_input_name
                and self.detect_input_name
            )
            focused = True if not vision_ready else False
            activity = "None"
            posture = "unknown"
            pose_points = {}
            h, w, _ = frame.shape
            
            inference_start = time.time()
            if vision_ready:
                try:
                    # Preprocess frame for YOLOv8 (640x640)
                    img = cv2.resize(frame, (640, 640))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    img = np.ascontiguousarray(np.expand_dims(img, axis=0))
                    
                    # 1. Head Pose Tracking via YOLOv8-Pose
                    outs_pose = self.ort_session_pose.run(None, {self.pose_input_name: img})[0]
                    preds_pose = outs_pose[0] # Shape: [56, 8400]
                    
                    person_confidences = preds_pose[4, :]
                    best_idx = np.argmax(person_confidences)
                    
                    if person_confidences[best_idx] > 0.5:
                        # COCO keypoints: 0=nose, 1=l_eye, 2=r_eye, 5=l_shoulder, 6=r_shoulder.
                        # Layout in preds_pose: index [5 + kpt*3 + (0/1/2)] = x / y / conf.
                        nose_x = preds_pose[5 + 0*3, best_idx]
                        nose_y = preds_pose[5 + 0*3 + 1, best_idx]
                        nose_conf = preds_pose[5 + 0*3 + 2, best_idx]
                        leye_x = preds_pose[5 + 1*3, best_idx]
                        leye_conf = preds_pose[5 + 1*3 + 2, best_idx]
                        reye_x = preds_pose[5 + 2*3, best_idx]
                        reye_conf = preds_pose[5 + 2*3 + 2, best_idx]
                        ls_x = preds_pose[5 + 5*3, best_idx]
                        ls_y = preds_pose[5 + 5*3 + 1, best_idx]
                        ls_conf = preds_pose[5 + 5*3 + 2, best_idx]
                        rs_x = preds_pose[5 + 6*3, best_idx]
                        rs_y = preds_pose[5 + 6*3 + 1, best_idx]
                        rs_conf = preds_pose[5 + 6*3 + 2, best_idx]

                        lear_conf = preds_pose[5 + 3*3 + 2, best_idx]
                        lear_x = preds_pose[5 + 3*3, best_idx]
                        lear_y = preds_pose[5 + 3*3 + 1, best_idx]
                        rear_conf = preds_pose[5 + 4*3 + 2, best_idx]
                        rear_x = preds_pose[5 + 4*3, best_idx]
                        rear_y = preds_pose[5 + 4*3 + 1, best_idx]
                        pose_points = {
                            "nose": (float(nose_x), float(nose_y), float(nose_conf)),
                            "left_eye": (float(leye_x), float(preds_pose[5 + 1*3 + 1, best_idx]), float(leye_conf)),
                            "right_eye": (float(reye_x), float(preds_pose[5 + 2*3 + 1, best_idx]), float(reye_conf)),
                            "left_ear": (float(lear_x), float(lear_y), float(lear_conf)),
                            "right_ear": (float(rear_x), float(rear_y), float(rear_conf)),
                            "left_shoulder": (float(ls_x), float(ls_y), float(ls_conf)),
                            "right_shoulder": (float(rs_x), float(rs_y), float(rs_conf)),
                        }
                        
                        face_kpts_x = []
                        if nose_conf > 0.5: face_kpts_x.append(nose_x)
                        if leye_conf > 0.5: face_kpts_x.append(leye_x)
                        if reye_conf > 0.5: face_kpts_x.append(reye_x)
                        if lear_conf > 0.5: face_kpts_x.append(lear_x)
                        if rear_conf > 0.5: face_kpts_x.append(rear_x)
                        if ls_conf > 0.5: face_kpts_x.append(ls_x)
                        if rs_conf > 0.5: face_kpts_x.append(rs_x)

                        if face_kpts_x:
                            avg_face_x = sum(face_kpts_x) / len(face_kpts_x)
                            # YOLOv8 input is 640x640. Check if face is within the center 80%.
                            if 640 * 0.1 < avg_face_x < 640 * 0.9:
                                focused = True

                        # Posture: needs nose + both shoulders. Measure head distance above
                        # the shoulder line, normalized by shoulder width — invariant to how
                        # close the user is to the camera.
                        if nose_conf > 0.5 and ls_conf > 0.5 and rs_conf > 0.5:
                            shoulder_dx = abs(ls_x - rs_x)
                            shoulder_dy = abs(ls_y - rs_y)
                            head_above = ((ls_y + rs_y) / 2.0) - nose_y
                            if shoulder_dx > 1:
                                if head_above / shoulder_dx < 0.35:
                                    posture = "slouching"  # head sinking toward shoulders
                                elif shoulder_dy / shoulder_dx > 0.25:
                                    posture = "leaning"    # shoulder line visibly tilted
                                else:
                                    posture = "good"

                    # 2. Distraction and Activity Detection via YOLOv8 Detect
                    if focused:
                        outs_detect = self.ort_session_detect.run(None, {self.detect_input_name: img})[0]
                        preds_detect = outs_detect[0] # Shape: [84, 8400]
                        
                        # Check class 67 (cell phone) in COCO
                        phone_probs = preds_detect[4+67, :]
                        if np.max(phone_probs) > 0.60:
                            focused = False # Cell phone detected, immediately penalize focus!
                            activity = "Distracted (Phone)"
                        else:
                            # Check class 63 (laptop) and 73 (book)
                            laptop_probs = preds_detect[4+63, :]
                            book_probs = preds_detect[4+73, :]
                            
                            max_laptop = np.max(laptop_probs)
                            max_book = np.max(book_probs)
                            
                            if max_laptop > 0.5 and max_laptop > max_book:
                                activity = "Laptop"
                            elif max_book > 0.5:
                                activity = "Book"
                            else:
                                activity = "Desk / iPad"

                    if not focused or activity == "Distracted (Phone)":
                        self.distracted_time_seconds += self.capture_interval
                        self.chroma.set_state("distracted")
                        self.was_distracted = True
                        
                        # Active Media Control
                        if self.distracted_time_seconds > 10.0 and not self.media_paused_by_us:
                            try:
                                ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
                                ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)
                                self.media_paused_by_us = True
                            except: pass
                    else:
                        self.distracted_time_seconds = 0.0
                        self.was_distracted = False
                        if self.media_paused_by_us:
                            try:
                                ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
                                ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)
                                self.media_paused_by_us = False
                            except: pass
                    
                    if not self.was_distracted:
                        if posture in ["slouching", "leaning"]:
                            self.consecutive_slouch_seconds += self.capture_interval
                            self.chroma.set_state("slouching")
                            self.was_slouching = True
                        else:
                            self.consecutive_slouch_seconds = 0.0
                            self.chroma.set_state("normal")
                            self.was_slouching = False
                        
                except Exception as e:
                    pass
            self.last_inference_ms = (time.time() - inference_start) * 1000.0

            self.current_posture = posture
            self.current_activity = activity
            self.current_focused = focused

            with self.media_lock:
                song_title = self.current_song_title
                song_artist = self.current_song_artist
            with self.whoop_lock:
                whoop_snapshot = copy.deepcopy(self.current_whoop)
            fatigue_snapshot = self.fatigue_detector.update(
                pose_points,
                focused=focused,
                posture=posture,
                activity=activity,
                dt=self.capture_interval,
                study_elapsed_sec=time.time() - self.start_time,
                whoop=whoop_snapshot,
            )
            with self.fatigue_lock:
                self.current_fatigue = copy.deepcopy(fatigue_snapshot)

            music_str = ""
            if song_title:
                music_str = f"{song_title} - {song_artist}" if song_artist else song_title
            music_cover = self._ensure_music_cover_asset() if music_str else ""

            if self.stop_event.is_set():
                break

            if not self.is_paused:
                filename = None
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(self._apply_recording_privacy(frame))
                else:
                    filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(self.frames_dir, filename)
                    cv2.imwrite(frame_path, self._apply_recording_privacy(frame), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                self._log_event("frame", frame_index=frame_count if filename is None else None, filename=filename,
                                focused=focused, music=music_str, music_cover=music_cover,
                                whoop=whoop_snapshot, fatigue=fatigue_snapshot,
                                activity=activity, posture=posture)
                frame_count += 1
                self.frames_captured = frame_count

                if self._should_auto_pause_for_fatigue(fatigue_snapshot):
                    self.fatigue_autopause_cooldown_until = time.time() + 900
                    self.chroma.set_state("slouching")
                    self.pause_session(
                        auto=True,
                        reason=f"Fatigue: {fatigue_snapshot.get('expression', 'tired')}",
                        recommended_break_seconds=fatigue_snapshot.get("recommended_break_seconds", 0),
                    )
                
                # Auto Pause Logic
                if not focused:
                    self.unfocused_counter += 1
                else:
                    self.unfocused_counter = 0
                    
                frames_until_pause = max(1, int(self.auto_pause_seconds / max(0.25, self.capture_interval)))
                if self.unfocused_counter >= frames_until_pause:
                    self.pause_session(auto=True)
                    
            else:
                # We are paused.
                # If we were AUTO-paused, we look for them to return
                if self.auto_paused:
                    if self.fatigue_pause_until and time.time() < self.fatigue_pause_until:
                        self.focused_counter = 0
                    elif focused:
                        self.focused_counter += 1
                        if self.focused_counter >= 3: # 3 consecutive seconds of focus
                            self.resume_session()
                    else:
                        self.focused_counter = 0
                else:
                    # Manually paused. Do nothing.
                    pass

            elapsed = time.time() - start_loop
            self.last_loop_ms = elapsed * 1000.0
            sleep_time = max(0, self.capture_interval - elapsed)
            if self.stop_event.wait(sleep_time):
                break
            
            loop_duration = time.time() - start_loop
            if self.is_paused:
                self.total_break_time_sec += loop_duration
            elif focused:
                self.total_focus_time_sec += loop_duration
            
        if self.cap:
            self.cap.release()
        if self.video_writer:
            # release() finalizes the mp4 (writes the moov atom). If the app crashes
            # before this point, the file may be unreadable — known v1 limitation.
            self.video_writer.release()
            self.video_writer = None
