import cv2
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Renderer:
    DEFAULT_OPTIONS = {
        "target_video_duration": 30,
        "fps": 30,
        "hud_style": "Glass",
        "aspect_ratio": "Source 16:9",
        "privacy_mode": "Off",
        "goal_label": "",
        "generate_social_export": False,
    }

    def __init__(self, session_dir, options=None, progress_callback=None):
        self.session_dir = session_dir
        self.events_file = os.path.join(session_dir, "events.json")
        self.frames_video = os.path.join(session_dir, "frames.mp4")
        self.frames_dir = os.path.join(session_dir, "frames")

        self.options = dict(self.DEFAULT_OPTIONS)
        if options:
            self.options.update(options)

        self.target_video_duration = max(5, int(self.options.get("target_video_duration", 30)))
        self.fps = max(10, int(self.options.get("fps", 30)))
        self.progress_callback = progress_callback
        self.cover_cache = {}
        self.font_cache = {}
        self.vignette_cache = {}

        self.out_dir = os.path.join(os.getcwd(), "timelapses")
        os.makedirs(self.out_dir, exist_ok=True)

    def render(self):
        self._progress("Loading session", 0.02)
        if not os.path.exists(self.events_file):
            return None

        with open(self.events_file, "r") as f:
            data = json.load(f)

        events = data.get("events", [])
        if not events:
            return None

        timeline = self._build_timeline(events)
        frames_to_process = timeline["frames"]
        pauses = timeline["pauses"]
        study_duration = timeline["study_duration"]
        focused_duration = timeline.get("focused_duration", 0.0)
        focus_score = timeline["focus_score"]
        if not frames_to_process:
            return None

        self._progress("Planning cinematic pacing", 0.12)
        content_budget = self._configure_timing_budget(len(pauses))
        selected_frames, pause_map, event_map = self._select_frames(frames_to_process, pauses, content_budget)
        if not selected_frames:
            return None

        source_mode = self._source_mode(selected_frames)
        first_frame = self._load_first_frame(source_mode, selected_frames)
        if first_frame is None:
            return None

        prepared_first = self._fit_aspect(first_frame)
        height, width = prepared_first.shape[:2]

        out_filename = f"timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(self.out_dir, out_filename)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not out.isOpened():
            return None

        last_processed_img = None
        pending_resume_card = None

        def write_selected_frame(raw_frame, sel_idx, frame_info):
            nonlocal last_processed_img, pending_resume_card
            img = self._prepare_frame(raw_frame, frame_info, event_map.get(sel_idx))

            if pending_resume_card is not None:
                self._write_card_to_frame_transition(out, pending_resume_card, img)
                pending_resume_card = None

            out.write(img)
            last_processed_img = img.copy()

            if sel_idx in pause_map:
                for pause_dur in pause_map[sel_idx]:
                    pending_resume_card = self._write_pause_screen(out, last_processed_img, pause_dur)

        try:
            self._progress("Rendering selected frames", 0.18)
            if source_mode == "video":
                self._render_from_video(selected_frames, write_selected_frame)
            else:
                self._render_from_jpegs(selected_frames, write_selected_frame)

            self._progress("Writing summary card", 0.94)
            if last_processed_img is not None:
                self._write_summary_screen(out, last_processed_img, study_duration, focused_duration, focus_score)
        finally:
            out.release()

        if self.options.get("generate_social_export", False):
            self._progress("Generating social export", 0.96)
            self._create_social_export(self.session_dir, timeline)
            
        self._progress("Render complete", 1.0)
        return out_path

    def _progress(self, stage, fraction):
        if self.progress_callback:
            self.progress_callback(stage, max(0.0, min(1.0, fraction)))

    def _build_timeline(self, events):
        frames = []
        pauses = []
        current_pause_start = None
        study_duration_so_far = 0.0
        focused_duration_so_far = 0.0
        last_frame_timestamp = None
        focus_streak_secs = 0
        total_focus_count = 0
        total_frame_count = 0

        for event in events:
            event_type = event.get("type")
            if event_type == "frame":
                ts = event["timestamp"]
                if last_frame_timestamp:
                    delta = max(0.0, ts - last_frame_timestamp)
                    study_duration_so_far += delta
                else:
                    delta = 0.0
                last_frame_timestamp = ts

                focused = bool(event.get("focused", True))
                if focused:
                    focused_duration_so_far += delta
                focus_streak_secs = focus_streak_secs + max(1, int(delta or 1)) if focused else 0
                total_focus_count += 1 if focused else 0
                total_frame_count += 1

                filename = event.get("filename", event.get("file", ""))
                frame_index = event.get("frame_index")
                if frame_index is None and filename.startswith("frame_"):
                    try:
                        frame_index = int(os.path.splitext(filename)[0].split("_")[-1])
                    except ValueError:
                        frame_index = None

                frames.append({
                    "source_idx": len(frames),
                    "frame_index": frame_index,
                    "file": filename,
                    "timestamp": ts,
                    "study_duration": study_duration_so_far,
                    "music": event.get("music", ""),
                    "music_cover": event.get("music_cover", ""),
                    "whoop": event.get("whoop", {}),
                    "fatigue": event.get("fatigue", {}),
                    "focused": focused,
                    "activity": event.get("activity", "None"),
                    "posture": event.get("posture", "unknown"),
                    "focus_streak_secs": focus_streak_secs,
                })
            elif event_type == "pause":
                current_pause_start = event.get("timestamp", 0)
                last_frame_timestamp = None
            elif event_type == "resume":
                if current_pause_start:
                    pauses.append({
                        "insert_after_frame_idx": len(frames) - 1,
                        "duration_secs": event.get("timestamp", 0) - current_pause_start,
                    })
                current_pause_start = None
                last_frame_timestamp = event.get("timestamp")

        focus_score = int((total_focus_count / max(1, total_frame_count)) * 100)
        return {
            "frames": frames,
            "pauses": pauses,
            "study_duration": study_duration_so_far,
            "focused_duration": focused_duration_so_far,
            "focus_score": focus_score,
        }

    def _configure_timing_budget(self, pause_count):
        target_total = max(1, int(round(self.target_video_duration * self.fps)))

        default_pause_morph = max(1, int(round(self.fps * 0.70)))
        default_pause_hold = max(1, int(round(self.fps * 1.35)))
        default_resume_morph = max(1, int(round(self.fps * 0.55)))
        default_summary_morph = max(1, int(round(self.fps * 0.90)))
        default_summary_hold = max(1, int(round(self.fps * 3.00)))

        default_reserved = default_summary_morph + default_summary_hold
        default_reserved += pause_count * (default_pause_morph + default_pause_hold + default_resume_morph)

        min_content = min(target_total, max(1, int(round(self.fps * 3.0))))
        if default_reserved > target_total - min_content and default_reserved > 0:
            scale = max(0.18, (target_total - min_content) / default_reserved)
        else:
            scale = 1.0

        self.pause_morph_frames = max(1, int(round(default_pause_morph * scale)))
        self.pause_hold_frames = max(1, int(round(default_pause_hold * scale)))
        self.resume_morph_frames = max(1, int(round(default_resume_morph * scale)))
        self.summary_morph_frames = max(1, int(round(default_summary_morph * scale)))
        self.summary_hold_frames = max(1, int(round(default_summary_hold * scale)))

        reserved = self.summary_morph_frames + self.summary_hold_frames
        reserved += pause_count * (self.pause_morph_frames + self.pause_hold_frames + self.resume_morph_frames)

        return max(1, target_total - reserved)

    def _select_frames(self, frames_to_process, pauses, content_budget):
        total_frames = len(frames_to_process)
        key_event_indices = set()
        event_labels = {}
        last_music = ""
        last_posture = "unknown"

        for i, frame in enumerate(frames_to_process):
            music = frame["music"]
            if music and music != last_music:
                key_event_indices.add(i)
            last_music = music

            activity = frame.get("activity", "")
            posture = frame.get("posture", "unknown")
            if activity == "Distracted (Phone)":
                key_event_indices.add(i)
                event_labels[i] = "Phone detected"

        for pause in pauses:
            key_event_indices.add(max(0, pause["insert_after_frame_idx"]))

        budget = max(1, int(round(content_budget)))
        if budget >= total_frames:
            selected_source_indices = np.linspace(0, total_frames - 1, budget, dtype=int).tolist()
            selected_frames = [frames_to_process[idx] for idx in selected_source_indices]
            pause_map = {}
            event_map = {}

            for pause in pauses:
                pause_idx = max(0, pause["insert_after_frame_idx"])
                selected_idx = self._nearest_selected_at_or_before(selected_source_indices, pause_idx)
                pause_map.setdefault(selected_idx, []).append(pause["duration_secs"])

            for label_idx, label in sorted(event_labels.items()):
                selected_idx = self._nearest_selected(selected_source_indices, label_idx)
                event_map.setdefault(selected_idx, label)

            return selected_frames, pause_map, event_map

        key_budget = max(0, min(len(key_event_indices), budget // 3))
        selected_indices = set()

        if total_frames > 0:
            selected_indices.add(0)
            selected_indices.add(total_frames - 1)

        sorted_keys = sorted(key_event_indices)
        if sorted_keys and key_budget > 0:
            if len(sorted_keys) <= key_budget:
                selected_indices.update(sorted_keys)
            else:
                chosen_positions = np.linspace(0, len(sorted_keys) - 1, key_budget, dtype=int)
                selected_indices.update(sorted_keys[int(pos)] for pos in chosen_positions)

        uniform_needed = max(0, budget - len(selected_indices))
        if uniform_needed > 0:
            selected_indices.update(int(round(i)) for i in np.linspace(0, total_frames - 1, uniform_needed))

        while len(selected_indices) < budget:
            ordered = sorted(selected_indices)
            largest_gap = None
            largest_size = -1
            for left, right in zip(ordered, ordered[1:]):
                gap = right - left
                if gap > largest_size:
                    largest_size = gap
                    largest_gap = (left, right)
            if not largest_gap or largest_size <= 1:
                for idx in range(total_frames):
                    if idx not in selected_indices:
                        selected_indices.add(idx)
                        break
                else:
                    break
            else:
                selected_indices.add((largest_gap[0] + largest_gap[1]) // 2)

        if len(selected_indices) > budget:
            ordered = sorted(selected_indices)
            keep_positions = np.linspace(0, len(ordered) - 1, budget, dtype=int)
            selected_indices = {ordered[int(pos)] for pos in keep_positions}

        selected_source_indices = sorted(selected_indices)
        selected_frames = [frames_to_process[idx] for idx in selected_source_indices]
        pause_map = {}
        event_map = {}

        for pause in pauses:
            pause_idx = max(0, pause["insert_after_frame_idx"])
            selected_idx = self._nearest_selected_at_or_before(selected_source_indices, pause_idx)
            pause_map.setdefault(selected_idx, []).append(pause["duration_secs"])

        for label_idx, label in sorted(event_labels.items()):
            selected_idx = self._nearest_selected(selected_source_indices, label_idx)
            event_map.setdefault(selected_idx, label)

        return selected_frames, pause_map, event_map

    def _nearest_selected_at_or_before(self, selected_indices, source_idx):
        if not selected_indices:
            return 0
        pos = int(np.searchsorted(selected_indices, source_idx, side="right")) - 1
        return max(0, min(pos, len(selected_indices) - 1))

    def _nearest_selected(self, selected_indices, source_idx):
        if not selected_indices:
            return 0
        pos = int(np.searchsorted(selected_indices, source_idx, side="left"))
        if pos <= 0:
            return 0
        if pos >= len(selected_indices):
            return len(selected_indices) - 1
        before = selected_indices[pos - 1]
        after = selected_indices[pos]
        return pos - 1 if abs(source_idx - before) <= abs(after - source_idx) else pos

    def _source_mode(self, selected_frames):
        has_video_indices = os.path.exists(self.frames_video) and all(
            frame.get("frame_index") is not None for frame in selected_frames
        )
        if has_video_indices:
            cap = cv2.VideoCapture(self.frames_video)
            ok, _ = cap.read()
            cap.release()
            if ok:
                return "video"
        return "jpeg"

    def _load_first_frame(self, source_mode, selected_frames):
        if source_mode == "video":
            cap = cv2.VideoCapture(self.frames_video)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        for info in selected_frames:
            frame = self._read_jpeg(info)
            if frame is not None:
                return frame
        return None

    def _render_from_video(self, selected_frames, write_selected_frame):
        selected_iter = iter(enumerate(selected_frames))
        next_sel = next(selected_iter, None)
        source_cap = cv2.VideoCapture(self.frames_video)
        video_frame_idx = 0
        rendered = 0
        total = len(selected_frames)

        while next_sel is not None:
            ret, frame = source_cap.read()
            if not ret:
                break
            sel_idx, frame_info = next_sel
            while next_sel is not None and video_frame_idx == frame_info["frame_index"]:
                write_selected_frame(frame, sel_idx, frame_info)
                rendered += 1
                if rendered % 12 == 0 or rendered == total:
                    self._progress("Rendering selected frames", 0.18 + 0.74 * (rendered / max(1, total)))
                next_sel = next(selected_iter, None)
                if next_sel is not None:
                    sel_idx, frame_info = next_sel
            video_frame_idx += 1

        source_cap.release()

    def _render_from_jpegs(self, selected_frames, write_selected_frame):
        total = len(selected_frames)
        for sel_idx, frame_info in enumerate(selected_frames):
            frame = self._read_jpeg(frame_info)
            if frame is None:
                continue
            write_selected_frame(frame, sel_idx, frame_info)
            if sel_idx % 12 == 0 or sel_idx == total - 1:
                self._progress("Rendering selected frames", 0.18 + 0.74 * ((sel_idx + 1) / max(1, total)))

    def _read_jpeg(self, frame_info):
        filename = frame_info.get("file", "")
        if not filename:
            return None
        return cv2.imread(os.path.join(self.frames_dir, filename))

    def _prepare_frame(self, img, frame_info, event_label=None):
        img = self._apply_privacy(img)
        img = self._fit_aspect(img)
        hud_style = self.options.get("hud_style", "Glass")
        if hud_style != "Off":
            self._draw_hud(img, frame_info, hud_style)
            self._draw_fatigue_card(img, frame_info, hud_style)
            self._draw_whoop_card(img, frame_info)
            self._draw_music_card(img, frame_info)
        if event_label:
            self._draw_event_badge(img, event_label)
        return img

    def _apply_privacy(self, img):
        mode = self.options.get("privacy_mode", "Off")
        if mode == "Off":
            return img

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

        return img

    def _fit_aspect(self, img):
        aspect = self.options.get("aspect_ratio", "Source 16:9")
        if aspect == "Vertical 9:16":
            return self._crop_resize(img, 9 / 16, 1080, 1920)
        if aspect == "Square 1:1":
            return self._crop_resize(img, 1.0, 1080, 1080)
        return img

    def _crop_resize(self, img, target_ratio, out_w, out_h):
        h, w = img.shape[:2]
        current_ratio = w / h
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            x1 = max(0, (w - new_w) // 2)
            cropped = img[:, x1:x1 + new_w]
        else:
            new_h = int(w / target_ratio)
            y1 = max(0, (h - new_h) // 2)
            cropped = img[y1:y1 + new_h, :]
        return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def _draw_hud(self, img, frame_info, hud_style):
        if hud_style == "Minimal":
            self._draw_minimal_hud(img, frame_info)
        elif hud_style == "Study Log":
            self._draw_study_log_hud(img, frame_info)
        elif hud_style == "Cinematic":
            self._draw_cinematic_hud(img, frame_info)
        else:
            self._draw_glass_hud(img, frame_info)

    def _draw_glass_hud(self, img, frame_info):
        actual_time_str = datetime.fromtimestamp(frame_info["timestamp"]).strftime("%I:%M %p")
        chrono_str = self._format_chrono(frame_info['study_duration'])

        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))

        panel_w = s(300)
        panel_h = s(108)
        x = img_w - panel_w - s(30)
        y = s(30)
        if y + panel_h > img_h or x < 0:
            return

        self._glass_panel(img, x, y, panel_w, panel_h, s(20))
        draw, fonts = self._pil_draw(img, scale)
        text_x = x + s(20)
        text_y = y + s(14)
        draw.text((text_x, text_y), "LOCAL TIME", font=fonts["small_bold"], fill=(151, 220, 255))
        draw.text((text_x, text_y + s(24)), actual_time_str, font=fonts["large"], fill=(255, 255, 255))
        draw.rounded_rectangle((text_x, text_y + s(66), x + panel_w - s(18), text_y + s(88)), radius=s(11), fill=(18, 24, 29))
        dot_color = (34, 197, 94) if frame_info.get("focused", True) else (245, 158, 11)
        draw.ellipse((text_x + s(9), text_y + s(73), text_x + s(17), text_y + s(81)), fill=dot_color)
        draw.text((text_x + s(24), text_y + s(68)), f"Focus {chrono_str}", font=fonts["small_bold"], fill=(220, 255, 229))
        self._commit_pil_draw(img, draw)

    def _draw_minimal_hud(self, img, frame_info):
        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))
        text = f"{self._format_chrono(frame_info['study_duration'])}  |  {frame_info.get('activity', 'None')}"
        if not frame_info.get("focused", True):
            text += "  |  distracted"
        text = self._ellipsize(text, max(18, int(58 * scale)))
        panel_w = min(img_w - s(52), s(560))
        x = s(26)
        y = s(26)
        self._solid_panel(img, x, y, panel_w, s(56), alpha=0.62)
        draw, fonts = self._pil_draw(img, scale)
        draw.text((x + s(22), y + s(16)), text, font=fonts["medium_bold"], fill=(245, 248, 255))
        self._commit_pil_draw(img, draw)

    def _draw_study_log_hud(self, img, frame_info):
        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))
        panel_w = s(390)
        panel_h = s(255)
        x = s(28)
        y = s(28)
        self._glass_panel(img, x, y, panel_w, panel_h, s(18))
        draw, fonts = self._pil_draw(img, scale)
        lines = [
            ("STUDY LOG", fonts["small_bold"], (155, 220, 255)),
            (datetime.fromtimestamp(frame_info["timestamp"]).strftime("%I:%M %p"), fonts["large"], (255, 255, 255)),
            (f"Duration: {self._format_chrono(frame_info['study_duration'])}", fonts["medium_bold"], (178, 255, 198)),
            (f"Streak: {self._format_chrono(frame_info.get('focus_streak_secs', 0))}", fonts["medium"], (220, 224, 232)),
            (f"Activity: {frame_info.get('activity', 'None')}", fonts["medium"], (220, 224, 232)),
            (f"Posture: {frame_info.get('posture', 'unknown')}", fonts["medium"], (220, 224, 232)),
        ]
        cy = y + s(20)
        for text, font, color in lines:
            draw.text((x + s(22), cy), text, font=font, fill=color)
            cy += s(34)
        self._commit_pil_draw(img, draw)

    def _draw_cinematic_hud(self, img, frame_info):
        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))
        bar_h = s(92)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, bar_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, img_h - bar_h), (img_w, img_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        draw, fonts = self._pil_draw(img, scale)
        title = self.options.get("goal_label") or "Study Session"
        meta = f"{self._format_chrono(frame_info['study_duration'])}  |  Focus streak {self._format_chrono(frame_info.get('focus_streak_secs', 0))}"
        draw.text((s(42), s(30)), meta, font=fonts["medium_bold"], fill=(178, 255, 198))
        if frame_info.get("music"):
            title = self._ellipsize(title, max(14, int(32 * scale)))
        draw.text((s(42), img_h - s(70)), title, font=fonts["large"], fill=(255, 255, 255))
        self._commit_pil_draw(img, draw)

    def _draw_whoop_card(self, img, frame_info):
        metrics = frame_info.get("whoop") or {}
        if not isinstance(metrics, dict) or not metrics.get("available"):
            return

        recovery = metrics.get("recovery") or {}
        sleep = metrics.get("sleep") or {}
        strain = metrics.get("strain") or {}
        if not any((recovery, sleep, strain)):
            return

        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))
        card_w = min(img_w - s(60), max(s(360), s(540)))
        card_h = s(212)
        if card_w <= s(220) or img_h <= card_h + s(40):
            return

        bottom_offset = s(116) if self.options.get("hud_style") == "Cinematic" else s(30)
        x = s(30)
        y = max(s(24), img_h - card_h - bottom_offset)
        if x + card_w > img_w or y + card_h > img_h:
            return

        self._glass_panel(img, x, y, card_w, card_h, s(22))
        draw, fonts = self._pil_draw(img, scale)
        header_y = y + s(14)
        draw.text((x + s(18), header_y), "WHOOP", font=fonts["small_bold"], fill=(236, 240, 245))
        draw.text((x + s(84), header_y), "BODY METRICS", font=fonts["small"], fill=(144, 153, 168))
        if metrics.get("status"):
            status = self._ellipsize(str(metrics.get("status")), 18)
            status_bbox = draw.textbbox((0, 0), status, font=fonts["small"])
            draw.text((x + card_w - s(18) - (status_bbox[2] - status_bbox[0]), header_y), status, font=fonts["small"], fill=(144, 153, 168))

        items = [
            ("RECOVERY", self._whoop_value_text(recovery, percent=True), self._whoop_progress(recovery, 100), self._whoop_recovery_color(recovery.get("value")), recovery.get("zone", "")),
            ("SLEEP", self._whoop_value_text(sleep, percent=True), self._whoop_progress(sleep, 100), (45, 212, 255), self._whoop_sleep_caption(sleep)),
            ("STRAIN", self._whoop_value_text(strain, percent=False), self._whoop_progress(strain, 21), (215, 255, 53), "OF 21" if strain else ""),
        ]

        cell_w = (card_w - s(36)) / 3.0
        radius = max(s(30), min(s(46), int(cell_w * 0.27)))
        center_y = y + s(88)
        for idx, (label, value_text, progress, color, caption) in enumerate(items):
            center_x = int(x + s(18) + cell_w * (idx + 0.5))
            self._draw_whoop_ring(draw, center_x, center_y, radius, progress, color, value_text, label, caption, fonts, scale)
        self._commit_pil_draw(img, draw)

    def _draw_whoop_ring(self, draw, center_x, center_y, radius, progress, color, value_text, label, caption, fonts, scale):
        progress = max(0.0, min(1.0, float(progress or 0.0)))
        s = lambda v: max(1, int(round(v * scale)))
        ring_w = max(4, s(8))
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        inner = radius - ring_w - s(3)
        draw.ellipse((center_x - inner, center_y - inner, center_x + inner, center_y + inner), fill=(15, 18, 23))
        draw.arc(bbox, -90, 269, fill=(47, 54, 64), width=ring_w)
        if progress > 0:
            draw.arc(bbox, -90, -90 + int(359 * progress), fill=color, width=ring_w)
        draw.ellipse((center_x + radius - s(5), center_y - s(5), center_x + radius + s(5), center_y + s(5)), fill=color)

        value = value_text or "--"
        value_bbox = draw.textbbox((0, 0), value, font=fonts["medium_bold"])
        draw.text((center_x - (value_bbox[2] - value_bbox[0]) // 2, center_y - s(11)), value, font=fonts["medium_bold"], fill=(248, 250, 252))

        label_bbox = draw.textbbox((0, 0), label, font=fonts["small_bold"])
        draw.text((center_x - (label_bbox[2] - label_bbox[0]) // 2, center_y + radius + s(9)), label, font=fonts["small_bold"], fill=(205, 211, 222))
        if caption:
            caption = str(caption).upper()
            caption_bbox = draw.textbbox((0, 0), caption, font=fonts["small"])
            draw.text((center_x - (caption_bbox[2] - caption_bbox[0]) // 2, center_y + radius + s(27)), caption, font=fonts["small"], fill=(125, 134, 150))

    def _whoop_value_text(self, item, percent):
        value = self._number(item.get("value") if isinstance(item, dict) else None)
        if value is None:
            return "--"
        if percent:
            return f"{int(round(value))}%"
        return f"{value:.1f}"

    def _whoop_progress(self, item, max_value):
        value = self._number(item.get("value") if isinstance(item, dict) else None)
        if value is None:
            return 0.0
        return value / max(1.0, float(max_value))

    def _whoop_recovery_color(self, value):
        value = self._number(value)
        if value is None:
            return (75, 85, 99)
        if value >= 67:
            return (24, 214, 107)
        if value >= 34:
            return (255, 210, 51)
        return (255, 75, 85)

    def _whoop_sleep_caption(self, sleep):
        hours = self._number(sleep.get("duration_hours") if isinstance(sleep, dict) else None)
        if not hours:
            return ""
        whole = int(hours)
        minutes = int(round((hours - whole) * 60))
        return f"{whole}H {minutes:02d}M"

    def _draw_fatigue_card(self, img, frame_info, hud_style):
        fatigue = frame_info.get("fatigue") or {}
        if not isinstance(fatigue, dict) or not fatigue.get("available"):
            return
        score = int(fatigue.get("score", 0) or 0)
        if score < 34 and not fatigue.get("break_recommended"):
            return

        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))
        card_w = min(img_w - s(60), s(380))
        card_h = s(106)
        x = s(30)
        y = s(30)
        if hud_style == "Minimal":
            y = s(96)
        elif hud_style == "Study Log":
            y = s(298)
        elif hud_style == "Cinematic":
            y = s(116)
        if x + card_w > img_w or y + card_h > img_h:
            return

        color = self._fatigue_color(score, fatigue.get("break_recommended"))
        self._glass_panel(img, x, y, card_w, card_h, s(20))
        draw, fonts = self._pil_draw(img, scale)
        draw.text((x + s(18), y + s(12)), "FATIGUE", font=fonts["small_bold"], fill=color)
        state = self._ellipsize(str(fatigue.get("state", "Watch")), 24)
        draw.text((x + s(18), y + s(34)), state, font=fonts["medium_bold"], fill=(248, 250, 252))

        value = f"{score}%"
        bbox = draw.textbbox((0, 0), value, font=fonts["large"])
        draw.text((x + card_w - s(18) - (bbox[2] - bbox[0]), y + s(20)), value, font=fonts["large"], fill=(248, 250, 252))

        bar_x = x + s(18)
        bar_y = y + s(70)
        bar_w = card_w - s(36)
        draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + s(8)), radius=s(4), fill=(42, 48, 58))
        draw.rounded_rectangle((bar_x, bar_y, bar_x + int(bar_w * max(0, min(1, score / 100))), bar_y + s(8)), radius=s(4), fill=color)

        if fatigue.get("break_recommended"):
            note = f"Break {fatigue.get('recommended_break_label', '')}".strip()
        else:
            reasons = fatigue.get("reasons", [])
            note = reasons[0].title() if reasons else fatigue.get("expression", "")
        draw.text((x + s(18), y + s(83)), self._ellipsize(str(note), 38), font=fonts["small"], fill=(190, 198, 214))
        self._commit_pil_draw(img, draw)

    def _fatigue_color(self, score, break_recommended=False):
        if break_recommended or score >= 72:
            return (255, 75, 85)
        if score >= 42:
            return (255, 210, 51)
        return (45, 212, 255)

    def _draw_music_card(self, img, frame_info):
        music_str = frame_info.get("music", "").strip()
        if not music_str:
            return

        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))
        if img_w < s(280) or img_h < s(260):
            return

        fonts = self._fonts(scale)
        measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        title, artist = self._split_music_label(music_str)

        cover_size = s(138)
        pad = s(16)
        gap = s(12)
        card_w = min(img_w - s(56), max(s(238), cover_size + pad * 2))
        text_w = max(1, card_w - pad * 2)
        title_lines = self._wrap_text(measure, title, fonts["small_bold"], text_w, 2)
        artist_line = self._ellipsize_to_width(measure, artist, fonts["small"], text_w) if artist else ""
        line_h = s(19)
        artist_h = s(20) if artist_line else 0
        text_h = max(line_h, line_h * len(title_lines)) + artist_h + s(18)
        card_h = pad + cover_size + gap + text_h + pad

        x = img_w - card_w - s(30)
        y = max(s(24), img_h - card_h - s(30))
        if x + card_w > img_w or y + card_h > img_h:
            return

        self._glass_panel(img, x, y, card_w, card_h, s(22))
        draw, fonts = self._pil_draw(img, scale)
        cover = self._load_cover_image(frame_info.get("music_cover", ""), cover_size)
        if cover is None:
            cover = self._placeholder_cover(cover_size, scale)

        cover_x = x + (card_w - cover_size) // 2
        cover_y = y + pad
        draw._img_ref.paste(cover, (cover_x, cover_y), cover)

        text_x = x + pad
        text_y = cover_y + cover_size + gap
        draw.text((text_x, text_y), "NOW PLAYING", font=fonts["small_bold"], fill=(151, 220, 255))
        text_y += s(18)
        for line in title_lines:
            draw.text((text_x, text_y), line, font=fonts["small_bold"], fill=(248, 250, 252))
            text_y += line_h
        if artist_line:
            draw.text((text_x, text_y + s(2)), artist_line, font=fonts["small"], fill=(190, 198, 214))
        self._commit_pil_draw(img, draw)

    def _load_cover_image(self, rel_path, size):
        rel_path = (rel_path or "").replace("\\", os.sep).replace("/", os.sep)
        if not rel_path:
            return None

        cache_key = (rel_path, size)
        if cache_key in self.cover_cache:
            return self.cover_cache[cache_key].copy()

        cover_path = os.path.abspath(os.path.normpath(os.path.join(self.session_dir, rel_path)))
        session_root = os.path.abspath(self.session_dir)
        if not cover_path.startswith(session_root + os.sep) or not os.path.exists(cover_path):
            return None

        try:
            with Image.open(cover_path) as source:
                source = source.convert("RGB")
                w, h = source.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                resample = getattr(Image, "Resampling", Image).LANCZOS
                cover = source.crop((left, top, left + side, top + side)).resize((size, size), resample)
                cover = cover.convert("RGBA")
        except Exception:
            return None

        radius = max(6, size // 12)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
        cover.putalpha(mask)
        self.cover_cache[cache_key] = cover.copy()
        return cover

    def _placeholder_cover(self, size, scale):
        cover = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(cover)
        radius = max(6, size // 12)
        for y in range(size):
            t = y / max(1, size - 1)
            r = int(234 * (1 - t) + 82 * t)
            g = int(78 * (1 - t) + 120 * t)
            b = int(96 * (1 - t) + 230 * t)
            draw.line((0, y, size, y), fill=(r, g, b, 255))
        draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, outline=(255, 255, 255, 76), width=max(1, int(2 * scale)))
        font = self._fonts(scale)["small_bold"]
        label = "MUSIC"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(((size - (bbox[2] - bbox[0])) // 2, (size - (bbox[3] - bbox[1])) // 2), label, font=font, fill=(255, 255, 255, 224))
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
        cover.putalpha(mask)
        return cover

    def _split_music_label(self, music_str):
        if " - " not in music_str:
            return music_str, ""
        title, artist = music_str.rsplit(" - ", 1)
        return title.strip() or music_str, artist.strip()

    def _wrap_text(self, draw, text, font, max_width, max_lines):
        words = text.split()
        if not words:
            return [""]

        lines = []
        current = ""
        truncated = False
        for idx, word in enumerate(words):
            candidate = f"{current} {word}".strip()
            if self._text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)
                current = word
            else:
                lines.append(self._ellipsize_to_width(draw, word, font, max_width))
                current = ""

            if len(lines) == max_lines:
                truncated = idx < len(words) - 1 or bool(current)
                break

        if current and len(lines) < max_lines:
            lines.append(current)
        elif current:
            truncated = True
        if not lines:
            lines = [self._ellipsize_to_width(draw, text, font, max_width)]

        if truncated and lines:
            lines[-1] = self._ellipsize_to_width(draw, lines[-1] + "...", font, max_width)
        return [self._ellipsize_to_width(draw, line, font, max_width) for line in lines[:max_lines]]

    def _ellipsize_to_width(self, draw, text, font, max_width):
        text = text.strip()
        if self._text_width(draw, text, font) <= max_width:
            return text
        suffix = "..."
        while text and self._text_width(draw, text + suffix, font) > max_width:
            text = text[:-1].rstrip()
        return text + suffix if text else suffix

    def _text_width(self, draw, text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def _number(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _draw_event_badge(self, img, text):
        img_h, img_w = img.shape[:2]
        scale = max(0.72, img_w / 1920.0)
        s = lambda v: max(1, int(round(v * scale)))
        badge_w = min(img_w - s(80), s(720))
        badge_h = s(62)
        x = (img_w - badge_w) // 2
        y = s(36)
        self._glass_panel(img, x, y, badge_w, badge_h, s(24))
        draw, fonts = self._pil_draw(img, scale)
        label = self._ellipsize(text, 56)
        bbox = draw.textbbox((0, 0), label, font=fonts["medium_bold"])
        draw.text((x + (badge_w - (bbox[2] - bbox[0])) // 2, y + s(17)), label, font=fonts["medium_bold"], fill=(255, 255, 255))
        self._commit_pil_draw(img, draw)

    def _glass_panel(self, img, x, y, w, h, radius):
        img_h, img_w = img.shape[:2]
        x = max(0, min(int(x), img_w - 1))
        y = max(0, min(int(y), img_h - 1))
        w = max(1, min(int(w), img_w - x))
        h = max(1, min(int(h), img_h - y))
        if w < 3 or h < 3:
            return
        radius = max(1, min(int(radius), min(w, h) // 2))
        roi = img[y:y + h, x:x + w]
        blur_k = min(max(3, radius * 2 + 1), min(w, h) if min(w, h) % 2 else min(w, h) - 1)
        blur_k = max(3, blur_k)
        blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
        blurred = cv2.convertScaleAbs(blurred, alpha=0.55, beta=12)

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
        mask_np = np.array(mask)[:, :, None] / 255.0

        glass = (blurred * mask_np + roi * (1 - mask_np)).astype(np.uint8)
        border = Image.new("L", (w, h), 0)
        border_draw = ImageDraw.Draw(border)
        border_draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, outline=255, width=max(1, radius // 14))
        border_np = np.array(border)[:, :, None] / 255.0
        glass = (255 * border_np * 0.24 + glass * (1 - border_np * 0.24)).astype(np.uint8)
        img[y:y + h, x:x + w] = glass

    def _solid_panel(self, img, x, y, w, h, alpha=0.65):
        img_h, img_w = img.shape[:2]
        x = max(0, min(int(x), img_w - 1))
        y = max(0, min(int(y), img_h - 1))
        w = max(1, min(int(w), img_w - x))
        h = max(1, min(int(h), img_h - y))
        roi = img[y:y + h, x:x + w]
        panel = np.zeros_like(roi)
        panel[:, :] = (10, 12, 16)
        radius = max(8, min(w, h) // 4)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=int(255 * alpha))
        mask_np = np.array(mask)[:, :, None] / 255.0
        img[y:y + h, x:x + w] = (panel * mask_np + roi * (1 - mask_np)).astype(np.uint8)

    def _pil_draw(self, img, scale):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        fonts = self._fonts(scale)
        draw._img_ref = img_pil
        return draw, fonts

    def _commit_pil_draw(self, img, draw):
        img[:] = cv2.cvtColor(np.array(draw._img_ref), cv2.COLOR_RGB2BGR)

    def _write_pause_screen(self, out, base_img, pause_dur):
        title = "Paused"
        subtitle = self._format_pause_str(pause_dur)
        details = ["Break logged", "Study timer held"]
        transition_frames = getattr(self, "pause_morph_frames", max(8, int(self.fps * 0.7)))

        final_card = None
        for t in np.linspace(0.0, 1.0, transition_frames):
            eased = self._ease_in_out(float(t))
            frame = self._create_interstitial_frame(
                base_img,
                eyebrow="TIMELAPSE HOLD",
                title=title,
                subtitle=subtitle,
                detail_lines=details,
                intensity=eased,
            )
            out.write(frame)
            final_card = frame

        if final_card is None:
            final_card = self._create_interstitial_frame(
                base_img,
                eyebrow="TIMELAPSE HOLD",
                title=title,
                subtitle=subtitle,
                detail_lines=details,
                intensity=1.0,
            )

        for _ in range(getattr(self, "pause_hold_frames", max(1, int(self.fps * 1.35)))):
            out.write(final_card)
        return final_card

    def _write_summary_screen(self, out, base_img, study_duration, focused_duration, focus_score):
        goal = self.options.get("goal_label", "").strip()
        metrics = [
            ("STUDY TIME", self._format_chrono(study_duration)),
            ("FOCUSED TIME", self._format_chrono(focused_duration)),
            ("FOCUS SCORE", f"{focus_score}%"),
        ]
        
        achievements = self.options.get("achievements", [])
        for ach in achievements:
            metrics.append(("ACHIEVEMENT", ach))
            
        subtitle = goal or "Study session complete"
        transition_frames = getattr(self, "summary_morph_frames", max(10, int(self.fps * 0.9)))

        final_card = None
        for t in np.linspace(0.0, 1.0, transition_frames):
            eased = self._ease_in_out(float(t))
            frame = self._create_interstitial_frame(
                base_img,
                eyebrow="SESSION COMPLETE",
                title="Nice work",
                subtitle=subtitle,
                metrics=metrics,
                intensity=eased,
            )
            out.write(frame)
            final_card = frame

        if final_card is None:
            final_card = self._create_interstitial_frame(
                base_img,
                eyebrow="SESSION COMPLETE",
                title="Nice work",
                subtitle=subtitle,
                metrics=metrics,
                intensity=1.0,
            )

        for _ in range(getattr(self, "summary_hold_frames", max(1, int(self.fps * 3.0)))):
            out.write(final_card)
        return final_card

    def _write_card_to_frame_transition(self, out, card_img, target_img):
        transition_frames = getattr(self, "resume_morph_frames", max(8, int(self.fps * 0.55)))
        for t in np.linspace(0.0, 1.0, transition_frames):
            eased = self._ease_in_out(float(t))
            blended = cv2.addWeighted(card_img, 1.0 - eased, target_img, eased, 0)
            out.write(blended)

    def _create_interstitial_frame(self, background, eyebrow, title, subtitle="", detail_lines=None, metrics=None, intensity=1.0):
        intensity = max(0.0, min(1.0, float(intensity)))
        frame = background.copy()
        h, w = frame.shape[:2]
        scale = w / 1920.0

        blur_k = 1
        if intensity > 0.01:
            blur_k = self._odd(max(3, int(round((3 + 72 * intensity) * scale))))
        if blur_k >= 3:
            frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)

        dark = np.zeros_like(frame)
        dark[:, :] = (18, 20, 26)
        frame = cv2.addWeighted(frame, 1.0 - 0.58 * intensity, dark, 0.58 * intensity, 0)

        vignette = self._vignette_mask(w, h)
        frame = (frame * (0.82 + 0.18 * vignette[:, :, None])).astype(np.uint8)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        fonts = self._interstitial_fonts(scale)

        text_alpha = int(255 * max(0.0, min(1.0, (intensity - 0.18) / 0.82)))
        if text_alpha > 0:
            center_x = w // 2
            block_center_y = int(h * (0.48 if metrics else 0.50))

            eyebrow_fill = (175, 222, 255, int(text_alpha * 0.82))
            title_fill = (248, 250, 252, text_alpha)
            subtitle_fill = (205, 213, 225, int(text_alpha * 0.92))

            self._draw_centered_text(draw, eyebrow, fonts["eyebrow"], center_x, block_center_y - int(132 * scale), eyebrow_fill)
            self._draw_centered_text(draw, title, fonts["display"], center_x, block_center_y - int(76 * scale), title_fill)

            if subtitle:
                self._draw_centered_text(draw, subtitle, fonts["subtitle"], center_x, block_center_y + int(8 * scale), subtitle_fill)

            if detail_lines:
                detail_y = block_center_y + int(62 * scale)
                for line in detail_lines:
                    self._draw_centered_text(draw, line, fonts["body"], center_x, detail_y, (180, 190, 205, int(text_alpha * 0.78)))
                    detail_y += int(34 * scale)

            if metrics:
                self._draw_metric_strip(draw, metrics, fonts, w, h, scale, text_alpha)

        composed = Image.alpha_composite(img, overlay).convert("RGB")
        return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)

    def _draw_metric_strip(self, draw, metrics, fonts, w, h, scale, alpha):
        strip_w = min(int(760 * scale), int(w * 0.72))
        item_w = strip_w // max(1, len(metrics))
        x0 = (w - strip_w) // 2
        y0 = int(h * 0.60)

        for idx, (label, value) in enumerate(metrics):
            x = x0 + idx * item_w
            self._draw_centered_text(draw, label, fonts["eyebrow"], x + item_w // 2, y0, (165, 175, 190, int(alpha * 0.82)))
            self._draw_centered_text(draw, value, fonts["metric"], x + item_w // 2, y0 + int(40 * scale), (248, 250, 252, alpha))

        line_y = y0 - int(24 * scale)
        draw.line((x0, line_y, x0 + strip_w, line_y), fill=(255, 255, 255, int(alpha * 0.18)), width=max(1, int(1 * scale)))

    def _draw_centered_text(self, draw, text, font, center_x, y, fill):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((center_x - text_w // 2, y), text, font=font, fill=fill)

    def _interstitial_fonts(self, scale):
        cache_key = ("interstitial", round(float(scale), 3))
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        def load(candidates, size):
            size = max(10, int(round(size * scale)))
            for name in candidates:
                try:
                    return ImageFont.truetype(name, size)
                except IOError:
                    continue
            return ImageFont.load_default()

        regular = ["segoeui.ttf", "arial.ttf"]
        semibold = ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "arial.ttf"]
        bold = ["segoeuib.ttf", "arialbd.ttf", "arial.ttf"]
        fonts = {
            "display": load(semibold, 86),
            "subtitle": load(regular, 30),
            "body": load(regular, 20),
            "eyebrow": load(bold, 15),
            "metric": load(semibold, 46),
        }
        self.font_cache[cache_key] = fonts
        return fonts

    def _vignette_mask(self, width, height):
        cache_key = (int(width), int(height))
        if cache_key in self.vignette_cache:
            return self.vignette_cache[cache_key]
        y, x = np.ogrid[:height, :width]
        cx, cy = width / 2.0, height / 2.0
        dist = np.sqrt(((x - cx) / max(1, cx)) ** 2 + ((y - cy) / max(1, cy)) ** 2)
        mask = 1.0 - np.clip((dist - 0.25) / 0.85, 0, 1)
        mask = mask.astype(np.float32)
        self.vignette_cache[cache_key] = mask
        return mask

    def _odd(self, value):
        value = max(1, int(value))
        return value if value % 2 else value + 1

    def _ease_in_out(self, t):
        t = max(0.0, min(1.0, float(t)))
        return t * t * (3.0 - 2.0 * t)

    def _fonts(self, scale):
        cache_key = ("hud", round(float(scale), 3))
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        def load(name, size):
            size = max(10, int(round(size * scale)))
            for font_name in name:
                try:
                    return ImageFont.truetype(font_name, size)
                except IOError:
                    continue
            return ImageFont.load_default()

        regular = ["segoeui.ttf", "arial.ttf"]
        bold = ["segoeuib.ttf", "arialbd.ttf", "arial.ttf"]
        fonts = {
            "large": load(regular, 31),
            "medium": load(regular, 20),
            "medium_bold": load(bold, 20),
            "small": load(regular, 15),
            "small_bold": load(bold, 15),
        }
        self.font_cache[cache_key] = fonts
        return fonts

    def _summary_text(self, study_duration, focus_score):
        lines = [
            "Session Complete",
            f"Total Study Time: {self._format_chrono(study_duration)}",
            f"Focus Score: {focus_score}%",
        ]
        goal = self.options.get("goal_label", "").strip()
        if goal:
            lines.insert(1, goal)
        return "\n".join(lines)

    def _format_chrono(self, study_duration):
        m, _ = divmod(int(study_duration), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}h {m:02d}m"
        return f"{m:02d}m"

    def _format_pause_str(self, duration_secs):
        m, s = divmod(int(duration_secs), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h} hr {m} mins pause"
        if m > 0:
            return f"{m} mins pause"
        return f"{s} seconds pause"

    def _create_text_screen(self, width, height, text):
        pause_img = Image.new("RGB", (width, height), color=(14, 15, 20))
        draw = ImageDraw.Draw(pause_img)
        scale = width / 1920.0
        font_size = max(12, int(round(56 * scale)))
        line_height = max(16, int(round(82 * scale)))

        try:
            font = ImageFont.truetype("segoeuib.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arialbd.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        lines = text.split("\n")
        y_offset = (height - (len(lines) * line_height)) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = (width - text_w) // 2
            draw.text((text_x, y_offset), line, font=font, fill=(240, 240, 245))
            y_offset += line_height

        return cv2.cvtColor(np.array(pause_img), cv2.COLOR_RGB2BGR)

    def _ellipsize(self, text, max_len):
        if len(text) <= max_len:
            return text
        return text[:max(0, max_len - 3)] + "..."
        
    def _create_social_export(self, session_dir, timeline):
        frames_to_process = timeline["frames"]
        if not frames_to_process:
            return None
        social_path = os.path.join(session_dir, "social_export.mp4")
        target_frames = int(10 * self.fps)
        sample_count = min(len(frames_to_process), target_frames)
        sample_indices = np.linspace(0, len(frames_to_process) - 1, sample_count, dtype=int)

        out_social = cv2.VideoWriter(social_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (1080, 1920))
        if not out_social.isOpened():
            return None
        focus_score = timeline["focus_score"]
        fonts = self._interstitial_fonts(1.4)
        source_mode = self._source_mode([frames_to_process[int(i)] for i in sample_indices])
        source_cap = cv2.VideoCapture(self.frames_video) if source_mode == "video" else None

        try:
            for i in sample_indices:
                frame_info = frames_to_process[int(i)]
                if source_cap is not None:
                    source_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_info["frame_index"]))
                    ok, img = source_cap.read()
                    if not ok:
                        continue
                else:
                    img = self._read_jpeg(frame_info)
                    if img is None:
                        continue

                cropped = self._crop_resize(img, 9 / 16, 1080, 1920)
                blurred = cv2.GaussianBlur(cropped, (71, 71), 24)
                pil_img = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((88, 1510), f"FOCUS {focus_score}%", font=fonts["display"], fill=(255, 255, 255))
                draw.text((92, 1635), "Study Timelapse", font=fonts["subtitle"], fill=(200, 210, 228))
                out_social.write(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
        finally:
            if source_cap is not None:
                source_cap.release()
            out_social.release()
        return social_path
