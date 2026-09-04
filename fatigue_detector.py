import math


class FatigueDetector:
    def __init__(self):
        self.score = 0.0
        self.eye_fade_seconds = 0.0
        self.slouch_seconds = 0.0
        self.high_fatigue_seconds = 0.0
        self.low_motion_seconds = 0.0
        self.previous_center = None
        self.last_snapshot = self._empty_snapshot("Waiting for face/body")

    def reset(self):
        self.__init__()

    def snapshot(self):
        return dict(self.last_snapshot)

    def update(self, pose_points, focused, posture, activity, dt, study_elapsed_sec=0.0, whoop=None):
        dt = max(0.0, float(dt or 0.0))
        pose_points = pose_points or {}
        whoop = whoop or {}

        signals = self._pose_signals(pose_points)
        if not signals["available"]:
            self.score = max(0.0, self.score - 10.0 * dt)
            self.eye_fade_seconds = max(0.0, self.eye_fade_seconds - 2.0 * dt)
            self.slouch_seconds = max(0.0, self.slouch_seconds - dt)
            self.high_fatigue_seconds = max(0.0, self.high_fatigue_seconds - 2.0 * dt)
            self.last_snapshot = self._empty_snapshot("Face/body not detected")
            self.last_snapshot["score"] = int(round(self.score))
            return self.snapshot()

        eye_fade = signals["eye_fade"]
        body_slump = max(signals["head_drop"], 0.7 if posture == "slouching" else 0.0)
        if eye_fade >= 0.58:
            self.eye_fade_seconds += dt
        else:
            self.eye_fade_seconds = max(0.0, self.eye_fade_seconds - 2.5 * dt)

        if body_slump >= 0.56 or posture in ("slouching", "leaning"):
            self.slouch_seconds += dt
        else:
            self.slouch_seconds = max(0.0, self.slouch_seconds - 1.5 * dt)

        if signals["low_motion"]:
            self.low_motion_seconds += dt
        else:
            self.low_motion_seconds = max(0.0, self.low_motion_seconds - 3.0 * dt)

        readiness_penalty, readiness_reasons = self._readiness_penalty(whoop)
        long_block_penalty = self._long_block_penalty(study_elapsed_sec)

        instant = 6.0
        instant += signals["head_drop"] * 34.0
        instant += signals["eye_fade"] * 30.0
        instant += signals["face_turn"] * 11.0
        instant += signals["shoulder_tilt"] * 16.0
        instant += readiness_penalty
        instant += long_block_penalty
        if not focused and activity != "Distracted (Phone)":
            instant += 8.0
        if self.low_motion_seconds >= 120 and max(signals["head_drop"], signals["eye_fade"]) > 0.35:
            instant += 8.0
        instant = max(0.0, min(100.0, instant))

        blend = 0.22 if dt <= 1.5 else 0.35
        self.score = self.score * (1.0 - blend) + instant * blend
        if self.score >= 66:
            self.high_fatigue_seconds += dt
        else:
            self.high_fatigue_seconds = max(0.0, self.high_fatigue_seconds - 2.0 * dt)

        reasons = self._reasons(signals, posture, study_elapsed_sec, readiness_reasons)
        break_recommended = (
            self.high_fatigue_seconds >= 35
            or self.eye_fade_seconds >= 18
            or self.slouch_seconds >= 150
            or self.score >= 78
        )
        break_seconds = self._recommended_break_seconds(study_elapsed_sec, readiness_penalty) if break_recommended else 0

        state = self._state(self.score, break_recommended)
        self.last_snapshot = {
            "available": True,
            "source": "YOLO pose ONNX",
            "score": int(round(self.score)),
            "state": state,
            "expression": self._expression(signals, posture),
            "break_recommended": break_recommended,
            "recommended_break_seconds": break_seconds,
            "recommended_break_label": self._format_break(break_seconds),
            "reasons": reasons,
            "eye_fade_seconds": round(self.eye_fade_seconds, 1),
            "slouch_seconds": round(self.slouch_seconds, 1),
            "high_fatigue_seconds": round(self.high_fatigue_seconds, 1),
            "signals": {key: round(value, 3) if isinstance(value, float) else value for key, value in signals.items()},
        }
        return self.snapshot()

    def _pose_signals(self, points):
        nose = self._point(points, "nose")
        left_eye = self._point(points, "left_eye")
        right_eye = self._point(points, "right_eye")
        left_ear = self._point(points, "left_ear")
        right_ear = self._point(points, "right_ear")
        left_shoulder = self._point(points, "left_shoulder")
        right_shoulder = self._point(points, "right_shoulder")

        available = bool(nose and (left_eye or right_eye or left_shoulder or right_shoulder))
        shoulder_width = self._distance(left_shoulder, right_shoulder) if left_shoulder and right_shoulder else 0.0
        shoulder_tilt = 0.0
        head_drop = 0.0
        if shoulder_width > 1.0 and nose and left_shoulder and right_shoulder:
            shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            head_above_ratio = (shoulder_mid_y - nose[1]) / shoulder_width
            head_drop = self._clamp((0.46 - head_above_ratio) / 0.24)
            shoulder_tilt = self._clamp(abs(left_shoulder[1] - right_shoulder[1]) / shoulder_width / 0.32)

        eye_conf = [p[2] for p in (left_eye, right_eye) if p]
        avg_eye_conf = sum(eye_conf) / len(eye_conf) if eye_conf else 0.0
        eye_fade = 0.0
        if nose:
            eye_fade = self._clamp((0.62 - avg_eye_conf) / 0.50)
            if left_eye and right_eye:
                eye_line_tilt = abs(left_eye[1] - right_eye[1]) / max(1.0, abs(left_eye[0] - right_eye[0]))
                eye_fade = max(eye_fade, self._clamp((eye_line_tilt - 0.22) / 0.42) * 0.45)

        face_turn = 0.0
        if nose and left_eye and right_eye:
            eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
            eye_span = max(1.0, abs(left_eye[0] - right_eye[0]))
            face_turn = self._clamp((abs(nose[0] - eye_mid_x) / eye_span - 0.38) / 0.62)
        elif nose and (left_eye or right_eye):
            face_turn = 0.35
        elif nose and (left_ear or right_ear):
            face_turn = 0.45

        low_motion = False
        center = self._center([nose, left_eye, right_eye, left_shoulder, right_shoulder])
        if center and shoulder_width > 1:
            if self.previous_center:
                motion = self._distance(center, self.previous_center) / shoulder_width
                low_motion = motion < 0.014
            self.previous_center = center

        return {
            "available": available,
            "head_drop": head_drop,
            "eye_fade": eye_fade,
            "face_turn": face_turn,
            "shoulder_tilt": shoulder_tilt,
            "low_motion": low_motion,
            "eye_confidence": avg_eye_conf,
        }

    def _readiness_penalty(self, whoop):
        reasons = []
        penalty = 0.0
        recovery = self._number(whoop.get("recovery_score"))
        sleep = self._number(whoop.get("sleep_performance"))
        if recovery is not None and recovery < 50:
            penalty += 7.0 if recovery >= 34 else 12.0
            reasons.append("low recovery")
        if sleep is not None and sleep < 75:
            penalty += 5.0 if sleep >= 60 else 10.0
            reasons.append("low sleep")
        return min(18.0, penalty), reasons

    def _long_block_penalty(self, seconds):
        minutes = max(0.0, float(seconds or 0.0) / 60.0)
        if minutes >= 120:
            return 16.0
        if minutes >= 90:
            return 11.0
        if minutes >= 55:
            return 6.0
        return 0.0

    def _recommended_break_seconds(self, study_elapsed_sec, readiness_penalty):
        minutes = 5
        if self.score >= 72:
            minutes = 8
        if self.score >= 84:
            minutes = 12
        if self.score >= 92:
            minutes = 18
        if self.eye_fade_seconds >= 30:
            minutes += 4
        if self.slouch_seconds >= 240:
            minutes += 3
        if study_elapsed_sec >= 90 * 60:
            minutes += 4
        if readiness_penalty >= 10:
            minutes += 3
        return int(max(3, min(25, minutes)) * 60)

    def _reasons(self, signals, posture, study_elapsed_sec, readiness_reasons):
        scored = []
        if signals["eye_fade"] >= 0.48:
            scored.append((signals["eye_fade"], "eyes fading"))
        if signals["head_drop"] >= 0.45 or posture == "slouching":
            scored.append((max(signals["head_drop"], 0.55), "head dropping"))
        if signals["shoulder_tilt"] >= 0.55 or posture == "leaning":
            scored.append((max(signals["shoulder_tilt"], 0.5), "body leaning"))
        if study_elapsed_sec >= 55 * 60:
            scored.append((0.48, "long focus block"))
        for reason in readiness_reasons:
            scored.append((0.45, reason))
        scored.sort(reverse=True)
        return [reason for _, reason in scored[:3]]

    def _expression(self, signals, posture):
        if signals["eye_fade"] >= 0.62:
            return "Eyes fading"
        if signals["head_drop"] >= 0.58 or posture == "slouching":
            return "Head dropping"
        if signals["shoulder_tilt"] >= 0.58 or posture == "leaning":
            return "Body leaning"
        if signals["face_turn"] >= 0.65:
            return "Looking away"
        return "Alert"

    def _state(self, score, break_recommended):
        if break_recommended:
            return "Break recommended"
        if score >= 58:
            return "Tired"
        if score >= 34:
            return "Watch"
        return "Fresh"

    def _empty_snapshot(self, state):
        return {
            "available": False,
            "source": "YOLO pose ONNX",
            "score": 0,
            "state": state,
            "expression": "Unknown",
            "break_recommended": False,
            "recommended_break_seconds": 0,
            "recommended_break_label": "No break",
            "reasons": [],
            "eye_fade_seconds": 0.0,
            "slouch_seconds": 0.0,
            "high_fatigue_seconds": 0.0,
            "signals": {},
        }

    def _format_break(self, seconds):
        if not seconds:
            return "No break"
        minutes = int(round(seconds / 60.0))
        return f"{minutes} min"

    def _point(self, points, key, min_conf=0.25):
        value = points.get(key)
        if not value or len(value) < 3:
            return None
        if value[2] < min_conf:
            return None
        return value

    def _center(self, points):
        valid = [p for p in points if p]
        if not valid:
            return None
        return (sum(p[0] for p in valid) / len(valid), sum(p[1] for p in valid) / len(valid), 1.0)

    def _distance(self, a, b):
        if not a or not b:
            return 0.0
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _clamp(self, value):
        return max(0.0, min(1.0, float(value or 0.0)))

    def _number(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
