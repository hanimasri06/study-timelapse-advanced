import sys
import os
import json
import glob
import threading
import time
import gc
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from capture_engine import CaptureEngine
from renderer import Renderer


class AppSignals(QObject):
    render_progress = Signal(str, float)
    render_complete = Signal(object, bool)
    coach_clear = Signal()
    coach_append = Signal(str)
    coach_done = Signal(str)
    whoop_status = Signal(object)
    preview_ready = Signal(object, int, int, int)


class MiniPlayer(QWidget):
    toggle_requested = Signal()
    expand_requested = Signal()

    def __init__(self, colors):
        super().__init__()
        self.colors = colors
        self._drag_offset = None
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setFixedSize(284, 138)
        self.setObjectName("mini")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.time_label = QLabel("00:00:00")
        self.time_label.setObjectName("miniTime")
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)

        self.focus_label = QLabel("Focus: 00:00:00  |  Break: 00:00:00")
        self.focus_label.setObjectName("muted")
        self.focus_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.focus_label)

        self.detail_label = QLabel("Not playing")
        self.detail_label.setObjectName("muted")
        self.detail_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detail_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.toggle_button = QPushButton("Pause")
        self.toggle_button.clicked.connect(self.toggle_requested.emit)
        buttons.addWidget(self.toggle_button)
        expand = QPushButton("Expand")
        expand.setObjectName("secondaryButton")
        expand.clicked.connect(self.expand_requested.emit)
        buttons.addWidget(expand)
        layout.addLayout(buttons)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None


class WhoopMetricRing(QWidget):
    def __init__(self, label, color):
        super().__init__()
        self.label = label
        self.value_text = "--"
        self.caption = ""
        self.progress = 0.0
        self.color = color
        self.setMinimumSize(104, 138)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_metric(self, value_text, progress, color=None, caption=""):
        self.value_text = value_text
        self.progress = max(0.0, min(1.0, float(progress or 0.0)))
        if color:
            self.color = color
        self.caption = caption
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        width = self.width()
        diameter = max(58, min(width - 18, self.height() - 56))
        x = (width - diameter) / 2
        y = 8
        rect = QRectF(x, y, diameter, diameter)
        ring_width = max(6, int(diameter * 0.095))

        inner = QRectF(x + ring_width + 5, y + ring_width + 5, diameter - (ring_width + 5) * 2, diameter - (ring_width + 5) * 2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#11151C"))
        painter.drawEllipse(inner)

        painter.setPen(QPen(QColor("#313741"), ring_width, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 90 * 16, -360 * 16)
        painter.setPen(QPen(QColor(self.color), ring_width, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 90 * 16, int(-360 * 16 * self.progress))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(self.color))
        painter.drawEllipse(QRectF(x + diameter - ring_width - 2, y + diameter / 2 - 4, 8, 8))

        value_font = QFont("Segoe UI", max(13, int(diameter * 0.23)))
        value_font.setWeight(QFont.Bold)
        painter.setFont(value_font)
        painter.setPen(QColor("#F8FAFC" if self.value_text != "--" else "#6F7787"))
        painter.drawText(rect, Qt.AlignCenter, self.value_text)

        label_font = QFont("Segoe UI", 9)
        label_font.setWeight(QFont.Bold)
        painter.setFont(label_font)
        painter.setPen(QColor("#D4DAE5"))
        painter.drawText(QRectF(0, y + diameter + 8, width, 18), Qt.AlignCenter, self.label.upper())

        if self.caption:
            caption_font = QFont("Segoe UI", 8)
            painter.setFont(caption_font)
            painter.setPen(QColor("#8A93A5"))
            painter.drawText(QRectF(0, y + diameter + 26, width, 16), Qt.AlignCenter, self.caption)


class TimelapseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Study Timelapse Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1160, 740)

        self.colors = {
            "bg": "#0F1115",
            "sidebar": "#151821",
            "surface": "#1B1F2A",
            "surface2": "#232937",
            "border": "#303747",
            "text": "#F8FAFC",
            "muted": "#A3AAB7",
            "subtle": "#6F7787",
            "accent": "#7DD3FC",
            "accent2": "#38BDF8",
            "success": "#22C55E",
            "warning": "#F59E0B",
            "danger": "#F43F5E",
        }
        self.setStyleSheet(self._stylesheet())

        self.signals = AppSignals()
        self.signals.render_progress.connect(self._set_render_progress)
        self.signals.render_complete.connect(self._render_complete)
        self.signals.coach_clear.connect(self._coach_clear)
        self.signals.coach_append.connect(self._coach_append)
        self.signals.coach_done.connect(self._coach_done)
        self.signals.whoop_status.connect(self._whoop_status_changed)
        self.signals.preview_ready.connect(self._preview_camera_ready)

        self.capture_engine = CaptureEngine()
        self.current_session_dir = None
        self.last_render_path = None
        self.preview_cap = None
        self.preview_open_generation = 0
        self.preview_open_lock = threading.Lock()
        self.preview_closing = False
        self.is_previewing = True
        self.is_rendering = False
        self.canvas_widget = None
        
        self.last_shield_time = 0

        self.session_summaries = []
        self.analytics_summary_cache = {}
        self.export_stats = {}
        self.mini = None

        self._build_ui()
        self._open_preview_camera()
        QTimer.singleShot(150, self._load_analytics)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(50)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)
        self.update_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(238)
        shell.addWidget(self.sidebar)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(18, 28, 18, 18)
        sidebar_layout.setSpacing(10)

        brand = QLabel("Study Studio")
        brand.setObjectName("brand")
        sidebar_layout.addWidget(brand)
        tagline = QLabel("Capture focus. Render beautiful recaps.")
        tagline.setObjectName("sidebarText")
        tagline.setWordWrap(True)
        sidebar_layout.addWidget(tagline)
        sidebar_layout.addSpacing(16)

        self.nav_buttons = {}
        for key, label in [
            ("session", "Live Session"),
            ("review", "Review"),
            ("coach", "Coach"),
            ("settings", "Settings"),
        ]:
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page=key: self._show_page(page))
            self.nav_buttons[key] = button
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch(1)
        self.sidebar_status_card = self._sidebar_status_card()
        sidebar_layout.addWidget(self.sidebar_status_card)

        main = QFrame()
        main.setObjectName("main")
        shell.addWidget(main, 1)

        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(18)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(2)
        self.page_title = QLabel("Live Session")
        self.page_title.setObjectName("pageTitle")
        header_text.addWidget(self.page_title)
        self.page_subtitle = QLabel("Capture a clean study session with focus, posture, and context.")
        self.page_subtitle.setObjectName("pageSubtitle")
        header_text.addWidget(self.page_subtitle)
        header.addLayout(header_text, 1)
        self.header_status = QLabel("Ready")
        self.header_status.setObjectName("statusPill")
        self.header_status.setAlignment(Qt.AlignCenter)
        self.header_status.setFixedWidth(132)
        header.addWidget(self.header_status)
        main_layout.addLayout(header)

        self.pages = QStackedWidget()
        main_layout.addWidget(self.pages, 1)

        self.session_page = self._build_session_page()
        self.review_page = self._build_review_page()
        self.coach_page = self._build_coach_page()
        self.settings_page = self._build_settings_page()

        self.pages.addWidget(self.session_page)
        self.pages.addWidget(self.review_page)
        self.pages.addWidget(self.coach_page)
        self.pages.addWidget(self.settings_page)
        self.page_indices = {
            "session": 0,
            "review": 1,
            "coach": 2,
            "settings": 3,
        }
        self._show_page("session")

    def _sidebar_status_card(self):
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)
        label = QLabel("Current Session")
        label.setObjectName("eyebrow")
        layout.addWidget(label)
        
        self.side_readiness = QLabel("Readiness: Standby")
        self.side_readiness.setObjectName("accentText")
        self.side_readiness.setStyleSheet(f"color: {self.colors['accent']}; font-weight: bold;")
        layout.addWidget(self.side_readiness)
        
        self.side_time = QLabel("00:00:00")
        self.side_time.setObjectName("sideTime")
        layout.addWidget(self.side_time)
        self.side_focus = QLabel("Focus: 00:00:00")
        self.side_focus.setObjectName("muted")
        layout.addWidget(self.side_focus)
        self.side_break = QLabel("Break: 00:00:00")
        self.side_break.setObjectName("muted")
        layout.addWidget(self.side_break)
        self.side_detail = QLabel("Idle")
        self.side_detail.setObjectName("muted")
        layout.addWidget(self.side_detail)
        return card

    def _build_session_page(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        layout.setColumnStretch(0, 5)
        layout.setColumnStretch(1, 2)
        layout.setRowStretch(0, 1)

        preview_card = self._card()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(18, 16, 18, 18)
        preview_layout.setSpacing(14)

        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.session_status_label = QLabel("Ready to capture")
        self.session_status_label.setObjectName("sectionTitle")
        title_box.addWidget(self.session_status_label)
        self.session_goal_label = QLabel(self._goal_label())
        self.session_goal_label.setObjectName("muted")
        title_box.addWidget(self.session_goal_label)
        top.addLayout(title_box, 1)
        self.quality_label = QLabel("Camera: 2560x1440")
        self.quality_label.setObjectName("muted")
        top.addWidget(self.quality_label)
        preview_layout.addLayout(top)

        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(420)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.video_label, 1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.btn_start = self._button("Start", "success")
        self.btn_start.clicked.connect(self.start_session)
        controls.addWidget(self.btn_start)
        self.btn_pause = self._button("Pause", "secondary")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_session)
        controls.addWidget(self.btn_pause)
        self.btn_resume = self._button("Resume", "secondary")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self.resume_session)
        controls.addWidget(self.btn_resume)
        self.btn_end = self._button("End & Render", "danger")
        self.btn_end.setEnabled(False)
        self.btn_end.clicked.connect(self.end_session)
        controls.addWidget(self.btn_end)
        self.btn_mini = self._button("Mini", "outline")
        self.btn_mini.clicked.connect(self.open_mini_player)
        controls.addWidget(self.btn_mini)
        preview_layout.addLayout(controls)
        layout.addWidget(preview_card, 0, 0)

        right = QVBoxLayout()
        right.setSpacing(18)

        signals_card = self._card()
        signal_layout = QVBoxLayout(signals_card)
        signal_layout.setContentsMargins(16, 16, 16, 16)
        signal_layout.setSpacing(10)
        signal_layout.addWidget(self._section_heading("Live Signals"))
        grid = QGridLayout()
        grid.setSpacing(10)
        self.lbl_live_focus = self._metric_box(grid, 0, 0, "Focus", "Standby")
        self.lbl_live_posture = self._metric_box(grid, 0, 1, "Posture", "Unknown")
        self.lbl_live_activity = self._metric_box(grid, 1, 0, "Activity", "None")
        self.lbl_live_backend = self._metric_box(grid, 1, 1, "AI Backend", "Pending")
        self.lbl_live_fatigue = self._metric_box(grid, 2, 0, "Fatigue", "Waiting")
        self.lbl_live_break = self._metric_box(grid, 2, 1, "Break", "No break")
        signal_layout.addLayout(grid)
        self.lbl_live_music = self._wide_metric(signal_layout, "Music", "Not playing")
        self.lbl_live_frames = self._wide_metric(signal_layout, "Capture", "0 frames")
        right.addWidget(signals_card)

        whoop_card = self._card()
        whoop_layout = QVBoxLayout(whoop_card)
        whoop_layout.setContentsMargins(16, 16, 16, 16)
        whoop_layout.setSpacing(10)
        whoop_header = QHBoxLayout()
        whoop_header.addWidget(self._section_heading("WHOOP"), 1)
        self.whoop_live_status = QLabel("Not connected")
        self.whoop_live_status.setObjectName("whoopStatus")
        self.whoop_live_status.setAlignment(Qt.AlignRight)
        whoop_header.addWidget(self.whoop_live_status)
        whoop_layout.addLayout(whoop_header)
        whoop_rings = QHBoxLayout()
        whoop_rings.setSpacing(8)
        self.whoop_recovery_ring = WhoopMetricRing("Recovery", self.colors["success"])
        self.whoop_sleep_ring = WhoopMetricRing("Sleep", self.colors["accent"])
        self.whoop_strain_ring = WhoopMetricRing("Strain", "#D7FF35")
        whoop_rings.addWidget(self.whoop_recovery_ring)
        whoop_rings.addWidget(self.whoop_sleep_ring)
        whoop_rings.addWidget(self.whoop_strain_ring)
        whoop_layout.addLayout(whoop_rings)
        right.addWidget(whoop_card)

        render_card = self._card()
        render_layout = QVBoxLayout(render_card)
        render_layout.setContentsMargins(16, 16, 16, 16)
        render_layout.setSpacing(10)
        render_layout.addWidget(self._section_heading("Render Queue"))
        self.render_progress_label = QLabel("Idle")
        self.render_progress_label.setObjectName("muted")
        render_layout.addWidget(self.render_progress_label)
        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 100)
        self.render_progress.setValue(0)
        self.render_progress.setTextVisible(False)
        render_layout.addWidget(self.render_progress)
        right.addWidget(render_card)

        plan_card = self._card()
        plan_layout = QVBoxLayout(plan_card)
        plan_layout.setContentsMargins(16, 16, 16, 16)
        plan_layout.setSpacing(12)
        plan_layout.addWidget(self._section_heading("Session Plan"))
        self.plan_goal_value = self._plan_row(plan_layout, "Goal", self._goal_label())
        self.plan_privacy_value = self._plan_row(plan_layout, "Privacy", self.privacy_combo_text())
        self.plan_capture_value = self._plan_row(plan_layout, "Capture", self.capture_rate_text())
        self.plan_render_value = self._plan_row(plan_layout, "Export", self.render_summary_text())
        settings_button = self._button("Open Settings", "secondary")
        settings_button.clicked.connect(lambda: self._show_page("settings"))
        plan_layout.addWidget(settings_button)
        right.addWidget(plan_card, 1)

        layout.addLayout(right, 0, 1)
        return page

    def _build_review_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        summaries = QHBoxLayout()
        summaries.setSpacing(14)
        total_card, self.lbl_total_time = self._summary_card("Study Time", "0h 0m", "accent")
        focus_card, self.lbl_avg_focus = self._summary_card("Average Focus", "0%", "success")
        activity_card, self.lbl_top_activity = self._summary_card("Primary Tool", "None", "warning")
        music_card, self.lbl_top_music = self._summary_card("Best Track", "None", "purple")
        summaries.addWidget(total_card)
        summaries.addWidget(focus_card)
        summaries.addWidget(activity_card)
        summaries.addWidget(music_card)
        layout.addLayout(summaries)

        self.chart_frame = self._card()
        self.chart_frame.setMinimumHeight(280)
        chart_layout = QVBoxLayout(self.chart_frame)
        chart_layout.setContentsMargins(12, 12, 12, 12)
        self.chart_mount = QWidget()
        self.chart_mount_layout = QVBoxLayout(self.chart_mount)
        self.chart_mount_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(self.chart_mount)
        layout.addWidget(self.chart_frame, 1)

        sessions_card = self._card()
        sessions_layout = QVBoxLayout(sessions_card)
        sessions_layout.setContentsMargins(16, 16, 16, 16)
        sessions_layout.setSpacing(12)
        header = QHBoxLayout()
        header.addWidget(self._section_heading("Sessions"), 1)
        refresh = self._button("Refresh", "secondary")
        refresh.clicked.connect(self._load_analytics)
        header.addWidget(refresh)
        sessions_layout.addLayout(header)

        self.sessions_scroll = QScrollArea()
        self.sessions_scroll.setWidgetResizable(True)
        self.sessions_scroll.setFrameShape(QFrame.NoFrame)
        self.sessions_content = QWidget()
        self.sessions_layout = QVBoxLayout(self.sessions_content)
        self.sessions_layout.setContentsMargins(0, 0, 0, 0)
        self.sessions_layout.setSpacing(10)
        self.sessions_layout.addStretch(1)
        self.sessions_scroll.setWidget(self.sessions_content)
        sessions_layout.addWidget(self.sessions_scroll)
        layout.addWidget(sessions_card, 2)
        return page

    def _build_coach_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        top = self._card()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(18, 16, 18, 16)
        copy = QVBoxLayout()
        copy.setSpacing(4)
        copy.addWidget(self._section_heading("AI Coach"))
        desc = QLabel("Generate a concise debrief from your session history and current goal.")
        desc.setObjectName("muted")
        copy.addWidget(desc)
        top_layout.addLayout(copy, 1)
        self.btn_coach = self._button("Generate Coach Insight", "primary")
        self.btn_coach.clicked.connect(self._generate_coach_insight)
        top_layout.addWidget(self.btn_coach)
        layout.addWidget(top)

        self.txt_coach = QTextEdit()
        self.txt_coach.setObjectName("coachText")
        self.txt_coach.setText("Ready to analyze your sessions. The model loads only when you ask for a coach insight.")
        layout.addWidget(self.txt_coach, 1)
        return page

    def _build_settings_page(self):
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        page.setWidget(content)

        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(18)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        session = self._settings_group("Session")
        self.goal_input = self._line_edit("Deep work session")
        self.goal_minutes_combo = self._combo(["25", "50", "60", "90", "120"], "60")
        self.capture_rate_combo = self._combo(["2 fps", "1 fps", "0.5 fps"], "1 fps")
        self.auto_pause_combo = self._combo(["60 sec", "120 sec", "180 sec", "300 sec"], "120 sec")
        self._settings_row(session, "Goal", self.goal_input)
        self._settings_row(session, "Goal length", self.goal_minutes_combo)
        self._settings_row(session, "Capture rate", self.capture_rate_combo)
        self._settings_row(session, "Auto-pause", self.auto_pause_combo)
        grid.addWidget(session, 0, 0)

        capture = self._settings_group("Capture")
        self.resolution_combo = self._combo(["2560x1440", "1920x1080", "1280x720"], "2560x1440")
        self.privacy_combo = self._combo(["Off", "Face Blur", "Background Blur", "Low Detail"], "Off")
        self._settings_row(capture, "Resolution", self.resolution_combo)
        self._settings_row(capture, "Privacy", self.privacy_combo)
        grid.addWidget(capture, 0, 1)

        render = self._settings_group("Render")
        self.hud_combo = self._combo(["Glass", "Minimal", "Study Log", "Cinematic", "Off"], "Glass")
        self.aspect_combo = self._combo(["Source 16:9", "Vertical 9:16", "Square 1:1"], "Source 16:9")
        self.duration_label = QLabel("Final video length: 30 sec")
        self.duration_label.setObjectName("muted")
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(15, 90)
        self.duration_slider.setSingleStep(5)
        self.duration_slider.setPageStep(5)
        self.duration_slider.setValue(30)
        self.duration_slider.valueChanged.connect(self._on_render_duration_change)
        self._settings_row(render, "HUD style", self.hud_combo)
        self._settings_row(render, "Output format", self.aspect_combo)
        self._settings_row(render, "Final length", self.duration_label)
        self._settings_row(render, "", self.duration_slider)
        grid.addWidget(render, 1, 0)

        actions = self._settings_group("Actions")
        self.btn_apply_settings = self._button("Apply Settings", "primary")
        self.btn_apply_settings.clicked.connect(self._apply_settings)
        self.btn_export = self._button("Export Summary Card", "secondary")
        self.btn_export.clicked.connect(self._export_summary_card)
        self._settings_row(actions, "", self.btn_apply_settings)
        self._settings_row(actions, "", self.btn_export)
        grid.addWidget(actions, 1, 1)

        whoop = self._settings_group("WHOOP")
        whoop_config = self.capture_engine.whoop_client.config
        self.whoop_client_id_input = self._line_edit(whoop_config.get("client_id", ""))
        self.whoop_client_secret_input = self._line_edit(whoop_config.get("client_secret", ""))
        self.whoop_client_secret_input.setEchoMode(QLineEdit.Password)
        self.whoop_redirect_input = self._line_edit(whoop_config.get("redirect_uri", self.capture_engine.whoop_client.DEFAULT_REDIRECT_URI))
        self.whoop_settings_status = QLabel("Connected" if self.capture_engine.whoop_client.is_connected() else "Not connected")
        self.whoop_settings_status.setObjectName("muted")
        self.whoop_code_input = self._line_edit("")
        self.whoop_code_input.setPlaceholderText("Paste the code or full redirect URL")
        self.btn_whoop_connect = self._button("Connect WHOOP", "primary")
        self.btn_whoop_connect.clicked.connect(self._connect_whoop)
        self.btn_whoop_finish = self._button("Finish with Code", "secondary")
        self.btn_whoop_finish.clicked.connect(self._finish_whoop_code)
        self.btn_whoop_refresh = self._button("Refresh WHOOP", "secondary")
        self.btn_whoop_refresh.clicked.connect(lambda: self._refresh_whoop(force=True))
        whoop_actions = QWidget()
        whoop_actions_layout = QHBoxLayout(whoop_actions)
        whoop_actions_layout.setContentsMargins(0, 0, 0, 0)
        whoop_actions_layout.setSpacing(10)
        whoop_actions_layout.addWidget(self.btn_whoop_connect)
        whoop_actions_layout.addWidget(self.btn_whoop_finish)
        whoop_actions_layout.addWidget(self.btn_whoop_refresh)
        self._settings_row(whoop, "Client ID", self.whoop_client_id_input)
        self._settings_row(whoop, "Client Secret", self.whoop_client_secret_input)
        self._settings_row(whoop, "Redirect URI", self.whoop_redirect_input)
        self._settings_row(whoop, "Auth Code", self.whoop_code_input)
        self._settings_row(whoop, "Status", self.whoop_settings_status)
        self._settings_row(whoop, "", whoop_actions)
        grid.addWidget(whoop, 2, 0, 1, 2)

        return page

    # ------------------------------------------------------------------
    # Small widget factories
    # ------------------------------------------------------------------
    def _card(self):
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    def _button(self, text, kind):
        button = QPushButton(text)
        button.setObjectName(f"{kind}Button")
        button.setMinimumHeight(40)
        return button

    def _section_heading(self, text):
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _metric_box(self, grid, row, col, label, value):
        box = QFrame()
        box.setObjectName("metricBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        label_widget = QLabel(label)
        label_widget.setObjectName("eyebrow")
        layout.addWidget(label_widget)
        value_widget = QLabel(value)
        value_widget.setObjectName("metricValue")
        layout.addWidget(value_widget)
        grid.addWidget(box, row, col)
        return value_widget

    def _wide_metric(self, parent_layout, label, value):
        box = QFrame()
        box.setObjectName("metricBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        label_widget = QLabel(label)
        label_widget.setObjectName("eyebrow")
        layout.addWidget(label_widget)
        value_widget = QLabel(value)
        value_widget.setObjectName("metricValue")
        layout.addWidget(value_widget)
        parent_layout.addWidget(box)
        return value_widget

    def _plan_row(self, parent_layout, label, value):
        row = QHBoxLayout()
        left = QLabel(label)
        left.setObjectName("muted")
        row.addWidget(left)
        right = QLabel(value)
        right.setObjectName("planValue")
        right.setAlignment(Qt.AlignRight)
        row.addWidget(right, 1)
        parent_layout.addLayout(row)
        return right

    def _summary_card(self, label, value, color_name):
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(3)
        label_widget = QLabel(label)
        label_widget.setObjectName("eyebrow")
        layout.addWidget(label_widget)
        value_widget = QLabel(value)
        value_widget.setObjectName(f"summaryValue_{color_name}")
        layout.addWidget(value_widget)
        return card, value_widget

    def _settings_group(self, title):
        group = self._card()
        layout = QGridLayout(group)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(12)
        group._settings_layout = layout
        group._settings_row = 1
        heading = self._section_heading(title)
        layout.addWidget(heading, 0, 0, 1, 2)
        return group

    def _settings_row(self, group, label, widget):
        layout = group._settings_layout
        row = group._settings_row
        group._settings_row += 1
        if label:
            label_widget = QLabel(label)
            label_widget.setObjectName("muted")
            layout.addWidget(label_widget, row, 0)
            layout.addWidget(widget, row, 1)
        else:
            layout.addWidget(widget, row, 0, 1, 2)

    def _combo(self, values, current):
        combo = QComboBox()
        combo.addItems(values)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(self._sync_plan_labels)
        return combo

    def _line_edit(self, value):
        entry = QLineEdit(value)
        entry.textChanged.connect(self._sync_plan_labels)
        return entry

    # ------------------------------------------------------------------
    # Page and settings behavior
    # ------------------------------------------------------------------
    def _show_page(self, page):
        meta = {
            "session": ("Live Session", "Capture a clean study session with focus, posture, and context."),
            "review": ("Review", "Browse sessions, inspect trends, and re-render recaps."),
            "coach": ("Coach", "Generate grounded local feedback from your study data."),
            "settings": ("Settings", "Tune capture, privacy, rendering, and session goals."),
        }
        for key, button in self.nav_buttons.items():
            button.setChecked(key == page)
        self.pages.setCurrentIndex(self.page_indices[page])
        self.page_title.setText(meta[page][0])
        self.page_subtitle.setText(meta[page][1])
        if page == "review":
            self._load_analytics()

    def capture_rate_text(self):
        return self.capture_rate_combo.currentText() if hasattr(self, "capture_rate_combo") else "1 fps"

    def privacy_combo_text(self):
        return self.privacy_combo.currentText() if hasattr(self, "privacy_combo") else "Off"

    def render_summary_text(self):
        if not hasattr(self, "aspect_combo"):
            return "Source 16:9 / Glass"
        return f"{self.aspect_combo.currentText()} / {self.hud_combo.currentText()}"

    def _goal_label(self):
        if not hasattr(self, "goal_input"):
            return "Deep work session (60 min)"
        goal = self.goal_input.text().strip()
        minutes = self.goal_minutes_combo.currentText().strip()
        if goal and minutes:
            return f"{goal} ({minutes} min)"
        return goal

    def _capture_interval_from_ui(self):
        value = self.capture_rate_combo.currentText()
        if value == "2 fps":
            return 0.5
        if value == "0.5 fps":
            return 2.0
        return 1.0

    def _auto_pause_seconds_from_ui(self):
        try:
            return int(self.auto_pause_combo.currentText().split()[0])
        except (ValueError, IndexError):
            return 120

    def _collect_render_options(self):
        return {
            "target_video_duration": self.duration_slider.value(),
            "hud_style": self.hud_combo.currentText(),
            "aspect_ratio": self.aspect_combo.currentText(),
            "privacy_mode": self.privacy_combo.currentText(),
            "goal_label": self._goal_label(),
        }

    def _sync_plan_labels(self):
        if not hasattr(self, "plan_goal_value"):
            return
        self.session_goal_label.setText(self._goal_label() or "No goal set")
        self.plan_goal_value.setText(self._goal_label() or "No goal")
        self.plan_privacy_value.setText(self.privacy_combo_text())
        self.plan_capture_value.setText(self.capture_rate_text())
        self.plan_render_value.setText(self.render_summary_text())

    def _apply_settings(self):
        self._sync_whoop_config_from_ui()
        self.capture_engine.configure(
            capture_interval=self._capture_interval_from_ui(),
            auto_pause_seconds=self._auto_pause_seconds_from_ui(),
            privacy_mode=self.privacy_combo.currentText(),
            session_goal=self._goal_label(),
            capture_resolution=self.resolution_combo.currentText(),
        )
        self._sync_plan_labels()
        self.btn_apply_settings.setText("Applied")
        QTimer.singleShot(1500, lambda: self.btn_apply_settings.setText("Apply Settings"))

    def _sync_whoop_config_from_ui(self):
        if not hasattr(self, "whoop_client_id_input"):
            return
        self.capture_engine.whoop_client.configure(
            client_id=self.whoop_client_id_input.text(),
            client_secret=self.whoop_client_secret_input.text(),
            redirect_uri=self.whoop_redirect_input.text(),
        )

    def _connect_whoop(self):
        self._sync_whoop_config_from_ui()
        if not self.capture_engine.whoop_client.uses_local_http_callback():
            try:
                self.capture_engine.whoop_client.begin_manual_authorization()
                self.whoop_settings_status.setText("Paste the final redirect URL or code, then Finish with Code")
                self.btn_whoop_connect.setText("Opened WHOOP")
                QTimer.singleShot(1800, lambda: self.btn_whoop_connect.setText("Connect WHOOP"))
            except Exception as e:
                self.whoop_settings_status.setText(f"Connection failed: {e}")
            return

        self.btn_whoop_connect.setEnabled(False)
        self.btn_whoop_connect.setText("Waiting for WHOOP")
        self.whoop_settings_status.setText("Browser authorization pending")
        threading.Thread(target=self._connect_whoop_worker, daemon=True).start()

    def _connect_whoop_worker(self):
        try:
            self.capture_engine.whoop_client.connect_via_browser()
            metrics = self.capture_engine.refresh_whoop_now(force=False)
        except Exception as e:
            metrics = self.capture_engine.whoop_client.cached_metrics()
            metrics = dict(metrics) if isinstance(metrics, dict) else {}
            metrics["status"] = f"Connection failed: {e}"
        self.signals.whoop_status.emit(metrics)

    def _finish_whoop_code(self):
        self._sync_whoop_config_from_ui()
        code_or_url = self.whoop_code_input.text().strip()
        self.btn_whoop_finish.setEnabled(False)
        self.btn_whoop_finish.setText("Finishing")
        self.whoop_settings_status.setText("Exchanging WHOOP code")
        threading.Thread(target=self._finish_whoop_code_worker, args=(code_or_url,), daemon=True).start()

    def _finish_whoop_code_worker(self, code_or_url):
        try:
            metrics = self.capture_engine.whoop_client.finish_manual_authorization(code_or_url)
            with self.capture_engine.whoop_lock:
                self.capture_engine.current_whoop = metrics
        except Exception as e:
            metrics = self.capture_engine.whoop_client.cached_metrics()
            metrics = dict(metrics) if isinstance(metrics, dict) else {}
            metrics["status"] = f"Connection failed: {e}"
        self.signals.whoop_status.emit(metrics)

    def _refresh_whoop(self, force=False):
        self._sync_whoop_config_from_ui()
        if hasattr(self, "btn_whoop_refresh"):
            self.btn_whoop_refresh.setEnabled(False)
            self.btn_whoop_refresh.setText("Refreshing")
        threading.Thread(target=self._refresh_whoop_worker, args=(force,), daemon=True).start()

    def _refresh_whoop_worker(self, force):
        metrics = self.capture_engine.refresh_whoop_now(force=force)
        self.signals.whoop_status.emit(metrics)

    def _whoop_status_changed(self, metrics):
        if hasattr(self, "btn_whoop_connect"):
            self.btn_whoop_connect.setEnabled(True)
            self.btn_whoop_connect.setText("Connect WHOOP")
        if hasattr(self, "btn_whoop_finish"):
            self.btn_whoop_finish.setEnabled(True)
            self.btn_whoop_finish.setText("Finish with Code")
        if hasattr(self, "btn_whoop_refresh"):
            self.btn_whoop_refresh.setEnabled(True)
            self.btn_whoop_refresh.setText("Refresh WHOOP")
        status = metrics.get("status", "Not connected") if isinstance(metrics, dict) else "Not connected"
        if hasattr(self, "whoop_settings_status"):
            self.whoop_settings_status.setText(status)
        self._update_whoop_widgets(metrics)

    def _on_render_duration_change(self, value):
        stepped = int(round(value / 5) * 5)
        if stepped != value:
            self.duration_slider.blockSignals(True)
            self.duration_slider.setValue(stepped)
            self.duration_slider.blockSignals(False)
        self.duration_label.setText(f"Final video length: {stepped} sec")

    # ------------------------------------------------------------------
    # Camera and status
    # ------------------------------------------------------------------
    def _open_preview_camera(self):
        self.preview_open_generation += 1
        generation = self.preview_open_generation
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        self.quality_label.setText("Opening camera")
        threading.Thread(
            target=self._open_preview_camera_worker,
            args=(generation,),
            daemon=True,
        ).start()

    def _open_preview_camera_worker(self, generation):
        with self.preview_open_lock:
            if self.preview_closing or generation != self.preview_open_generation:
                return
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not cap.isOpened() or actual_w <= 0 or actual_h <= 0:
                cap.release()
                cap = None
                actual_w = 0
                actual_h = 0
            if self.preview_closing or generation != self.preview_open_generation:
                if cap:
                    cap.release()
                return
            self.signals.preview_ready.emit(cap, actual_w, actual_h, generation)

    def _preview_camera_ready(self, cap, actual_w, actual_h, generation):
        if self.preview_closing or generation != self.preview_open_generation or self.capture_engine.is_running:
            if cap:
                cap.release()
            return
        if not cap:
            self.quality_label.setText("Camera unavailable")
            return
        self.preview_cap = cap
        capture_resolution = self.resolution_combo.currentText() if hasattr(self, "resolution_combo") else "2560x1440"
        self.quality_label.setText(f"Preview {actual_w}x{actual_h} | Capture {capture_resolution}")

    def update_preview(self):
        if not self.is_previewing or not self.preview_cap or self.isMinimized():
            return
        ret, frame = self.preview_cap.read()
        if not ret:
            return
        target_w = max(1, self.video_label.width())
        target_h = max(1, self.video_label.height())
        h, w = frame.shape[:2]
        scale = min(target_w / max(1, w), target_h / max(1, h), 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def update_status(self):
        snapshot = self.capture_engine.get_live_snapshot()
        backend = snapshot["backend"]
        backend_text = backend.get("selected", "Pending")
        if backend.get("benchmark_ms"):
            backend_text += f" ({backend['benchmark_ms']:.0f} ms)"

        focused = bool(snapshot["focused"])
        self.lbl_live_focus.setText("Focused" if focused else "Distracted")
        self.lbl_live_focus.setProperty("state", "good" if focused else "warn")
        self.lbl_live_focus.style().unpolish(self.lbl_live_focus)
        self.lbl_live_focus.style().polish(self.lbl_live_focus)
        self.lbl_live_posture.setText(str(snapshot["posture"]).title())
        self.lbl_live_activity.setText(self._compact(snapshot["activity"], 18))
        self.lbl_live_backend.setText(self._compact(backend_text, 18))
        music = self.capture_engine.current_song_title or "Not playing"
        self.lbl_live_music.setText(self._compact(music, 42))
        self.lbl_live_frames.setText(f"{snapshot['frames_captured']} frames | {snapshot['inference_ms']:.0f} ms AI")
        capture_error = snapshot.get("capture_error", "")
        fatigue = snapshot.get("fatigue", {})
        self._update_fatigue_widgets(fatigue)
        self._update_whoop_widgets(snapshot.get("whoop", {}))

        if self.capture_engine.is_running:
            # Focus Shield
            now = time.time()
            if not self.capture_engine.is_paused and (now - self.last_shield_time > 60):
                self.last_shield_time = now
                threading.Thread(target=self._run_focus_shield, daemon=True).start()

            elapsed = time.time() - self.capture_engine.start_time
            time_str = self._format_hms(elapsed)
            focus_time_str = self._format_hms(snapshot.get("total_focus_time_sec", 0))
            break_time_str = self._format_hms(snapshot.get("total_break_time_sec", 0))
            
            self.side_time.setText(time_str)
            self.side_focus.setText(f"Focus: {focus_time_str}")
            self.side_break.setText(f"Break: {break_time_str}")
            self._update_sidebar_readiness(fatigue, focused)
            if self.capture_engine.is_paused:
                remaining = int(fatigue.get("break_remaining_seconds", 0) or 0)
                if remaining > 0:
                    pause_label = f"Break {self._format_break_duration(remaining)} left"
                else:
                    pause_label = "Paused"
                self.side_detail.setText(pause_label)
                self.header_status.setText(pause_label)
                self.session_status_label.setText(pause_label)
                self.btn_pause.setEnabled(False)
                self.btn_resume.setEnabled(True)
                if self.mini:
                    self.mini.toggle_button.setText("Resume")
            else:
                if fatigue.get("break_recommended"):
                    break_label = fatigue.get("recommended_break_label", "break")
                    self.side_detail.setText(f"Break recommended: {break_label}")
                    self.header_status.setText("Break recommended")
                    self.session_status_label.setText("Break recommended")
                else:
                    self.side_detail.setText("Recording")
                    self.header_status.setText("Recording")
                    self.session_status_label.setText("Recording")
                self.btn_pause.setEnabled(True)
                self.btn_resume.setEnabled(False)
                if self.mini:
                    self.mini.toggle_button.setText("Pause")
            if self.mini:
                self.mini.time_label.setText(time_str)
                self.mini.focus_label.setText(f"Focus: {focus_time_str}  |  Break: {break_time_str}")
                mini_detail = fatigue.get("state") if fatigue.get("break_recommended") else music
                self.mini.detail_label.setText(self._compact(mini_detail, 30))
        elif not self.is_rendering:
            self.side_time.setText("00:00:00")
            self.side_focus.setText("Focus: 00:00:00")
            self.side_break.setText("Break: 00:00:00")
            self.side_detail.setText(self._compact(capture_error, 28) if capture_error else "Idle")
            self.side_readiness.setText("Readiness: Standby")
            self.side_readiness.setStyleSheet(f"color: {self.colors['accent']}; font-weight: bold;")
            self.header_status.setText("Camera error" if capture_error else "Ready")
            self.session_status_label.setText(self._compact(capture_error, 48) if capture_error else "Ready to capture")
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(False)
            self.btn_end.setEnabled(False)

    def _format_hms(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _compact(self, value, limit):
        text = "" if value is None else str(value)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _update_fatigue_widgets(self, fatigue):
        fatigue = fatigue if isinstance(fatigue, dict) else {}
        available = bool(fatigue.get("available"))
        score = int(fatigue.get("score", 0) or 0)
        state = fatigue.get("state", "Waiting")
        expression = fatigue.get("expression", "")
        if available:
            self.lbl_live_fatigue.setText(self._compact(f"{state} {score}%", 18))
        else:
            self.lbl_live_fatigue.setText(self._compact(state, 18))
        fatigue_state = "good"
        if fatigue.get("break_recommended") or score >= 72:
            fatigue_state = "danger"
        elif score >= 42:
            fatigue_state = "warn"
        self.lbl_live_fatigue.setProperty("state", fatigue_state)
        self.lbl_live_fatigue.style().unpolish(self.lbl_live_fatigue)
        self.lbl_live_fatigue.style().polish(self.lbl_live_fatigue)

        remaining = int(fatigue.get("break_remaining_seconds", 0) or 0)
        if remaining > 0:
            break_text = self._format_break_duration(remaining)
            break_state = "danger"
        elif fatigue.get("break_recommended"):
            break_text = fatigue.get("recommended_break_label", "Take break")
            break_state = "danger"
        else:
            break_text = expression if available and expression != "Alert" else "No break"
            break_state = "good" if break_text == "No break" else "warn"
        self.lbl_live_break.setText(self._compact(break_text, 18))
        self.lbl_live_break.setProperty("state", break_state)
        self.lbl_live_break.style().unpolish(self.lbl_live_break)
        self.lbl_live_break.style().polish(self.lbl_live_break)

    def _format_break_duration(self, seconds):
        seconds = max(0, int(seconds or 0))
        minutes, secs = divmod(seconds, 60)
        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}h {minutes:02d}m"
        return f"{minutes:02d}:{secs:02d}"

    def _update_sidebar_readiness(self, fatigue, focused):
        fatigue = fatigue if isinstance(fatigue, dict) else {}
        score = int(fatigue.get("score", 0) or 0)
        if fatigue.get("break_recommended"):
            text = f"Readiness: Break {fatigue.get('recommended_break_label', '')}".strip()
            color = self.colors["danger"]
        elif score >= 58:
            text = "Readiness: Tired"
            color = self.colors["warning"]
        elif score >= 34 or not focused:
            text = "Readiness: Watch"
            color = self.colors["warning"]
        else:
            text = "Readiness: Fresh"
            color = self.colors["success"]
        self.side_readiness.setText(text)
        self.side_readiness.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _update_whoop_widgets(self, metrics):
        if not hasattr(self, "whoop_recovery_ring"):
            return
        metrics = metrics if isinstance(metrics, dict) else {}
        recovery = metrics.get("recovery")
        sleep = metrics.get("sleep")
        strain = metrics.get("strain")
        status = metrics.get("status", "Not connected")
        synced = self._whoop_sync_label(metrics.get("last_synced_at", 0))
        self.whoop_live_status.setText(f"{status} {synced}".strip())
        if hasattr(self, "whoop_settings_status") and not self.capture_engine.is_running:
            self.whoop_settings_status.setText(status)

        if recovery:
            value = recovery.get("value", 0)
            self.whoop_recovery_ring.set_metric(f"{int(round(value))}%", value / 100.0, self._whoop_recovery_color(value), recovery.get("zone", "").upper())
        else:
            self.whoop_recovery_ring.set_metric("--", 0.0, "#4B5563", "")

        if sleep:
            value = sleep.get("value", 0)
            caption = self._hours_caption(sleep.get("duration_hours"))
            self.whoop_sleep_ring.set_metric(f"{int(round(value))}%", value / 100.0, "#2DD4FF", caption)
        else:
            self.whoop_sleep_ring.set_metric("--", 0.0, "#4B5563", "")

        if strain:
            value = float(strain.get("value", 0))
            self.whoop_strain_ring.set_metric(f"{value:.1f}", value / 21.0, "#D7FF35", "OF 21")
        else:
            self.whoop_strain_ring.set_metric("--", 0.0, "#4B5563", "")

    def _whoop_recovery_color(self, value):
        if value >= 67:
            return "#18D66B"
        if value >= 34:
            return "#FFD233"
        return "#FF4B55"

    def _whoop_sync_label(self, timestamp):
        try:
            timestamp = float(timestamp or 0)
        except (TypeError, ValueError):
            timestamp = 0
        if timestamp <= 0:
            return ""
        minutes = int((time.time() - timestamp) / 60)
        if minutes <= 0:
            return "now"
        if minutes < 60:
            return f"{minutes}m ago"
        return f"{minutes // 60}h ago"

    def _hours_caption(self, hours):
        try:
            hours = float(hours)
        except (TypeError, ValueError):
            return ""
        if hours <= 0:
            return ""
        whole = int(hours)
        minutes = int(round((hours - whole) * 60))
        return f"{whole}H {minutes:02d}M"

    # ------------------------------------------------------------------
    # Session controls and rendering
    # ------------------------------------------------------------------
    def start_session(self):
        self._apply_settings()
        self.is_previewing = False
        self.preview_open_generation += 1
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Recording in progress")

        self.current_session_dir = self.capture_engine.start_session()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_end.setEnabled(True)
        self.render_progress.setValue(0)
        self.render_progress_label.setText("Idle")

    def pause_session(self):
        self.capture_engine.pause_session()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)

    def resume_session(self):
        self.capture_engine.resume_session()
        self.btn_resume.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def end_session(self):
        self.capture_engine.end_session()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_end.setEnabled(False)
        self.is_rendering = True
        self.header_status.setText("Rendering")
        self.session_status_label.setText("Rendering video")
        self.render_progress.setValue(2)
        self.render_progress_label.setText("Loading session")
        self._start_render(self.current_session_dir, reopen_preview=True)

    def _start_render(self, session_dir, reopen_preview):
        if not session_dir:
            return
        self.is_rendering = True
        self.btn_start.setEnabled(False)
        
        achievements = []
        try:
            event_file = os.path.join(session_dir, "events.json")
            if os.path.exists(event_file):
                current_summary = self._summarize_session(event_file)
                if current_summary:
                    current_focus = current_summary["focused_frames"]
                    max_past_focus = 0
                    for s in getattr(self, "session_summaries", []):
                        if s["session_dir"] != session_dir and s["focused_frames"] > max_past_focus:
                            max_past_focus = s["focused_frames"]
                    if current_focus > max_past_focus and current_focus > 60:
                        achievements.append("Personal Record! Top Focus Time")
        except Exception:
            pass
            
        options = self._collect_render_options()
        options["achievements"] = achievements
        
        thread = threading.Thread(target=self._render_video, args=(session_dir, reopen_preview, options), daemon=True)
        thread.start()

    def _render_video(self, session_dir, reopen_preview, options=None):
        try:
            if options is None:
                options = self._collect_render_options()
            renderer = Renderer(
                session_dir,
                options=options,
                progress_callback=lambda stage, fraction: self.signals.render_progress.emit(stage, fraction),
            )
            out_path = renderer.render()
        except Exception as e:
            self.signals.render_progress.emit(f"Render failed: {e}", 0.0)
            out_path = None
        self.signals.render_complete.emit(out_path, reopen_preview)

    def _render_existing_session(self, session_dir):
        if self.capture_engine.is_running:
            self.header_status.setText("Finish session first")
            return
        self._show_page("session")
        self.session_status_label.setText(f"Rendering {os.path.basename(session_dir)}")
        self.render_progress.setValue(2)
        self.render_progress_label.setText("Loading session")
        self._start_render(session_dir, reopen_preview=False)

    def _set_render_progress(self, stage, fraction):
        self.render_progress.setValue(int(fraction * 100))
        self.render_progress_label.setText(f"{stage} ({int(fraction * 100)}%)")

    def _render_complete(self, out_path, reopen_preview):
        self.is_rendering = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_end.setEnabled(False)

        if out_path:
            self.last_render_path = out_path
            self.header_status.setText("Render complete")
            self.session_status_label.setText(f"Saved: {os.path.basename(out_path)}")
            self.render_progress.setValue(100)
            self.render_progress_label.setText("Render complete")
        else:
            self.header_status.setText("Render failed")
            self.session_status_label.setText("Render failed or empty session")
            self.render_progress_label.setText("Render failed")

        if reopen_preview:
            self._open_preview_camera()
            self.is_previewing = True
        self._load_analytics()

    # ------------------------------------------------------------------
    # Mini player
    # ------------------------------------------------------------------
    def open_mini_player(self):
        if self.mini:
            self.mini.show()
            return
        self.mini = MiniPlayer(self.colors)
        self.mini.setStyleSheet(self._stylesheet())
        self.mini.toggle_requested.connect(self._mini_toggle_pause)
        self.mini.expand_requested.connect(self.close_mini_player)
        self.mini.show()
        self.hide()

    def _mini_toggle_pause(self):
        if self.capture_engine.is_paused:
            self.resume_session()
        else:
            self.pause_session()

    def close_mini_player(self):
        if self.mini:
            self.mini.close()
            self.mini = None
        self.show()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def _load_analytics(self):
        sessions_dir = os.path.join(os.getcwd(), "sessions")
        if not os.path.exists(sessions_dir):
            self._populate_session_browser([])
            return

        event_files = sorted(glob.glob(os.path.join(sessions_dir, "*", "events.json")))
        total_seconds = 0
        total_focused_frames = 0
        total_frames = 0
        posture_stats = {"good": 0, "slouching": 0, "leaning": 0, "unknown": 0}
        music_stats = {}
        activity_stats = {"Laptop": 0, "Book": 0, "Distracted (Phone)": 0}
        most_recent_timeline = []
        summaries = []

        active_cache_keys = set()
        for event_file in event_files:
            try:
                stat = os.stat(event_file)
                cache_key = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
            active_cache_keys.add(event_file)
            cached = self.analytics_summary_cache.get(event_file)
            if cached and cached[0] == cache_key:
                summary = cached[1]
            else:
                summary = self._summarize_session(event_file)
                self.analytics_summary_cache[event_file] = (cache_key, summary)
            if not summary:
                continue
            summaries.append(summary)
            total_seconds += summary["duration_seconds"]
            total_focused_frames += summary["focused_frames"]
            total_frames += summary["total_frames"]

            for activity, count in summary["activity_counts"].items():
                activity_stats[activity] = activity_stats.get(activity, 0) + count
            for posture, count in summary["posture_counts"].items():
                posture_stats[posture] = posture_stats.get(posture, 0) + count
            for song, stats in summary["music_stats"].items():
                music_stats.setdefault(song, [0, 0])
                music_stats[song][0] += stats[0]
                music_stats[song][1] += stats[1]
            most_recent_timeline = summary["timeline"]

        self.analytics_summary_cache = {
            path: value for path, value in self.analytics_summary_cache.items() if path in active_cache_keys
        }

        h, rem = divmod(int(total_seconds), 3600)
        m = rem // 60
        focus_score = int((total_focused_frames / max(1, total_frames)) * 100) if total_frames else 0
        self.lbl_total_time.setText(f"{h}h {m}m")
        self.lbl_avg_focus.setText(f"{focus_score}%")

        top_song = "None"
        best_score = 0
        for song, stats in music_stats.items():
            if stats[1] >= 60:
                score = (stats[0] / stats[1]) * 100
                if score > best_score:
                    best_score = score
                    top_song = song

        top_activity = "None"
        if activity_stats.get("Laptop", 0) > activity_stats.get("Book", 0):
            top_activity = "Laptop"
        elif activity_stats.get("Book", 0) > activity_stats.get("Laptop", 0):
            top_activity = "Book"
        self.lbl_top_activity.setText(top_activity)
        self.lbl_top_music.setText(self._compact(top_song, 18))

        self.export_stats = {
            "time": f"{h}h {m}m",
            "focus": f"{focus_score}%",
            "song": top_song,
            "posture": max(posture_stats, key=posture_stats.get) if posture_stats else "unknown",
            "sessions": summaries,
        }
        self.session_summaries = summaries
        self._draw_chart(most_recent_timeline, activity_stats)
        self._populate_session_browser(summaries)

    def _summarize_session(self, event_file):
        try:
            with open(event_file, "r") as f:
                data = json.load(f)
        except Exception:
            return None

        events = data.get("events", [])
        last_ts = None
        duration = 0
        total_frames = 0
        focused_frames = 0
        activity_counts = {}
        posture_counts = {}
        music_stats = {}
        timeline = []
        session_start_time = data.get("start_time", 0)

        for event in events:
            event_type = event.get("type")
            if event_type == "frame":
                ts = event.get("timestamp", 0)
                if last_ts:
                    duration += max(0, ts - last_ts)
                last_ts = ts
                total_frames += 1
                focused = event.get("focused", True)
                if focused:
                    focused_frames += 1
                activity = event.get("activity", "None")
                posture = event.get("posture", "unknown")
                music = event.get("music", "")
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
                posture_counts[posture] = posture_counts.get(posture, 0) + 1
                if music:
                    music_stats.setdefault(music, [0, 0])
                    music_stats[music][1] += 1
                    if focused:
                        music_stats[music][0] += 1
                if session_start_time:
                    timeline.append(((ts - session_start_time) / 60.0, 1 if focused else 0))
            elif event_type == "pause":
                last_ts = None
            elif event_type == "resume":
                last_ts = event.get("timestamp")

        focus_score = int((focused_frames / max(1, total_frames)) * 100)
        session_dir = os.path.dirname(event_file)
        return {
            "session_dir": session_dir,
            "name": os.path.basename(session_dir),
            "goal": data.get("goal", ""),
            "duration_seconds": duration,
            "focus_score": focus_score,
            "focused_frames": focused_frames,
            "total_frames": total_frames,
            "activity_counts": activity_counts,
            "posture_counts": posture_counts,
            "music_stats": music_stats,
            "timeline": timeline,
            "has_video": os.path.exists(os.path.join(session_dir, "frames.mp4")),
            "has_legacy_frames": os.path.exists(os.path.join(session_dir, "frames")),
        }

    def _populate_session_browser(self, summaries):
        self._clear_layout(self.sessions_layout)
        if not summaries:
            empty = QLabel("No sessions yet.")
            empty.setObjectName("muted")
            self.sessions_layout.addWidget(empty)
            self.sessions_layout.addStretch(1)
            return

        for summary in reversed(summaries[-10:]):
            item = QFrame()
            item.setObjectName("sessionItem")
            row = QHBoxLayout(item)
            row.setContentsMargins(14, 12, 14, 12)
            row.setSpacing(12)

            text = QVBoxLayout()
            title = QLabel(f"{summary['name']}  |  {self._format_duration(summary['duration_seconds'])}")
            title.setObjectName("sessionTitle")
            text.addWidget(title)
            posture = max(summary["posture_counts"], key=summary["posture_counts"].get) if summary["posture_counts"] else "unknown"
            storage = "video" if summary["has_video"] else "legacy frames"
            detail = QLabel(f"Focus {summary['focus_score']}% | Posture {posture} | {summary['total_frames']} frames | {storage}")
            detail.setObjectName("muted")
            text.addWidget(detail)
            row.addLayout(text, 1)

            render = self._button("Render", "primary")
            render.clicked.connect(lambda checked=False, path=summary["session_dir"]: self._render_existing_session(path))
            row.addWidget(render)
            self.sessions_layout.addWidget(item)
        self.sessions_layout.addStretch(1)

    def _draw_chart(self, timeline_data, activity_stats):
        self._clear_layout(self.chart_mount_layout)
        if not timeline_data:
            empty = QLabel("No timeline data yet.")
            empty.setObjectName("muted")
            empty.setAlignment(Qt.AlignCenter)
            self.chart_mount_layout.addWidget(empty)
            return

        times = [d[0] for d in timeline_data]
        focus_vals = [d[1] for d in timeline_data]

        fig = Figure(figsize=(9, 3.6), dpi=100, facecolor=self.colors["surface"])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_facecolor(self.colors["surface"])
        ax2.set_facecolor(self.colors["surface"])

        ax1.fill_between(times, 0, focus_vals, color=self.colors["success"], alpha=0.55)
        ax1.fill_between(times, focus_vals, 1, color=self.colors["danger"], alpha=0.36)
        ax1.set_title("Recent Session Focus", color=self.colors["text"], fontsize=12, pad=12)
        ax1.set_xlabel("Minutes", color=self.colors["muted"])
        ax1.set_yticks([])
        ax1.tick_params(colors=self.colors["muted"])
        for spine in ax1.spines.values():
            spine.set_visible(False)

        labels = []
        sizes = []
        colors = [self.colors["accent"], "#FBBF24", self.colors["danger"]]
        for label in ["Laptop", "Book", "Distracted (Phone)"]:
            if activity_stats.get(label, 0) > 0:
                labels.append(label.replace("Distracted ", ""))
                sizes.append(activity_stats[label])
        if sizes:
            ax2.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct="%1.0f%%", startangle=90, textprops={"color": "w", "fontsize": 9})
            ax2.set_title("Tools", color=self.colors["text"], fontsize=12)
        else:
            ax2.axis("off")

        fig.tight_layout()
        self.canvas_widget = FigureCanvas(fig)
        self.chart_mount_layout.addWidget(self.canvas_widget)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _format_duration(self, seconds):
        m, _ = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m"
        return f"{m}m"

    # ------------------------------------------------------------------
    # Coach and exports
    # ------------------------------------------------------------------
    def _export_summary_card(self):
        w, h = 1080, 1920
        img = Image.new("RGB", (w, h), color=self.colors["bg"])
        draw = ImageDraw.Draw(img)
        for y in range(h):
            shade = int(15 + (y / h) * 28)
            draw.line([(0, y), (w, y)], fill=(shade, shade + 3, shade + 8))

        try:
            font_title = ImageFont.truetype("segoeuib.ttf", 86)
            font_label = ImageFont.truetype("segoeui.ttf", 42)
            font_value = ImageFont.truetype("segoeuib.ttf", 84)
        except IOError:
            font_title = font_label = font_value = ImageFont.load_default()

        draw.text((92, 170), "Study Recap", font=font_title, fill=self.colors["text"])
        draw.text((92, 286), self._goal_label() or "Session summary", font=font_label, fill=self.colors["muted"])

        stats = [
            ("TOTAL STUDY TIME", self.export_stats.get("time", "0h 0m"), self.colors["accent"]),
            ("AVERAGE FOCUS", self.export_stats.get("focus", "0%"), self.colors["success"]),
            ("POSTURE MODE", self.export_stats.get("posture", "unknown"), "#FBBF24"),
            ("TOP FOCUS TRACK", self.export_stats.get("song", "None"), "#C4B5FD"),
        ]
        y_off = 520
        for label, value, color in stats:
            draw.text((92, y_off), label, font=font_label, fill=self.colors["muted"])
            draw.text((92, y_off + 58), self._compact(value, 20), font=font_value, fill=color)
            y_off += 290

        out_path = os.path.join(os.path.expanduser("~"), "Desktop", "summary_card.png")
        img.save(out_path)
        self.btn_export.setText("Saved to Desktop")
        QTimer.singleShot(2500, lambda: self.btn_export.setText("Export Summary Card"))

    def _generate_coach_insight(self):
        self.btn_coach.setEnabled(False)
        self.btn_coach.setText("Preparing model...")
        self.txt_coach.setText("Preparing AI coach. First run may download the Phi-3 model.\n")
        threading.Thread(target=self._run_llm_thread, daemon=True).start()

    def _run_llm_thread(self):
        try:
            from huggingface_hub import snapshot_download
            import onnxruntime_genai as og

            model_id = "microsoft/Phi-3-mini-4k-instruct-onnx"
            local_dir = os.path.join(os.getcwd(), "models", "phi3")
            self.signals.coach_append.emit("Checking local models...\n")

            model_path = snapshot_download(
                repo_id=model_id,
                allow_patterns=["directml/directml-int4-awq-block-128/*"],
                local_dir=local_dir,
            )
            actual_model_path = os.path.join(model_path, "directml", "directml-int4-awq-block-128")

            config_path = os.path.join(actual_model_path, "genai_config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

            original_providers = config["model"]["decoder"]["session_options"].get("provider_options", [])
            config["model"]["decoder"]["session_options"]["provider_options"] = []
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            self.signals.coach_append.emit("Loading Phi-3 on CPU while vision keeps the accelerator.\n")
            try:
                model = og.Model(actual_model_path)
            finally:
                config["model"]["decoder"]["session_options"]["provider_options"] = original_providers
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)

            tokenizer = og.Tokenizer(model)
            stats = self.export_stats
            prompt = "<|system|>\nYou are a supportive, concise study coach. Give one paragraph and one concrete next-session experiment.<|end|>\n"
            prompt += (
                "<|user|>\n"
                f"Goal: {self._goal_label() or 'None'}\n"
                f"Total Study Time: {stats.get('time', '0h 0m')}\n"
                f"Average Focus: {stats.get('focus', '0%')}\n"
                f"Dominant Posture: {stats.get('posture', 'unknown')}\n"
                f"Top Song: {stats.get('song', 'None')}\n"
                f"Primary Tool: {self.lbl_top_activity.text()}\n"
                "What should I improve next session?<|end|>\n<|assistant|>\n"
            )

            input_tokens = tokenizer.encode(prompt)
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=420)
            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)
            stream = tokenizer.create_stream()

            self.signals.coach_clear.emit()
            while not generator.is_done():
                generator.generate_next_token()
                decoded = stream.decode(generator.get_next_tokens()[0])
                if "<|end|>" in decoded:
                    break
                self.signals.coach_append.emit(decoded)

            del generator, tokenizer, model, params
            gc.collect()
            self.signals.coach_done.emit("Generate Coach Insight")
        except Exception as e:
            self.signals.coach_append.emit(f"\nError: {e}")
            self.signals.coach_done.emit("Retry Coach Insight")
            gc.collect()

    def _coach_clear(self):
        self.txt_coach.clear()

    def _coach_append(self, text):
        self.txt_coach.moveCursor(QTextCursor.End)
        self.txt_coach.insertPlainText(text)
        self.txt_coach.moveCursor(QTextCursor.End)

    def _coach_done(self, text):
        self.btn_coach.setText(text)
        self.btn_coach.setEnabled(True)

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _stylesheet(self):
        c = self.colors
        return f"""
        QWidget#root, QFrame#main {{
            background: {c['bg']};
            color: {c['text']};
            font-family: "Segoe UI";
        }}
        QFrame#sidebar {{
            background: {c['sidebar']};
            border-right: 1px solid {c['border']};
        }}
        QLabel#brand {{
            color: {c['text']};
            font-size: 24px;
            font-weight: 700;
        }}
        QLabel#sidebarText, QLabel#pageSubtitle, QLabel#muted, QLabel#eyebrow {{
            color: {c['muted']};
        }}
        QLabel#pageTitle {{
            color: {c['text']};
            font-size: 28px;
            font-weight: 700;
        }}
        QLabel#sectionTitle {{
            color: {c['text']};
            font-size: 17px;
            font-weight: 700;
        }}
        QLabel#statusPill {{
            background: {c['surface']};
            color: {c['success']};
            border: 1px solid {c['border']};
            border-radius: 16px;
            font-weight: 700;
            padding: 6px 10px;
        }}
        QFrame#card {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: 12px;
        }}
        QFrame#metricBox, QFrame#sessionItem {{
            background: {c['surface2']};
            border: 1px solid {c['border']};
            border-radius: 9px;
        }}
        QLabel#metricValue, QLabel#planValue, QLabel#sessionTitle {{
            color: {c['text']};
            font-size: 15px;
            font-weight: 700;
        }}
        QLabel#metricValue[state="good"] {{
            color: {c['success']};
        }}
        QLabel#metricValue[state="warn"] {{
            color: {c['warning']};
        }}
        QLabel#metricValue[state="danger"] {{
            color: {c['danger']};
        }}
        QLabel#whoopStatus {{
            color: {c['muted']};
            font-size: 12px;
            font-weight: 700;
        }}
        QLabel#sideTime {{
            color: {c['text']};
            font-size: 25px;
            font-weight: 700;
        }}
        QLabel#summaryValue_accent, QLabel#summaryValue_success,
        QLabel#summaryValue_warning, QLabel#summaryValue_purple {{
            font-size: 24px;
            font-weight: 700;
        }}
        QLabel#summaryValue_accent {{ color: {c['accent']}; }}
        QLabel#summaryValue_success {{ color: {c['success']}; }}
        QLabel#summaryValue_warning {{ color: {c['warning']}; }}
        QLabel#summaryValue_purple {{ color: #C4B5FD; }}
        QLabel#video {{
            background: #07080A;
            color: {c['muted']};
            border-radius: 10px;
            font-size: 18px;
            font-weight: 700;
        }}
        QPushButton {{
            border: 0;
            border-radius: 8px;
            padding: 9px 14px;
            color: {c['text']};
            font-weight: 700;
            background: {c['surface2']};
        }}
        QPushButton:hover {{
            background: #2B3343;
        }}
        QPushButton:disabled {{
            color: {c['subtle']};
            background: #1A1E27;
        }}
        QPushButton#navButton {{
            background: transparent;
            color: {c['muted']};
            text-align: left;
            padding: 11px 12px;
        }}
        QPushButton#navButton:checked {{
            background: {c['surface2']};
            color: {c['text']};
        }}
        QPushButton#primaryButton {{
            background: {c['accent']};
            color: #041016;
        }}
        QPushButton#primaryButton:hover {{
            background: {c['accent2']};
        }}
        QPushButton#successButton {{
            background: {c['success']};
            color: #06110A;
        }}
        QPushButton#dangerButton {{
            background: {c['danger']};
            color: white;
        }}
        QPushButton#secondaryButton {{
            background: {c['surface2']};
            color: {c['text']};
        }}
        QPushButton#outlineButton {{
            background: transparent;
            border: 1px solid {c['border']};
            color: {c['text']};
        }}
        QLineEdit, QComboBox, QTextEdit {{
            background: {c['surface2']};
            border: 1px solid {c['border']};
            border-radius: 8px;
            color: {c['text']};
            padding: 8px;
            selection-background-color: {c['accent2']};
        }}
        QTextEdit#coachText {{
            background: {c['surface']};
            border-radius: 12px;
            font-size: 15px;
        }}
        QProgressBar {{
            background: #11141B;
            border: 1px solid {c['border']};
            border-radius: 6px;
            height: 12px;
        }}
        QProgressBar::chunk {{
            background: {c['accent']};
            border-radius: 5px;
        }}
        QScrollArea {{
            border: 0;
            background: transparent;
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: #11141B;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
            background: {c['accent']};
        }}
        QWidget#mini {{
            background: {c['surface']};
            border: 1px solid {c['border']};
        }}
        QLabel#miniTime {{
            color: {c['text']};
            font-size: 28px;
            font-weight: 700;
        }}
        """

    def _run_focus_shield(self):
        apps_to_kill = ["Discord.exe", "WhatsApp.exe", "Telegram.exe"]
        for app in apps_to_kill:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", app],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
            except OSError:
                pass

    def closeEvent(self, event):
        self.preview_closing = True
        self.preview_open_generation += 1
        self.is_previewing = False
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        if self.capture_engine.is_running:
            self.capture_engine.end_session()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimelapseApp()
    window.show()
    sys.exit(app.exec())
