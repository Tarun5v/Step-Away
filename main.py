"""Step-Away: a webcam-powered presence guard for your Mac.

Watches for you at the desk, locks the machine when you step away,
and nudges you to stretch after long focus sessions.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
CHECK_INTERVAL_SECONDS = 3
PRESENCE_WINDOW_SIZE = 5
PRESENCE_MIN_HITS = 2
MIN_DETECTION_CONFIDENCE = 0.4
DEMO_STRETCH_SECONDS = 40
DEMO_EYE_REST_SECONDS = 10
DRAIN_GRABS = 6
PREVIEW_TARGET_FPS = 60
FPS_SMOOTHING = 0.9
DEBUG_FRAME_DIR = ".debug_frames"
DEBUG_SNAPSHOT_LIMIT = 40
MOTION_RESIZE_WIDTH = 160
PREVIEW_WINDOW_TITLE = "Step-Away camera"

FIRST_RUN_CHECKLIST = """
First-time setup checklist:
  1. Camera        - System Settings > Privacy & Security > Camera:
                     enable your terminal app (presence detection)
  2. Accessibility - System Settings > Privacy & Security > Accessibility:
                     enable your terminal app (auto-lock keystroke)
  3. Notifications - allow them when macOS first asks (stretch reminders)
Re-check anytime with: python main.py --doctor
"""


class PresenceVote:
    """Rolling-window vote over raw face readings to smooth flicker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hits = deque(maxlen=PRESENCE_WINDOW_SIZE)

    def record(self, raw_present: bool) -> None:
        with self._lock:
            self._hits.append(raw_present)

    @property
    def present(self) -> bool:
        with self._lock:
            return sum(self._hits) >= PRESENCE_MIN_HITS

    @property
    def tally(self):
        with self._lock:
            return sum(self._hits), len(self._hits)
ABSENCE_LOCK_SECONDS = 10
ABSENCE_WARNING_SECONDS = 5
LOCK_COMMAND_TIMEOUT_SECONDS = 5
LOCK_RETRY_SECONDS = 30
FOCUS_REMINDER_MINUTES = 50
EYE_RULE_MINUTES = 20
BREAK_THRESHOLD_SECONDS = 300
BREAK_OVERLAY_SECONDS = 30
EYE_REST_OVERLAY_SECONDS = 20
DAY_DEFAULTS = {
    "focus_seconds": 0,
    "reminders": 0,
    "breaks": 0,
    "eye_rests": 0,
    "water_glasses": 0,
}
STATS_FILE = "step_away_stats.json"
SAVE_INTERVAL_SECONDS = 60


def open_camera(camera_index: int):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. "
            "Make sure it exists and is not in use by another app."
        )
    return capture


class PresenceMonitor(threading.Thread):
    """Background thread that keeps track of whether anyone sits at the desk."""

    def __init__(self, camera_index: int = CAMERA_INDEX, debug: bool = False):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.debug = debug
        self.vote = PresenceVote()
        self._running = True
        self.error = None
        self._prev_motion_frame = None
        self._snapshot_index = 0

    @property
    def face_present(self) -> bool:
        return self.vote.present

    def stop(self) -> None:
        self._running = False

    def _open_camera(self):
        return open_camera(self.camera_index)

    def _detect_face(self, detector, frame):
        """Return (present, pixel-space boxes) for faces in the frame."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = detector.process(image)
        detections = result.detections or []
        height, width = frame.shape[:2]
        boxes = []
        for detection in detections:
            box = detection.location_data.relative_bounding_box
            boxes.append(
                (
                    int(box.xmin * width),
                    int(box.ymin * height),
                    int(box.width * width),
                    int(box.height * height),
                )
            )
        return bool(detections), boxes

    def run(self) -> None:
        try:
            capture = self._open_camera()
        except RuntimeError as error:
            self.error = str(error)
            self._running = False
            return

        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE
        ) as detector:
            failed_reads = 0
            while self._running:
                grabbed, frame = capture.read()
                if not grabbed:
                    failed_reads += 1
                    if failed_reads >= 5:
                        self.error = (
                            "Camera stopped delivering frames. "
                            "Check macOS camera permission for your terminal app."
                        )
                        break
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue

                failed_reads = 0
                frame = self._read_fresh_frame(capture)
                if frame is None:
                    failed_reads += 1
                    if failed_reads >= 5:
                        self.error = (
                            "Camera stopped delivering frames. "
                            "Check macOS camera permission for your terminal app."
                        )
                        break
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue

                raw_present, boxes = self._detect_face(detector, frame)
                motion = self._motion_score(frame)
                self.vote.record(raw_present)
                hits, total = self.vote.tally

                if self.debug:
                    stamp = time.strftime("%H:%M:%S")
                    verdict = "HIT " if raw_present else "MISS"
                    print(
                        f"[{stamp}] check {verdict} window {hits}/"
                        f"{total} motion {motion:.1f}"
                    )
                    self._save_debug_snapshot(frame, raw_present)

                time.sleep(CHECK_INTERVAL_SECONDS)

        capture.release()

    def _read_fresh_frame(self, capture):
        """Drain buffered frames so analysis always uses the live image."""
        frame = None
        grabbed = False
        for _ in range(DRAIN_GRABS):
            grabbed, frame = capture.read()
            if not grabbed:
                return None
        return frame if grabbed else None

    def _motion_score(self, frame) -> float:
        """Mean pixel change since the previous check (0 = identical image)."""
        height = max(1, int(frame.shape[0] * MOTION_RESIZE_WIDTH / frame.shape[1]))
        small = cv2.cvtColor(
            cv2.resize(frame, (MOTION_RESIZE_WIDTH, height)), cv2.COLOR_BGR2GRAY
        ).astype("float32")
        if self._prev_motion_frame is None:
            self._prev_motion_frame = small
            return 0.0
        score = float(abs(small - self._prev_motion_frame).mean())
        self._prev_motion_frame = small
        return score

    def _save_debug_snapshot(self, frame, hit: bool) -> None:
        out_dir = Path(DEBUG_FRAME_DIR)
        out_dir.mkdir(exist_ok=True)
        self._snapshot_index = self._snapshot_index % DEBUG_SNAPSHOT_LIMIT + 1
        tag = "hit" if hit else "miss"
        stamp = time.strftime("%H%M%S")
        cv2.imwrite(str(out_dir / f"check_{stamp}_{tag}_{self._snapshot_index}.jpg"), frame)


def lock_screen() -> bool:
    """Lock the Mac with the Cmd+Ctrl+Q shortcut via AppleScript."""
    script = (
        'tell application "System Events" to '
        'keystroke "q" using {command down, control down}'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=LOCK_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        print("Automatic locking needs macOS (osascript was not found).")
        return False
    except subprocess.TimeoutExpired:
        print("The screen lock command timed out.")
        return False
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        reason = f" ({detail[0]})" if detail else ""
        print(
            "Could not lock the screen automatically"
            f"{reason}. Grant Accessibility permission to your terminal app in "
            "System Settings > Privacy & Security > Accessibility."
        )
        return False
    return True


def send_stretch_notification(message: str = None) -> None:
    """Post a desktop reminder to stand up and stretch."""
    title = "Step-Away"
    if message is None:
        message = "You have been focused for a while. Time to stretch and hydrate!"
    try:
        from plyer import notification

        notification.notify(title=title, message=message, timeout=10)
    except Exception:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True)


_overlay_ui_cache = {}


def _load_overlay_ui():
    """Define the native overlay classes once; pyobjc forbids re-registering."""
    if "ui" in _overlay_ui_cache:
        return _overlay_ui_cache["ui"]

    import AppKit
    import objc
    from Foundation import NSObject

    class OverlayWindow(AppKit.NSWindow):
        def canBecomeKeyWindow(self):
            return True

        def sendEvent_(self, event):
            if event.type() in (
                AppKit.NSKeyDown,
                AppKit.NSLeftMouseDown,
                AppKit.NSRightMouseDown,
            ):
                AppKit.NSApplication.sharedApplication().stopModal()
                return
            return objc.super(OverlayWindow, self).sendEvent_(event)

    class Coordinator(NSObject):
        def tick_(self, timer):
            AppKit.NSApplication.sharedApplication().stopModal()

    _overlay_ui_cache["ui"] = (OverlayWindow, Coordinator)
    return _overlay_ui_cache["ui"]


def show_break_overlay(title: str, subtitle: str, duration_seconds: float) -> None:
    """Take over the screen with a blurred, Apple-style break card.

    Falls back to a regular notification when the native UI stack is
    unavailable. Blocks until the user clicks/presses a key or the
    duration elapses.
    """
    try:
        import AppKit
        from Foundation import NSMakeRect, NSTimer, NSRunLoop

        OverlayWindow, Coordinator = _load_overlay_ui()
        NSMakeRect  # keep linters honest about the conditional import
    except Exception:
        send_stretch_notification(message=f"{title}. {subtitle}")
        return

    app = AppKit.NSApplication.sharedApplication()
    screen = AppKit.NSScreen.mainScreen().frame()

    window = OverlayWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, screen.size.width, screen.size.height),
        AppKit.NSBorderlessWindowMask,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    window.setLevel_(AppKit.NSScreenSaverWindowLevel)
    window.setOpaque_(False)
    window.setBackgroundColor_(AppKit.NSColor.clearColor())
    window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
    )

    blur = AppKit.NSVisualEffectView.alloc().initWithFrame_(
        NSMakeRect(0, 0, screen.size.width, screen.size.height)
    )
    blur.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
    blur.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
    blur.setState_(AppKit.NSVisualEffectStateActive)

    def centered_label(y_ratio, height, text, font_size, alpha):
        field = AppKit.NSTextField.alloc().initWithFrame_(
            NSMakeRect(0, screen.size.height * y_ratio, screen.size.width, height)
        )
        field.setStringValue_(text)
        field.setFont_(AppKit.NSFont.boldSystemFontOfSize_(font_size))
        field.setAlignment_(AppKit.NSTextAlignmentCenter)
        color = AppKit.NSColor.whiteColor()
        field.setTextColor_(
            color.colorWithAlphaComponent_(alpha)
        )
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setEditable_(False)
        field.setSelectable_(False)
        return field

    content = window.contentView()
    content.addSubview_(blur)
    content.addSubview_(centered_label(0.58, 70, title, 48, 1.0))
    content.addSubview_(centered_label(0.50, 40, subtitle, 24, 0.85))
    content.addSubview_(centered_label(0.10, 30, "Press any key or click to continue", 15, 0.55))

    coordinator = Coordinator.alloc().init()
    from Foundation import NSTimer, NSRunLoop

    timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
        duration_seconds, coordinator, "tick:", None, False
    )
    for mode in (
        AppKit.NSModalPanelRunLoopMode,
        AppKit.NSDefaultRunLoopMode,
    ):
        NSRunLoop.currentRunLoop().addTimer_forMode_(timer, mode)

    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)
    try:
        app.runModalForWindow_(window)
    finally:
        timer.invalidate()
        window.orderOut_(None)


def format_duration(total_seconds: int) -> str:
    minutes, seconds = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


class StatsStore:
    """Persists daily focus time, reminders and breaks to a local JSON file."""

    def __init__(self, path: str = STATS_FILE):
        self.path = Path(path)
        self.days = {}
        self.meta = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
            self.days = payload.get("days", {})
            self.meta = payload.get("meta", {})
        except (json.JSONDecodeError, OSError):
            print("Stats file was unreadable; starting a fresh one.")
            self.days = {}

    def save(self) -> None:
        try:
            payload = {"days": self.days, "meta": self.meta}
            self.path.write_text(json.dumps(payload, indent=2))
        except OSError as error:
            print(f"Could not save stats: {error}")

    @property
    def setup_acknowledged(self) -> bool:
        return bool(self.meta.get("setup_shown"))

    def mark_setup_shown(self) -> None:
        self.meta["setup_shown"] = True

    def _today(self) -> dict:
        today = self.days.setdefault(date.today().isoformat(), {})
        for key, default in DAY_DEFAULTS.items():
            today.setdefault(key, default)
        return today

    def add_focus(self, seconds: float) -> None:
        self._today()["focus_seconds"] += int(seconds)

    def record_reminder(self) -> None:
        self._today()["reminders"] += 1

    def record_eye_rest(self) -> None:
        self._today()["eye_rests"] += 1

    def record_water(self) -> int:
        today = self._today()
        today["water_glasses"] += 1
        return today["water_glasses"]

    def water_today(self) -> int:
        return self._today().get("water_glasses", 0)

    def record_break(self) -> None:
        self._today()["breaks"] += 1

    def current_streak(self) -> int:
        active = {
            day for day, stats in self.days.items() if stats.get("focus_seconds", 0) > 0
        }
        day = date.today()
        if day.isoformat() not in active:
            day -= timedelta(days=1)
            if day.isoformat() not in active:
                return 0
        streak = 0
        while day.isoformat() in active:
            streak += 1
            day -= timedelta(days=1)
        return streak

    def summary_report(self) -> str:
        today = self._today()
        streak = self.current_streak()
        lines = [
            "Today's report",
            f"  Focus time:     {format_duration(today['focus_seconds'])}",
            f"  Stretch nudges: {today['reminders']}",
            f"  Eye-rest nudges: {today['eye_rests']}",
            f"  Water logged:   {today.get('water_glasses', 0)} glass(es)",
            f"  Breaks taken:   {today['breaks']}",
            f"  Current streak: {streak} day{'s' if streak != 1 else ''}",
        ]
        return "\n".join(lines)


def run_doctor() -> int:
    """Verify every permission and dependency the app relies on."""
    import numpy as np

    print("Step-Away doctor")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  OpenCV  : {cv2.__version__}")
    print(f"  MediaPipe: {mp.__version__}")

    failures = 0

    print("- Camera         : ", end="", flush=True)
    try:
        capture = open_camera(CAMERA_INDEX)
        grabbed, _ = capture.read()
        capture.release()
        if grabbed:
            print("OK (frames are arriving)")
        else:
            print(
                "FAILED - grant Camera permission to your terminal app in "
                "System Settings > Privacy & Security > Camera"
            )
            failures += 1
    except RuntimeError as error:
        print(f"FAILED - {error}")
        failures += 1

    print("- Face detection : ", end="", flush=True)
    try:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE
        ) as detector:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            detector.process(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
        print("OK (model loads and runs)")
    except Exception as error:
        print(f"FAILED - {error}")
        failures += 1

    print("- Accessibility  : ", end="", flush=True)
    probe = subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "System Events" to get name of first application process',
        ],
        capture_output=True,
        text=True,
        timeout=LOCK_COMMAND_TIMEOUT_SECONDS,
    )
    error_text = probe.stderr.lower()
    if probe.returncode == 0:
        print("OK (System Events reachable)")
    elif "assistive" in error_text or "not allowed" in error_text or "1002" in error_text:
        print(
            "MISSING - enable your terminal app in "
            "System Settings > Privacy & Security > Accessibility"
        )
        failures += 1
    else:
        print(f"UNKNOWN ({probe.stderr.strip()}) - will diagnose again at lock time")

    print("- Notifications  : ", end="", flush=True)
    send_stretch_notification(message="Doctor test - if you can read this, nudges work!")
    print("sent a test alert (check your notification centre)")

    print()
    if failures:
        print(f"{failures} check(s) need attention.")
        return 1
    print("All checks passed. Step-Away is ready to guard your desk.")
    return 0


class StepAwayApp:
    """Ties presence monitoring into security and wellness behaviours."""

    def __init__(self, debug: bool = False, show_preview: bool = False, demo: bool = False):
        self.monitor = PresenceMonitor(debug=debug)
        self.show_preview = show_preview
        self.presence_announced = False
        self.stats = StatsStore()
        self.absent_since = None
        self.locked = False
        self.warned = False
        self.focus_start = None
        self.eye_rest_start = None
        self.break_counted = True
        self.next_lock_attempt = 0.0
        self.last_tick = time.time()
        self.last_save = time.time()

        if demo:
            self.stretch_interval = DEMO_STRETCH_SECONDS
            self.eye_rest_interval = DEMO_EYE_REST_SECONDS
            self.break_overlay_duration = 8
            self.eye_overlay_duration = 5
        else:
            self.stretch_interval = FOCUS_REMINDER_MINUTES * 60
            self.eye_rest_interval = EYE_RULE_MINUTES * 60
            self.break_overlay_duration = BREAK_OVERLAY_SECONDS
            self.eye_overlay_duration = EYE_REST_OVERLAY_SECONDS

    def _update_security(self, now: float) -> None:
        if self.monitor.face_present:
            if self.locked:
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] Welcome back! Security guard re-armed.")
            self.absent_since = None
            self.locked = False
            self.warned = False
            self.next_lock_attempt = 0.0
            return

        if self.absent_since is None:
            self.absent_since = now
            return

        absence = now - self.absent_since
        if not self.locked:
            if absence >= ABSENCE_LOCK_SECONDS and now >= self.next_lock_attempt:
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] No one at the desk - locking the screen.")
                if lock_screen():
                    self.locked = True
                else:
                    print(f"Lock failed - retrying every {LOCK_RETRY_SECONDS}s.")
                    self.next_lock_attempt = now + LOCK_RETRY_SECONDS
            elif absence >= ABSENCE_WARNING_SECONDS and not self.warned:
                remaining = round(ABSENCE_LOCK_SECONDS - absence)
                print(f"Desk empty - locking in {remaining}s. Hop back in view to cancel.")
                self.warned = True

    def _update_wellness(self, now: float, delta: float) -> None:
        if self.monitor.face_present:
            self.stats.add_focus(delta)

            if self.focus_start is None:
                self.focus_start = now
            elif now - self.focus_start >= self.stretch_interval:
                self.stats.record_reminder()
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] Focus limit reached - showing break screen.")
                show_break_overlay(
                    "Time for a break",
                    "You've reached your focus limit. Stand up, stretch, breathe.",
                    self.break_overlay_duration,
                )
                self.focus_start = time.time()

            if self.eye_rest_start is None:
                self.eye_rest_start = now
            elif now - self.eye_rest_start >= self.eye_rest_interval:
                self.stats.record_eye_rest()
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] 20-20-20 - showing eye-rest screen.")
                show_break_overlay(
                    "20-20-20",
                    "Look at something 6 metres away for 20 seconds.",
                    self.eye_overlay_duration,
                )
                self.eye_rest_start = time.time()

            return

        if self.absent_since is None:
            return
        if now - self.absent_since >= BREAK_THRESHOLD_SECONDS and not self.break_counted:
            self.stats.record_break()
            self.break_counted = True
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] Break logged. Nice step away!")

    def _reset_focus_session(self) -> None:
        if self.focus_start is not None and self.monitor.face_present:
            return
        if not self.monitor.face_present:
            self.focus_start = None
            self.eye_rest_start = None
            self.break_counted = False

    def _announce(self, present: bool) -> None:
        if present == self.presence_announced:
            return
        stamp = time.strftime("%H:%M:%S")
        state = "arrived" if present else "left"
        print(f"[{stamp}] You {state} the desk.")
        self.presence_announced = present

    def _draw_preview_frame(self, frame, boxes, fps: float) -> None:
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 80), 2)
        state = "watching you" if self.presence_announced else "desk empty"
        cv2.putText(
            frame,
            f"{fps:4.0f} FPS",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 200, 80),
            2,
        )
        cv2.putText(
            frame,
            f"Step-Away: {state}",
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 200, 80),
            2,
        )
        cv2.putText(
            frame,
            f"water: {self.stats.water_today()} glasses (w to log)",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 80),
            2,
        )
        cv2.imshow(PREVIEW_WINDOW_TITLE, frame)

    def _tick_common(self, now: float, delta: float) -> None:
        self._announce(self.monitor.vote.present)
        self._update_security(now)
        self._update_wellness(now, delta)
        self._reset_focus_session()

    def run(self) -> int:
        print("Step-Away is starting up. Press Ctrl+C to quit.")
        if self.stretch_interval == DEMO_STRETCH_SECONDS:
            print(
                f"Demo mode: break screen every {DEMO_STRETCH_SECONDS}s, "
                f"eye rest every {DEMO_EYE_REST_SECONDS}s."
            )
        print(f"Streak so far: {self.stats.current_streak()} day(s).")
        if not self.stats.setup_acknowledged:
            print(FIRST_RUN_CHECKLIST)
            self.stats.mark_setup_shown()
        self.stats.save()

        if self.show_preview:
            exit_code = self._run_preview()
        else:
            exit_code = self._run_headless()

        if self.show_preview:
            cv2.destroyAllWindows()
        else:
            self.monitor.stop()
            self.monitor.join(timeout=CHECK_INTERVAL_SECONDS * 2)

            if self.monitor.error:
                print(f"Stopped: {self.monitor.error}")
                exit_code = 1
            elif self.monitor.is_alive():
                print("Warning: camera thread did not shut down cleanly.")
                exit_code = 1

        self.stats.save()
        print()
        print(self.stats.summary_report())
        print("Step-Away closed. See you soon!")
        return exit_code

    def _run_headless(self) -> int:
        self.monitor.start()
        try:
            while self.monitor.is_alive():
                now = time.time()
                delta = min(now - self.last_tick, CHECK_INTERVAL_SECONDS * 2)
                self.last_tick = now

                self._tick_common(now, delta)

                if now - self.last_save >= SAVE_INTERVAL_SECONDS:
                    self.stats.save()
                    self.last_save = now

                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return 0

    def _run_preview(self) -> int:
        try:
            capture = open_camera(CAMERA_INDEX)
        except RuntimeError as error:
            print(f"Stopped: {error}")
            return 1

        frame_interval = 1.0 / PREVIEW_TARGET_FPS
        fps = 0.0
        boxes = []
        last_detection = 0.0
        try:
            with mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE
            ) as detector:
                while True:
                    frame_start = time.perf_counter()
                    now = time.time()
                    delta = min(now - self.last_tick, CHECK_INTERVAL_SECONDS * 2)
                    self.last_tick = now

                    grabbed, frame = capture.read()
                    if not grabbed:
                        print("Camera stopped delivering frames.")
                        break

                    if now - last_detection >= CHECK_INTERVAL_SECONDS:
                        raw_present, boxes = self.monitor._detect_face(detector, frame)
                        self.monitor.vote.record(raw_present)
                        last_detection = now
                        if self.monitor.debug:
                            hits, total = self.monitor.vote.tally
                            stamp = time.strftime("%H:%M:%S")
                            verdict = "HIT " if raw_present else "MISS"
                            print(f"[{stamp}] check {verdict} window {hits}/{total}")

                    self._tick_common(now, delta)
                    self._draw_preview_frame(frame, boxes, fps)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("w"):
                        glasses = self.stats.record_water()
                        stamp = time.strftime("%H:%M:%S")
                        print(f"[{stamp}] Logged a glass of water ({glasses} today).")
                    elif key in (ord("q"), 0x1B):
                        break

                    if now - self.last_save >= SAVE_INTERVAL_SECONDS:
                        self.stats.save()
                        self.last_save = now

                    elapsed = time.perf_counter() - frame_start
                    instant = 1.0 / max(elapsed, 1e-6)
                    fps = instant if fps == 0 else FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * instant
                    remaining = frame_interval - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        capture.release()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-Away presence guard")
    parser.add_argument(
        "--debug", action="store_true", help="print every webcam check result"
    )
    parser.add_argument(
        "--preview", action="store_true", help="show a live camera window with face boxes"
    )
    parser.add_argument(
        "--doctor", action="store_true", help="verify permissions and dependencies"
    )
    parser.add_argument(
        "--demo", action="store_true", help="shrink wellness timers to seconds for testing"
    )
    args = parser.parse_args()
    if args.doctor:
        sys.exit(run_doctor())
    sys.exit(StepAwayApp(debug=args.debug, show_preview=args.preview, demo=args.demo).run())
