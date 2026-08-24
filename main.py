"""Step-Away: a webcam-powered presence guard for your Mac.

Watches for you at the desk, locks the machine when you step away,
and nudges you to stretch after long focus sessions.
"""

import argparse
import ctypes
import json
import os
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
DEMO_OVERLAY_COOLDOWN_SECONDS = 15

OVERLAY_CLOSE = "__overlay_close__"
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


class QuietStderr:
    """Temporarily silence native C++ log noise (MediaPipe/absl) on fd 2."""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr.flush()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        return False


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

    @property
    def settled(self) -> bool:
        """True once the window holds enough samples to trust the verdict."""
        with self._lock:
            return len(self._hits) >= PRESENCE_MIN_HITS


ABSENCE_LOCK_SECONDS = 10
ABSENCE_WARNING_SECONDS = 5
LOCK_COMMAND_TIMEOUT_SECONDS = 5
LOCK_RETRY_SECONDS = 30
INPUT_ACTIVITY_GRACE_SECONDS = 5.0
STARTUP_GRACE_SECONDS = 60
TERMINAL_APPS = {
    "Terminal",
    "iTerm2",
    "Warp",
    "Hyper",
    "Alacritty",
    "kitty",
    "Ghostty",
    "WezTerm",
}
FOCUS_REMINDER_MINUTES = 50
EYE_RULE_MINUTES = 20
DAILY_WATER_GOAL = 8
DRINK_GESTURE_SECONDS = 1.5
DRINK_DEDUP_SECONDS = 4 * 60
HAND_SCORE_MIN = 0.5
BREAK_THRESHOLD_SECONDS = 300
BREAK_AWAY_SECONDS = 60
DEMO_BREAK_AWAY_SECONDS = 5
EYE_REST_AWAY_SECONDS = 20
DEMO_EYE_REST_AWAY_SECONDS = 4
OVERLAY_COOLDOWN_SECONDS = 60
OVERLAY_FOCUS_GUARD_SECONDS = 0.25
EYE_REST_POLL_SECONDS = 1.0
EYE_REST_NAG_STEP_SECONDS = 4.0
STRETCH_NAG_STEP_SECONDS = 6.0
STRETCH_NAG_MESSAGES = [
    "Stand up.",
    "Reach for the ceiling. Big stretch.",
    "Walk to the far side of the room.",
    "Roll those shoulders.",
    "Touch your toes if you can.",
    "One more lap - maybe refill your water?",
]
EYE_REST_NAG_MESSAGES = [
    "I can still see you watching\u2026",
    "The screen is fine without you. Look away.",
    "Your eyes are begging for this break. Take it.",
    "Staring contest accepted. You will lose.",
]
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
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    @property
    def face_present(self) -> bool:
        return self.vote.present

    @property
    def latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

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

        with QuietStderr(), mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE
        ) as detector:
            failed_reads = 0
            while self._running:
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

                failed_reads = 0
                with QuietStderr():
                    raw_present, boxes = self._detect_face(detector, frame)
                motion = self._motion_score(frame)
                self.vote.record(raw_present)
                with self._frame_lock:
                    self._latest_frame = frame.copy()
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


def _input_idle_seconds():
    """Seconds since the last mouse/keyboard event, or None when unknown.

    Uses CoreGraphics session state - read-only, no permissions needed.
    """
    try:
        core_graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        core_graphics.CGEventSourceSecondsSinceLastEventType.restype = ctypes.c_double
        core_graphics.CGEventSourceSecondsSinceLastEventType.argtypes = [
            ctypes.c_int32,
            ctypes.c_uint32,
        ]
        return core_graphics.CGEventSourceSecondsSinceLastEventType(0, 0xFFFFFFFF)
    except Exception:
        return None


def _frontmost_app_name():
    """Name of the app currently receiving keystrokes, or None."""
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return app.localizedName()
    except Exception:
        return None


def _lock_with_security_framework() -> bool:
    """Lock the session via Security.framework - no permissions, no focus.

    SACLockScreenImmediate locks the screen directly regardless of which
    app is frontmost, unlike synthetic keystrokes that can be swallowed
    by the focused window (e.g. the camera preview).
    """
    try:
        security = ctypes.CDLL(
            "/System/Library/Frameworks/Security.framework/Security"
        )
        security.SACLockScreenImmediate()
        return True
    except Exception:
        return False


def _lock_by_display_sleep() -> bool:
    """Last-resort lock: sleep the display (password gate still applies)."""
    try:
        subprocess.run(
            ["pmset", "displaysleepnow"],
            capture_output=True,
            timeout=LOCK_COMMAND_TIMEOUT_SECONDS,
        )
        print(
            "Locked by display sleep instead - set 'Require password "
            "immediately' in System Settings > Lock Screen for full effect."
        )
        return True
    except Exception:
        return False


def lock_screen() -> bool:
    """Lock the Mac using the strongest mechanism available right now."""
    if _lock_with_security_framework():
        return True

    # A synthetic Cmd+Ctrl+Q only reaches the system shortcut when a real
    # terminal is frontmost; any other focused app swallows it silently.
    if _frontmost_app_name() in TERMINAL_APPS:
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
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            print(f"Keystroke lock unavailable ({error.__class__.__name__}).")

    print("Falling back to display-sleep lock.")
    return _lock_by_display_sleep()


_face_mesh_lock = threading.Lock()
_face_mesh = None


def _get_face_mesh():
    """Lazily create the shared FaceMesh used for eye-rest supervision."""
    global _face_mesh
    with _face_mesh_lock:
        if _face_mesh is None:
            import mediapipe as mp

            with QuietStderr():
                _face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=0.5,
                )
    return _face_mesh


def gaze_on_screen(frame) -> bool:
    """True when a roughly frontal face with open eyes fills the frame.

    Used during the eye-rest screen to notice the user still watching.
    Returns False when no face, a turned-away face or closed eyes are seen.
    """
    import cv2 as _cv2

    detector_mesh = _get_face_mesh()
    image = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
    with QuietStderr():
        result = detector_mesh.process(image)
    if not result.multi_face_landmarks:
        return False

    landmarks = result.multi_face_landmarks[0].landmark
    height, width = frame.shape[:2]

    def pixel(index):
        point = landmarks[index]
        return point.x * width, point.y * height

    nose_x, _ = pixel(1)
    left_x, _ = pixel(234)
    right_x, _ = pixel(454)
    face_width = max(right_x - left_x, 1e-6)
    centre_offset = abs((nose_x - left_x) / face_width - 0.5)
    if centre_offset > 0.18:

        return False

    def eye_ratio(top, bottom, inner, outer):
        vertical = abs(pixel(top)[1] - pixel(bottom)[1])
        horizontal = max(abs(pixel(inner)[0] - pixel(outer)[0]), 1e-6)
        return vertical / horizontal

    left_eye = eye_ratio(159, 145, 33, 133)
    right_eye = eye_ratio(386, 374, 362, 263)
    return (left_eye + right_eye) / 2 > 0.15


def face_in_frame(frame) -> bool:
    """True when any face is visible - i.e. the user is still at the desk.

    Used during the break screen: a face means the user has not stood up.
    """
    import cv2 as _cv2

    detector_mesh = _get_face_mesh()
    image = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
    with QuietStderr():
        result = detector_mesh.process(image)
    return bool(result.multi_face_landmarks)


_hands_lock = threading.Lock()
_hands = None


def _get_hands():
    """Lazily create the shared hand tracker used for drink detection."""
    global _hands
    with _hands_lock:
        if _hands is None:
            import mediapipe as mp

            with QuietStderr():
                _hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.6,
                )
    return _hands


def _drink_metrics(hands_result, face_result) -> dict:
    """Judge whether a detected hand is a drink at the mouth.

    Following HydroVisor's approach, the hand's posture is irrelevant -
    fingers curl around cups and wrists rotate, so judging geometry makes
    natural drinking unrecognisable. Only space matters: a confident
    detection whose centre sits in the mouth zone (below the nose, within
    about one face-width of the mouth, down to just under chin level)
    counts. That keeps face-phantoms above the nose and desk hands below
    the zone out, while any natural grip inside it passes.
    """
    hand_lists = getattr(hands_result, "multi_hand_landmarks", None)
    if not hand_lists:
        return {"hand": False}

    score = 0.0
    handedness = getattr(hands_result, "multi_handedness", None)
    if handedness:
        score = handedness[0].classification[0].score

    face_lists = getattr(face_result, "multi_face_landmarks", None)
    if not face_lists:
        return {"hand": True, "score": round(score, 2), "drinking": False,
                "why": "no_face"}

    face = face_lists[0].landmark
    mouth_x = (face[61].x + face[291].x) / 2
    mouth_y = (face[61].y + face[291].y) / 2
    nose_y = face[1].y
    chin_y = face[152].y
    span = abs(face[454].x - face[234].x)
    height = max(abs(chin_y - face[10].y), 1e-4)

    points = hand_lists[0].landmark
    centre_x = sum(point.x for point in points) / len(points)
    centre_y = sum(point.y for point in points) / len(points)

    checks = {
        "score": score >= HAND_SCORE_MIN,
        "level": centre_y > nose_y - 0.05 * height,
        "near_mouth": (
            abs(centre_x - mouth_x) < 1.1 * span
            and mouth_y - 0.05 * height < centre_y < chin_y + 0.9 * height
        ),
    }
    return {
        "hand": True,
        "score": round(score, 2),
        "y": round(centre_y, 2),
        **checks,
        "drinking": all(checks.values()),
    }


class DrinkDetector:
    """Detects drinking as a verified hand-at-the-mouth gesture.

    Hand detections are cross-checked against face landmarks so phantom
    detections on faces or desks never count. The gesture must hold for
    DRINK_GESTURE_SECONDS before consider() fires once and re-arms;
    progress decays gently between sips instead of resetting.
    """

    def __init__(self):
        self.sustained = 0.0
        self.last_obs = {}

    def consider(self, frame, seconds: float) -> bool:
        import cv2 as _cv2

        try:
            image = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            with QuietStderr():
                face_result = _get_face_mesh().process(image)
                hands_result = _get_hands().process(image)
        except Exception as error:
            self.sustained = 0.0
            self.last_obs = {"error": f"{type(error).__name__}: {error}"[:140]}
            return False

        self.last_obs = _drink_metrics(hands_result, face_result)
        drinking = bool(self.last_obs.get("drinking"))

        if drinking:
            self.sustained += min(max(seconds, 0.0), 2.0)
        else:
            # Micro-dips between sips should not wipe accumulated progress;
            # only sustained non-drinking decays it.
            self.sustained = max(
                0.0,
                self.sustained - 0.7 * min(max(seconds, 0.0), 2.0),
            )
        if self.sustained >= DRINK_GESTURE_SECONDS:
            self.sustained = 0.0
            return True
        return False


class StretchNagger:
    """Holds the break screen open until the user is actually up and about.

    While a face is visible the user is considered still seated and the
    instructions escalate. The screen releases only after the face stays
    out of frame for the required away time.
    """

    def __init__(
        self,
        frame_provider,
        step_seconds: float = STRETCH_NAG_STEP_SECONDS,
        required_away_seconds: float = BREAK_AWAY_SECONDS,
    ):
        self.frame_provider = frame_provider
        self.step_seconds = step_seconds
        self.required_away_seconds = required_away_seconds
        self.seated_seconds = 0.0
        self.away_seconds = 0.0
        self.level = 0

    def next_message(self):
        try:
            frame = self.frame_provider()
        except Exception:
            frame = None

        seated = frame is not None and face_in_frame(frame)

        if seated:
            self.seated_seconds += EYE_REST_POLL_SECONDS
            self.away_seconds = 0.0
            reached = int(self.seated_seconds // self.step_seconds)
            if reached > self.level:
                self.level = reached
                index = min(self.level - 1, len(STRETCH_NAG_MESSAGES) - 1)
                return STRETCH_NAG_MESSAGES[index]
            return None

        self.away_seconds += EYE_REST_POLL_SECONDS
        self.seated_seconds = max(0.0, self.seated_seconds - EYE_REST_POLL_SECONDS)
        if self.away_seconds >= self.required_away_seconds:
            return OVERLAY_CLOSE
        return None


class GazeNagger:
    """Holds the eye-rest screen open and escalates while the user watches.

    The screen only releases after the user keeps their eyes off the
    screen (or out of frame) for the required away time.
    """

    def __init__(
        self,
        frame_provider,
        step_seconds: float = EYE_REST_NAG_STEP_SECONDS,
        required_away_seconds: float = EYE_REST_AWAY_SECONDS,
    ):
        self.frame_provider = frame_provider
        self.step_seconds = step_seconds
        self.required_away_seconds = required_away_seconds
        self.watch_seconds = 0.0
        self.away_seconds = 0.0
        self.level = 0

    def next_message(self):
        try:
            frame = self.frame_provider()
        except Exception:
            frame = None

        watching = frame is not None and gaze_on_screen(frame)

        if watching:
            self.watch_seconds += EYE_REST_POLL_SECONDS
            self.away_seconds = 0.0
            reached = int(self.watch_seconds // self.step_seconds)
            if reached > self.level:
                self.level = reached
                index = min(self.level - 1, len(EYE_REST_NAG_MESSAGES) - 1)
                return EYE_REST_NAG_MESSAGES[index]
            return None

        self.away_seconds += EYE_REST_POLL_SECONDS
        self.watch_seconds = max(0.0, self.watch_seconds - EYE_REST_POLL_SECONDS)
        if self.away_seconds >= self.required_away_seconds:
            return OVERLAY_CLOSE
        return None


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

        def guardFocus_(self, timer):
            try:
                self.guard_focus(timer)
            except Exception:
                pass

        def refreshTitle_(self, timer):
            try:
                replacement = self.on_title_refresh()
            except Exception:
                return
            if replacement == OVERLAY_CLOSE:
                AppKit.NSApplication.sharedApplication().stopModal()
            elif replacement:
                self.title_field.setStringValue_(replacement)

    _overlay_ui_cache["ui"] = (OverlayWindow, Coordinator)
    return _overlay_ui_cache["ui"]


def show_break_overlay(
    title: str,
    subtitle: str,
    duration_seconds,
    on_title_refresh=None,
) -> None:
    """Take over the screen with a blurred, Apple-style break card.

    Falls back to a regular notification when the native UI stack is
    unavailable. Blocks until the user clicks/presses a key or the
    duration elapses; pass duration_seconds=None for screens that
    release themselves via on_title_refresh returning OVERLAY_CLOSE.
    When on_title_refresh is provided it is polled periodically and may
    return replacement headline text.
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
        | AppKit.NSWindowCollectionBehaviorStationary
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
    title_field = centered_label(0.55, 90, title, 40, 1.0)
    content.addSubview_(title_field)
    if subtitle:
        content.addSubview_(centered_label(0.47, 40, subtitle, 24, 0.85))
    content.addSubview_(centered_label(0.10, 30, "Press any key or click to continue", 15, 0.55))

    coordinator = Coordinator.alloc().init()
    from Foundation import NSTimer, NSRunLoop

    def schedule(interval, repeats, selector):
        poll_timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, coordinator, selector, None, repeats
        )
        for mode in (
            AppKit.NSModalPanelRunLoopMode,
            AppKit.NSDefaultRunLoopMode,
        ):
            NSRunLoop.currentRunLoop().addTimer_forMode_(poll_timer, mode)
        return poll_timer

    close_timer = schedule(duration_seconds, False, "tick:") if duration_seconds else None
    poll_timer = None
    if on_title_refresh is not None:
        coordinator.on_title_refresh = on_title_refresh
        coordinator.title_field = title_field
        poll_timer = schedule(EYE_REST_POLL_SECONDS, True, "refreshTitle:")

    # Kiosk lockdown: hide the dock and menu bar, block app switching,
    # force quit and logout, then keep reclaiming focus so nothing else
    # can take the screen while the break card is up.
    kiosk_mask = (
        AppKit.NSApplicationPresentationHideDock
        | AppKit.NSApplicationPresentationHideMenuBar
        | AppKit.NSApplicationPresentationDisableProcessSwitching
        | AppKit.NSApplicationPresentationDisableForceQuit
        | AppKit.NSApplicationPresentationDisableSessionTermination
    )
    previous_presentation = app.presentationOptions()
    app.setPresentationOptions_(kiosk_mask)

    def guard_focus(timer):
        app.activateIgnoringOtherApps_(True)
        window.makeKeyAndOrderFront_(None)

    focus_guard = schedule(OVERLAY_FOCUS_GUARD_SECONDS, True, "guardFocus:")
    coordinator.guard_focus = guard_focus

    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)
    try:
        app.runModalForWindow_(window)
    finally:
        if close_timer is not None:
            close_timer.invalidate()
        if poll_timer is not None:
            poll_timer.invalidate()
        focus_guard.invalidate()
        app.setPresentationOptions_(previous_presentation)
        window.orderOut_(None)


def format_duration(total_seconds: int) -> str:
    minutes, seconds = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _water_streak(days: dict, goal: int = DAILY_WATER_GOAL) -> int:
    """Consecutive days (ending today or yesterday) that hit the water goal.

    Today only counts once the goal is already reached; otherwise the run
    starts from yesterday, Snapchat-style.
    """
    day = date.today()
    if days.get(day.isoformat(), {}).get("water_glasses", 0) < goal:
        day -= timedelta(days=1)
    streak = 0
    while days.get(day.isoformat(), {}).get("water_glasses", 0) >= goal:
        streak += 1
        day -= timedelta(days=1)
    return streak


def _best_water_streak(days: dict, goal: int = DAILY_WATER_GOAL) -> int:
    """Longest goal-hitting run anywhere in the recorded history.

    Days with no entry count as missed so gaps break the chain.
    """
    if not days:
        return 0
    earliest = min(date.fromisoformat(iso) for iso in days)
    best = run = 0
    day = earliest
    while day <= date.today():
        if days.get(day.isoformat(), {}).get("water_glasses", 0) >= goal:
            run += 1
            best = max(best, run)
        else:
            run = 0
        day += timedelta(days=1)
    return best


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

    def water_streak(self) -> int:
        return _water_streak(self.days)

    def best_water_streak(self) -> int:
        return _best_water_streak(self.days)

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
            f"  Water logged:   {today.get('water_glasses', 0)}/{DAILY_WATER_GOAL} glass(es)",
            f"  Water streak:   {self.water_streak()} day(s) (best {self.best_water_streak()})",
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
        with QuietStderr(), mp.solutions.face_detection.FaceDetection(
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
        self.debug = debug
        self.monitor = PresenceMonitor(debug=debug)
        self.show_preview = show_preview
        self._preview_capture = None
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
            self.break_required_away = 5
            self.eye_rest_required_away = 3
            self.overlay_cooldown = DEMO_OVERLAY_COOLDOWN_SECONDS
        else:
            self.stretch_interval = FOCUS_REMINDER_MINUTES * 60
            self.eye_rest_interval = EYE_RULE_MINUTES * 60
            self.break_required_away = BREAK_AWAY_SECONDS
            self.eye_rest_required_away = EYE_REST_AWAY_SECONDS
            self.overlay_cooldown = OVERLAY_COOLDOWN_SECONDS
        self.last_overlay_closed_at = 0.0
        self.input_hold_note_shown = False
        self.drink_detector = DrinkDetector()
        self.last_pose_call = 0.0
        self.last_auto_water_ts = 0.0
        self.started_at = time.time()
        self.grace_note_shown = False

    def _update_security(self, now: float) -> None:
        face = self.monitor.face_present

        # The vote needs a couple of samples before "absent" means anything;
        # never nag or count down while the camera is still forming its view.
        if not face and not self.monitor.vote.settled:
            return

        idle = None if face else _input_idle_seconds()
        attended = face or (
            idle is not None and idle < INPUT_ACTIVITY_GRACE_SECONDS
        )

        if attended:
            if face:
                if self.locked or self.absent_since is not None:
                    stamp = time.strftime("%H:%M:%S")
                    print(f"[{stamp}] Welcome back! Security guard re-armed.")
                self.input_hold_note_shown = False
            elif not self.input_hold_note_shown:
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] Out of frame, but input is active - staying unlocked.")
                self.input_hold_note_shown = True
            self.absent_since = None
            self.locked = False
            self.warned = False
            self.next_lock_attempt = 0.0
            return

        self.input_hold_note_shown = False

        if now - self.started_at < STARTUP_GRACE_SECONDS:
            if not self.grace_note_shown:
                remaining = int(STARTUP_GRACE_SECONDS - (now - self.started_at))
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] Guard warming up - auto-lock arms in ~{remaining}s.")
                self.grace_note_shown = True
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

    def _warm_drink_model(self) -> None:
        """Load the hand model on the main thread before loops start.

        MediaPipe binds its compute context to the creating thread on
        macOS, so the model must be created and first used here rather
        than in a helper thread, or every later call would fail.
        """
        try:
            import numpy as np

            _get_hands().process(np.zeros((64, 64, 3), dtype=np.uint8))
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] Drink detector ready.")
        except Exception as error:
            print(f"Drink detector unavailable: {error.__class__.__name__}: {error}")

    def _consider_drink(self, now: float) -> None:
        """Run pose detection at most once a second; log deduped drinks."""
        gap = now - self.last_pose_call
        if gap < EYE_REST_POLL_SECONDS:
            return
        self.last_pose_call = now
        read_frame = self._eye_rest_frame_provider()
        if not callable(read_frame):
            return
        frame = read_frame()
        if frame is None:
            return
        drinking_now = self.drink_detector.consider(frame, gap)
        self.pose_reads = getattr(self, "pose_reads", 0) + 1
        obs = dict(self.drink_detector.last_obs)
        obs["held"] = round(self.drink_detector.sustained, 1)
        if drinking_now or self.drink_detector.sustained > 0.5 or (
            self.pose_reads % 5 == 0 and self.monitor.face_present
        ):
            stamp = time.strftime("%H:%M:%S")
            print(
                f"[{stamp}] drink: {obs}"
                f"{' - LOGGED' if drinking_now else ''}"
            )
        if not drinking_now:
            return
        if now - self.last_auto_water_ts < DRINK_DEDUP_SECONDS:
            return
        self.last_auto_water_ts = now
        self._announce_water(self.stats.record_water(), auto=True)

    def _announce_water(self, count: int, auto: bool = False) -> None:
        stamp = time.strftime("%H:%M:%S")
        verb = "Detected a drink" if auto else "Logged a glass"
        goal = DAILY_WATER_GOAL
        if count == goal:
            print(f"[{stamp}] {verb} ({count}/{goal}) - daily goal hit, streak secured!")
        elif count > goal:
            print(f"[{stamp}] {verb} ({count}/{goal}) - bonus glass.")
        else:
            streak = self.stats.water_streak()
            remaining = goal - count
            tail = (
                f"{remaining} more today keeps the {streak}-day streak."
                if streak
                else f"{remaining} more today starts your streak."
            )
            print(f"[{stamp}] {verb} ({count}/{goal}). {tail}")

    def _update_wellness(self, now: float, delta: float) -> None:
        if self.monitor.face_present:
            self.stats.add_focus(delta)
            self._consider_drink(now)
            if now - self.last_overlay_closed_at < self.overlay_cooldown:
                return

            if self.focus_start is None:
                self.focus_start = now
            elif now - self.focus_start >= self.stretch_interval:
                self.stats.record_reminder()
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] Focus limit reached - showing break screen.")
                nagger = StretchNagger(
                    self._eye_rest_frame_provider(),
                    required_away_seconds=self.break_required_away,
                )
                show_break_overlay(
                    "Time for a break",
                    "Stand up and move around - the screen waits for you.",
                    None,
                    on_title_refresh=nagger.next_message,
                )
                self.last_overlay_closed_at = time.time()
                self.focus_start = self.last_overlay_closed_at
                self.eye_rest_start = None

            if self.eye_rest_start is None:
                self.eye_rest_start = now
            elif now - self.eye_rest_start >= self.eye_rest_interval:
                self.stats.record_eye_rest()
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] 20-20-20 - showing eye-rest screen.")
                nagger = GazeNagger(
                    self._eye_rest_frame_provider(),
                    required_away_seconds=self.eye_rest_required_away,
                )
                show_break_overlay(
                    "20-20-20",
                    "Look at something 6 metres away for 20 seconds.",
                    None,
                    on_title_refresh=nagger.next_message,
                )
                self.last_overlay_closed_at = time.time()
                self.eye_rest_start = self.last_overlay_closed_at

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
            f"water {self.stats.water_today()}/{DAILY_WATER_GOAL} | streak {self.stats.water_streak()}d (w)",
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

    def _eye_rest_frame_provider(self):
        if self.show_preview:
            capture = self._preview_capture

            def from_preview():
                grabbed, frame = capture.read()
                return frame if grabbed else None

            return from_preview

        def from_monitor():
            return self.monitor.latest_frame()

        return from_monitor

    def run(self) -> int:
        print("Step-Away is starting up. Press Ctrl+C to quit.")
        self._warm_drink_model()
        if self.stretch_interval == DEMO_STRETCH_SECONDS:
            print(
                f"Demo mode: break screen every {DEMO_STRETCH_SECONDS}s "
                f"(stays up until you stay away for 5s), eye rest every "
                f"{DEMO_EYE_REST_SECONDS}s (until you look away for 3s)."
            )
        print(f"Streak so far: {self.stats.current_streak()} day(s).")
        print(
            f"Water streak: {self.stats.water_streak()} day(s) "
            f"(best {self.stats.best_water_streak()}) - goal {DAILY_WATER_GOAL} glasses/day."
        )
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
        self._preview_capture = capture

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
                        self._announce_water(self.stats.record_water())
                        self.stats.save()
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
