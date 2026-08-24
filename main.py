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
DRAIN_GRABS = 6
DEBUG_FRAME_DIR = ".debug_frames"
DEBUG_SNAPSHOT_LIMIT = 40
MOTION_RESIZE_WIDTH = 160
PREVIEW_WINDOW_TITLE = "Step-Away camera"
ABSENCE_LOCK_SECONDS = 10
ABSENCE_WARNING_SECONDS = 5
LOCK_COMMAND_TIMEOUT_SECONDS = 5
LOCK_RETRY_SECONDS = 30
FOCUS_REMINDER_MINUTES = 50
BREAK_THRESHOLD_SECONDS = 300
STATS_FILE = "step_away_stats.json"
SAVE_INTERVAL_SECONDS = 60


class PresenceMonitor(threading.Thread):
    """Background thread that keeps track of whether anyone sits at the desk."""

    def __init__(self, camera_index: int = CAMERA_INDEX, debug: bool = False):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.debug = debug
        self._state_lock = threading.Lock()
        self._recent_hits = deque(maxlen=PRESENCE_WINDOW_SIZE)
        self._running = True
        self.error = None
        self._prev_motion_frame = None
        self._snapshot_index = 0
        self._latest_view = None

    @property
    def face_present(self) -> bool:
        with self._state_lock:
            return sum(self._recent_hits) >= PRESENCE_MIN_HITS

    @property
    def latest_view(self):
        with self._state_lock:
            return self._latest_view

    def stop(self) -> None:
        self._running = False

    def _open_camera(self):
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Make sure it exists and is not in use by another app."
            )
        return capture

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
                with self._state_lock:
                    self._recent_hits.append(raw_present)
                    self._latest_view = (frame.copy(), boxes)
                    hits = sum(self._recent_hits)

                if self.debug:
                    stamp = time.strftime("%H:%M:%S")
                    verdict = "HIT " if raw_present else "MISS"
                    print(
                        f"[{stamp}] check {verdict} window {hits}/"
                        f"{len(self._recent_hits)} motion {motion:.1f}"
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


def send_stretch_notification() -> None:
    """Post a desktop reminder to stand up and stretch."""
    title = "Step-Away"
    message = "You have been focused for a while. Time to stretch and hydrate!"
    try:
        from plyer import notification

        notification.notify(title=title, message=message, timeout=10)
    except Exception:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True)


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
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
            self.days = payload.get("days", {})
        except (json.JSONDecodeError, OSError):
            print("Stats file was unreadable; starting a fresh one.")
            self.days = {}

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps({"days": self.days}, indent=2))
        except OSError as error:
            print(f"Could not save stats: {error}")

    def _today(self) -> dict:
        return self.days.setdefault(
            date.today().isoformat(), {"focus_seconds": 0, "reminders": 0, "breaks": 0}
        )

    def add_focus(self, seconds: float) -> None:
        self._today()["focus_seconds"] += int(seconds)

    def record_reminder(self) -> None:
        self._today()["reminders"] += 1

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
            f"  Breaks taken:   {today['breaks']}",
            f"  Current streak: {streak} day{'s' if streak != 1 else ''}",
        ]
        return "\n".join(lines)


class StepAwayApp:
    """Ties presence monitoring into security and wellness behaviours."""

    def __init__(self, debug: bool = False, show_preview: bool = False):
        self.monitor = PresenceMonitor(debug=debug)
        self.show_preview = show_preview
        self.presence_announced = False
        self.stats = StatsStore()
        self.absent_since = None
        self.locked = False
        self.warned = False
        self.focus_start = None
        self.break_counted = True
        self.next_lock_attempt = 0.0
        self.last_tick = time.time()
        self.last_save = time.time()

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
            elif now - self.focus_start >= FOCUS_REMINDER_MINUTES * 60:
                send_stretch_notification()
                self.stats.record_reminder()
                stamp = time.strftime("%H:%M:%S")
                minutes = FOCUS_REMINDER_MINUTES
                print(f"[{stamp}] {minutes} minutes of focus - stretch nudge sent.")
                self.focus_start = now
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
            self.break_counted = False

    def _draw_preview(self) -> None:
        view = self.monitor.latest_view
        if view is None:
            return
        frame, boxes = view
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 80), 2)
        state = "watching you" if self.presence_announced else "desk empty"
        cv2.putText(
            frame,
            f"Step-Away: {state}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 200, 80),
            2,
        )
        cv2.imshow(PREVIEW_WINDOW_TITLE, frame)
        cv2.waitKey(1)

    def run(self) -> int:
        print("Step-Away is starting up. Press Ctrl+C to quit.")
        print(f"Streak so far: {self.stats.current_streak()} day(s).")
        self.stats.save()
        self.monitor.start()

        exit_code = 0
        try:
            while self.monitor.is_alive():
                now = time.time()
                delta = min(now - self.last_tick, CHECK_INTERVAL_SECONDS * 2)
                self.last_tick = now

                present = self.monitor.face_present
                if present != self.presence_announced:
                    stamp = time.strftime("%H:%M:%S")
                    state = "arrived" if present else "left"
                    print(f"[{stamp}] You {state} the desk.")
                    self.presence_announced = present

                if self.show_preview:
                    self._draw_preview()

                self._update_security(now)
                self._update_wellness(now, delta)
                self._reset_focus_session()

                if now - self.last_save >= SAVE_INTERVAL_SECONDS:
                    self.stats.save()
                    self.last_save = now

                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.monitor.stop()
        self.monitor.join(timeout=CHECK_INTERVAL_SECONDS * 2)

        if self.show_preview:
            cv2.destroyAllWindows()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-Away presence guard")
    parser.add_argument(
        "--debug", action="store_true", help="print every webcam check result"
    )
    parser.add_argument(
        "--preview", action="store_true", help="show a live camera window with face boxes"
    )
    args = parser.parse_args()
    sys.exit(StepAwayApp(debug=args.debug, show_preview=args.preview).run())
