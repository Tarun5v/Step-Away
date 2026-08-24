"""Step-Away: a webcam-powered presence guard for your Mac.

Watches for you at the desk, locks the machine when you step away,
and nudges you to stretch after long focus sessions.
"""

import json
import subprocess
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
CHECK_INTERVAL_SECONDS = 3
MISSES_TO_MARK_ABSENT = 2
HITS_TO_MARK_PRESENT = 1
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

    def __init__(self, camera_index: int = CAMERA_INDEX):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self._state_lock = threading.Lock()
        self._face_present = False
        self._pending_present = False
        self._pending_count = 0
        self._running = True
        self.error = None

    @property
    def face_present(self) -> bool:
        with self._state_lock:
            return self._face_present

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

    def _detect_face(self, detector, frame) -> bool:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = detector.process(image)
        return bool(result.detections)

    def run(self) -> None:
        try:
            capture = self._open_camera()
        except RuntimeError as error:
            self.error = str(error)
            self._running = False
            return

        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
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
                raw_present = self._detect_face(detector, frame)
                with self._state_lock:
                    changed = self._apply_debounce(raw_present)
                    present = self._face_present

                if changed:
                    state = "arrived" if present else "left"
                    stamp = time.strftime("%H:%M:%S")
                    print(f"[{stamp}] You {state} the desk.")

                time.sleep(CHECK_INTERVAL_SECONDS)

        capture.release()

    def _apply_debounce(self, raw_present: bool) -> bool:
        """Flip the stable state only after repeated identical readings."""
        if raw_present == self._face_present:
            self._pending_count = 0
            return False

        needed = HITS_TO_MARK_PRESENT if raw_present else MISSES_TO_MARK_ABSENT
        if self._pending_present == raw_present:
            self._pending_count += 1
        else:
            self._pending_present = raw_present
            self._pending_count = 1

        if self._pending_count < needed:
            return False

        self._face_present = raw_present
        self._pending_count = 0
        return True


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

    def __init__(self):
        self.monitor = PresenceMonitor()
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
    sys.exit(StepAwayApp().run())
