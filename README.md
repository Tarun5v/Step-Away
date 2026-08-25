# Step-Away

A webcam-powered presence guard for your Mac. It watches whether you're at your
desk, locks the screen the moment you step away, and forces you to stretch when
you've been focused for too long.

## Download & Run (no coding needed)

1. Grab **`Step-Away-v0.3.0-macOS.dmg`** from the
   [latest release](https://github.com/Tarun5v/Step-Away/releases/latest)
   and double-click it.
2. Drag **Step Away** into your Applications folder, then open it from there.
3. First launch only — if macOS says *"Apple can't check app for malicious
   software"*:
   1. Click **Done**, do **not** delete the app.
   2. Open **System Settings > Privacy & Security**.
   3. Scroll to Security, click **Open Anyway** > **Open**.
   4. Enter your login password.
   This is a one-time step. After that the app opens like any other.
4. Allow camera access when macOS asks. Everything stays on your Mac.

Requires an Apple Silicon Mac (M1/M2/M3/M4).

## Features

- **Auto-lock** — arms after a 60-second warm-up; 5 seconds away triggers a
  warning, 10 seconds locks the screen. Typing or moving the mouse counts as
  presence too.
- **Wellness breaks** — after 50 minutes of uninterrupted focus a full-screen
  break card appears and won't go away until you've actually stood up and
  walked away for a minute.
- **20-20-20 eye rule** — every 20 minutes a blurred eye-rest screen appears
  and stays up until gaze tracking confirms you looked away for 20 seconds.
- **Water tracking** — press `w` to log a glass. Hit 8 glasses a day to keep
  the streak alive; miss a day and it resets. Snapchat rules apply.
- **Break tracking** — stepping away for 5+ minutes counts as a logged break.
- **Daily stats** — focus time, nudges, breaks and streaks are saved locally
  and summarised at startup and shutdown.
- **Kiosk overlays** — while a break or eye-rest card is up, the dock and menu
  bar hide and app switching is blocked so you can't ignore it.
- **Light on resources** — webcam is sampled once every 3 seconds, not
  full-frame-rate.

## Keyboard Shortcuts

| Key     | Action                          |
| ------- | ------------------------------- |
| `w`     | Log a glass of water            |
| `q`     | Quit the app                    |
| Escape  | Quit the app                    |
| `Cmd+Q` | Quit (works even during breaks) |

## How It Works

Every 3 seconds a webcam frame is fed to MediaPipe's Face Mesh, which
localises the eyes, nose and face outline in 3D. A presence vote accumulates
evidence over several frames to avoid flicker. Once the vote is settled on
"away", the lock timer starts; once the vote is settled on "present", it
resets. Input activity (mouse/keyboard) is checked independently as a
fallback — you can be out of frame and still typing without triggering a lock.

## Why I Built This

I kept sitting for hours without moving. Existing break-reminder apps show a
notification you can dismiss in a second. I wanted something that actually
forces you to get up: the screen locks, the break card won't close, and the
eye-rest screen won't let you stare. If you're going to cheat, the app has to
work harder than you do.

## Technical Challenges

A few things that went wrong on the way here:

- **Drink detection tried and removed** — I spent weeks building AI-powered
  hand-pose tracking to auto-detect when you drink water. MediaPipe Hands,
  face-landmark verification, lip-overlap detection from the HydroVisor paper,
  cup-down separation logic. In the end the false-positive rate was too high to
  trust. Manual logging with `w` turned out to be simpler and more reliable.
- **MediaPipe 1.0 broke everything** — the `mp.solutions` API was removed in
  MediaPipe 1.0. Pinned to `mediapipe==0.10.21` to keep it working.
- **Gatekeeper crash** — macOS aborts unsigned apps that touch the camera
  without an `NSCameraUsageDescription` in their Info.plist. The crash looked
  terrifying ("Apple can't check for malicious software") but the fix was one
  plist key.
- **App Translocation** — unsigned apps downloaded from the internet run from
  a randomised read-only path. The app had to be built to never write outside
  its own directory.

## For Developers

### Setup

```bash
git clone https://github.com/Tarun5v/Step-Away.git
cd Step-Away
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py                 # headless — locks screen when you leave
python main.py --preview       # live camera window with face boxes
python main.py --preview --demo # fast timers for testing
python main.py --doctor        # verify camera, model, permissions
```

Press `Ctrl+C` to quit; a summary report prints on exit.

### macOS Permissions

| Permission    | Why it is needed                                         | Where to grant it                                                    |
| ------------- | -------------------------------------------------------- | -------------------------------------------------------------------- |
| Camera        | Face detection on webcam feed                            | System Settings > Privacy & Security > Camera (allow your terminal)  |
| Accessibility | `Cmd+Ctrl+Q` keystroke that locks the screen             | System Settings > Privacy & Security > Accessibility                  |
| Notifications | Stretch reminders                                        | System Settings > Notifications (allowed on first prompt)            |

### Configuration

All tuning constants live at the top of `main.py`. The important ones:

| Constant                 | Default | What it does                            |
| ------------------------ | ------- | --------------------------------------- |
| `ABSENCE_LOCK_SECONDS`   | `10`    | Seconds away before screen locks        |
| `FOCUS_REMINDER_MINUTES` | `50`    | Focus minutes per stretch nudge         |
| `EYE_RULE_MINUTES`       | `20`    | Minutes between eye-rest screens        |
| `DAILY_WATER_GOAL`       | `8`     | Glasses per day for the streak          |
| `STARTUP_GRACE_SECONDS`  | `60`    | Delay before auto-lock arms             |

<details>
<summary>All configuration constants</summary>

| Constant                      | Default | Meaning                                            |
| ----------------------------- | ------- | -------------------------------------------------- |
| `CHECK_INTERVAL_SECONDS`      | `3`     | How often the webcam frame is analysed             |
| `ABSENCE_LOCK_SECONDS`        | `10`    | Seconds away before the screen locks               |
| `ABSENCE_WARNING_SECONDS`     | `5`     | Seconds away before the lock warning shows         |
| `LOCK_RETRY_SECONDS`          | `30`    | Delay between lock retries when locking fails      |
| `PREVIEW_TARGET_FPS`          | `60`    | Frame rate target for the `--preview` window       |
| `FOCUS_REMINDER_MINUTES`      | `50`    | Uninterrupted focus minutes per stretch nudge      |
| `EYE_RULE_MINUTES`            | `20`    | Minutes between 20-20-20 eye-rest screens          |
| `DAILY_WATER_GOAL`            | `8`     | Glasses per day needed to extend the water streak  |
| `BREAK_AWAY_SECONDS`          | `60`    | Away-from-desk seconds before break screen releases|
| `INPUT_ACTIVITY_GRACE_SECONDS`| `5`     | Mouse/keyboard idle gap still counted as presence  |
| `STARTUP_GRACE_SECONDS`       | `60`    | Delay after launch before auto-lock arms           |
| `OVERLAY_FOCUS_GUARD_SECONDS` | `0.25`  | How often overlays reclaim focus while up          |
| `EYE_REST_AWAY_SECONDS`       | `20`    | Away-look seconds before the eye screen releases   |
| `OVERLAY_COOLDOWN_SECONDS`    | `60`    | Minimum quiet gap between wellness screens         |
| `BREAK_THRESHOLD_SECONDS`     | `300`   | Away-time that counts as a real break              |

</details>

### Building the App

```bash
./scripts/build_mac.sh    # produces dist/Step Away.app (~350 MB)
```

Requires PyInstaller (`pip install pyinstaller`). The build script injects the
camera permission plist entry and re-signs the bundle.

## Roadmap

- **Windows & Linux support** — the camera/face-mesh pipeline is already
  cross-platform; only screen locking and kiosk overlays need per-OS backends.
- **Weekly report charts** — matplotlib charts for the weekly review screen,
  like GitHub's contribution graph.
- **Menu bar app wrapper** — live status in the menu bar instead of (or
  alongside) the preview window.
- **Configurable lock action** — let the user choose sleep vs. lock vs. logout.

## Privacy

All processing happens locally. No video ever leaves your machine. The only
file written is `step_away_stats.json` in the project folder.

## License

[MIT](LICENSE)
