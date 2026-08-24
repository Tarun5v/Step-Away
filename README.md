# Step-Away

A webcam-powered presence guard for your Mac. Step-Away watches whether you are
at your desk, locks your screen the moment you step away, and reminds you to
stretch when you have been focused for too long.

## Features

- **Auto-lock security** - after a 60-second warm-up the guard arms: 5
  seconds away triggers a warning and at the 10-second mark the screen
  locks. Mouse or keyboard activity counts as presence too: if you are out
  of frame but actively using the machine, it stays unlocked.
- **Kiosk-style wellness screens** - while a break or eye-rest card is up,
  the dock and menu bar hide, app switching is blocked, and focus keeps
  returning to the card until you press a key or click.
- **Wellness breaks** - after 50 minutes of uninterrupted focus, the screen
  fades into a full-screen blurred break card that will not dismiss itself:
  it cycles through stretch instructions ("Stand up.", "Walk to the far
  side of the room.") and only releases once the camera confirms you have
  actually been up and away for a minute.
- **20-20-20 eye rule** - every 20 minutes at the desk a blurred eye-rest
  screen appears, and it does not blink first: it stays up until gaze
  tracking confirms you actually looked away for 20 seconds. Keep staring
  and it will call you out.
- **Automatic hydration tracking** - hand tracking watches for a hand
  a cup near your mouth and logs a glass by itself (max one per
  4 minutes); press `w` in the preview window to log manually.
- **Water streaks** - hit your daily goal of 8 glasses to extend the
  streak; miss a day and it resets to zero. Progress and streak live in
  the preview HUD and the exit report. Snapchat rules apply: keep the
  fire alive.
- **Break tracking** - stays away long enough (5+ minutes) and it counts as a
  logged break.
- **Daily stats & streaks** - focus time, nudges, breaks and day streaks are
  saved locally to `step_away_stats.json` and summarised at startup/shutdown.
- **Light on resources** - the webcam is sampled once every 3 seconds instead of
  running full-frame-rate detection.

## Requirements

- macOS
- Python 3.9+
- A built-in or external webcam

## Setup

```bash
git clone https://github.com/Tarun5v/Step-Away.git
cd Step-Away
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify that everything is in place (camera, face model, permissions):

```bash
python main.py --doctor
```

## Usage

```bash
python main.py
```

Press `Ctrl+C` to quit; a summary report is printed on exit.

### macOS permissions

| Permission    | Why it is needed                                              | Where to grant it                                                        |
| ------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Camera        | Face detection runs on the webcam feed                        | System Settings > Privacy & Security > Camera (allow your terminal app)   |
| Accessibility | Sending the `Cmd+Ctrl+Q` keystroke that locks the screen      | System Settings > Privacy & Security > Accessibility                      |
| Notifications | Stretch reminders                                             | System Settings > Notifications (allowed automatically on first prompt)   |

## Configuration

Tune the behaviour by editing the constants at the top of `main.py`:

| Constant                 | Default | Meaning                                       |
| ------------------------ | ------- | --------------------------------------------- |
| `CHECK_INTERVAL_SECONDS` | `3`     | How often the webcam frame is analysed        |
| `ABSENCE_LOCK_SECONDS`   | `10`    | Seconds away before the screen locks          |
| `ABSENCE_WARNING_SECONDS`| `5`     | Seconds away before the lock warning shows    |
| `LOCK_RETRY_SECONDS`     | `30`    | Delay between lock retries when locking fails |
| `PREVIEW_TARGET_FPS`     | `60`    | Frame rate target for the `--preview` window  |
| `FOCUS_REMINDER_MINUTES` | `50`    | Uninterrupted focus minutes per stretch nudge |
| `EYE_RULE_MINUTES`       | `20`    | Minutes between 20-20-20 eye-rest screens     |
| `DAILY_WATER_GOAL`       | `8`     | Glasses per day needed to extend the water streak |
| `DRINK_GESTURE_SECONDS`  | `2`     | How long a verified hand-at-mouth must hold to count |
| `DRINK_DEDUP_SECONDS`    | `240`   | Minimum gap between auto-logged glasses |
| `BREAK_AWAY_SECONDS`     | `60`    | Away-from-desk seconds before the break screen releases |
| `INPUT_ACTIVITY_GRACE_SECONDS` | `5` | Mouse/keyboard idle gap still counted as presence |
| `STARTUP_GRACE_SECONDS`  | `60`    | Delay after launch before auto-lock arms      |
| `OVERLAY_FOCUS_GUARD_SECONDS`  | `0.25` | How often overlays reclaim focus while up |
| `EYE_REST_AWAY_SECONDS`  | `20`    | Away-look seconds before the eye screen releases |
| `OVERLAY_COOLDOWN_SECONDS`| `60`   | Minimum quiet gap between wellness screens    |
| `BREAK_THRESHOLD_SECONDS`| `300`   | Away-time that counts as a real break         |

## Privacy

All processing happens locally. No video ever leaves your machine, and the only
file written is the local stats JSON in the project folder.

## Roadmap

- [ ] Menu bar app wrapper
- [ ] Weekly/monthly statistics charts
- [ ] Configurable lock action (sleep vs. lock)
- [ ] Windows/Linux support

## License

[MIT](LICENSE)
