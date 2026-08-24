# Step-Away

A webcam-powered presence guard for your Mac. Step-Away watches whether you are
at your desk, locks your screen the moment you step away, and reminds you to
stretch when you have been focused for too long.

## Features

- **Auto-lock security** - after 5 seconds away you get a warning; if you are
  still gone at the 10-second mark, the screen locks with `Cmd+Ctrl+Q`.
- **Wellness nudges** - a desktop notification after 50 minutes of
  uninterrupted focus so you remember to stand up and stretch.
- **20-20-20 eye rule** - every 20 minutes at the desk you get an eye-rest
  nudge: look 6 metres away for 20 seconds to fight eye strain.
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
| `EYE_RULE_MINUTES`       | `20`    | Minutes between 20-20-20 eye-rest nudges      |
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
