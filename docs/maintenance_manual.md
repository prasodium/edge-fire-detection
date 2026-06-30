# Maintenance Manual

## 1. Routine maintenance schedule

| Task | Frequency | How |
|---|---|---|
| Visual lens check (dust, obstruction) | Weekly | Compare live dashboard view against a baseline reference photo |
| Disk space check | Weekly | `df -h` on the Pi; `recording.max_storage_gb` in `configs/alarm.yaml` bounds snapshot/clip growth, but confirm pruning is actually running (see 3.) |
| Log review | Weekly | `journalctl -u fire-detection.service --since "7 days ago" \| grep -i error` |
| Soak-test-style health check | Monthly | `vcgencmd get_throttled`, `/api/status` CPU/temp trend, confirm no creeping resource growth |
| Camera recalibration | After any re-mounting, lighting change (new fixtures, room reconfiguration), or seasonal lighting shift | Repeat `docs/optimization_guide.md` "Camera Calibration" |
| Full system update | Quarterly, or on security advisory | `docs/deployment_guide.md` "Rollback / updating" |
| Physical alarm test | Monthly | Trigger via a controlled test (Section 5 below), confirm siren/relay/LED all activate and the dashboard reflects it |

## 2. Log locations

- `logs/fire_detection.log` — full application log (rotates at 10MB, 10 backups)
- `logs/alarm_events.log` — alarm-specific log only (rotates at 5MB, 20 backups)
- `storage/events.db` — SQLite event history (query directly with `sqlite3 storage/events.db`)
- `storage/snapshots/`, `storage/clips/` — saved evidence media
- `journalctl -u fire-detection.service` — systemd-captured stdout/stderr if running as a service

## 3. Storage pruning

`configs/alarm.yaml:database.retention_days` and `recording.max_storage_gb`
declare the intended retention policy. Enforce the database side via:

```bash
python3 -c "
from storage.db import EventDatabase
from utils.config import load_config
cfg = load_config()
db = EventDatabase(cfg.project_root() / cfg.system['paths']['database_path'])
removed = db.prune_older_than(cfg.alarm['database']['retention_days'])
print(f'Pruned {removed} old event rows')
"
```

For snapshot/clip files on disk, set up a cron job to delete files in
`storage/snapshots/` and `storage/clips/` older than `retention_days`, or
extend `EventDatabase`/`alarm_manager.py` to delete the referenced files
when pruning DB rows — not implemented by default because file-deletion
policy (e.g. "never delete critical-severity evidence") is a site-specific
decision worth making deliberately rather than defaulting silently.

## 4. Tuning false positives / false negatives

All thresholds live in `configs/decision.yaml`. After collecting real
operational data (`storage/events.db` + saved clips):

- **Too many false alarms**: review the saved clips for the false-alarm
  events. Identify which Stage 2 check should have rejected it
  (`fp_filter_detail`/`smoke_detail` are logged at DEBUG level in
  `inference/decision_engine.py`) and tighten that specific threshold
  (e.g. raise `min_average_confidence`, narrow `flame_hue_range_deg`,
  shorten `max_frequency_hz` if reacting to a faster-than-flame-flicker
  light source). Avoid blanket-raising `consecutive_frames_required` first —
  that increases detection latency for real fires, which is the cost you
  want to minimize last, not first.
- **Missed a real event (false negative)**: check whether the detector
  itself missed it (Stage 1 — needs retraining/more data for that scenario)
  or whether the decision engine rejected a true detection (Stage 2-4 —
  review whether the flicker/color/motion assumptions hold for that fire
  type, e.g. a very slow-smoldering fire may not show the expected motion
  signature; consider a "smoldering" sub-mode if this becomes a recurring
  pattern in your venue).
- **Per-venue drift**: re-run the static-background learning
  (`known_light_source_suppression`) after any fixed-lighting change by
  restarting the service — the background model only learns at startup.

## 5. Testing the physical alarm path without a real detection

```python
from alarm.alarm_manager import AlarmManager
from inference.decision_engine import AlarmEvent
from utils.config import load_config
import numpy as np, time

cfg = load_config()
mgr = AlarmManager(
    alarm_cfg=cfg.alarm,
    db_path=str(cfg.project_root() / cfg.system["paths"]["database_path"]),
    snapshots_dir=str(cfg.project_root() / cfg.system["paths"]["snapshots_dir"]),
    clips_dir=str(cfg.project_root() / cfg.system["paths"]["clips_dir"]),
)
fake_event = AlarmEvent(
    track_id=0, class_name="small_flame", severity="critical", confidence=0.95,
    box_xyxy=(0, 0, 100, 100), zones=["stage"],
)
mgr.trigger(fake_event, np.zeros((480, 640, 3), dtype="uint8"))
time.sleep(2)
mgr.acknowledge_and_reset()
```

This exercises GPIO, recording, DB logging, and notifications end-to-end
without needing a real or simulated camera detection.

## 6. Hardware troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Dashboard shows no live image | Camera not detected / CSI cable seated wrong | `libcamera-hello --list-cameras`; reseat CSI ribbon (blue side toward Ethernet port on most Pi 5 camera ports) |
| Siren doesn't sound on test trigger | GPIO wiring, or running off-Pi (mock backend silently no-ops) | Confirm `gpio.backend: gpiozero` in `configs/alarm.yaml` and check `logs/fire_detection.log` for "GpioController using mock backend" — that message means it's not actually on the real GPIO path |
| High CPU / dropped frames | Thermal throttling, or `active_model` pointing at FP32/640px when INT8/416px was intended | Check `vcgencmd get_throttled`; confirm `configs/model.yaml:active_model` |
| Pi randomly reboots under load | Undervoltage | Use the official 27W supply; check `vcgencmd get_throttled` for the undervoltage bit |
| MQTT notifications not arriving | Broker not running / wrong host | `systemctl status mosquitto`; `mosquitto_sub -t 'firealert/#'` to confirm publish is reaching the broker |

## 7. Backup before any major change

```bash
cp configs/*.yaml /path/to/backup/
cp storage/events.db /path/to/backup/
cp -r weights/ /path/to/backup/weights_backup/
```
