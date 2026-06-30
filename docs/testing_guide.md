# Testing Guide

## 1. Unit tests (run on any dev machine, no Pi/camera required)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

Covers: config loading, the temporal verifier's frame-count + confidence
gating, the IoU tracker's track lifecycle, the false-positive color/bbox
checks, and the bounded frame queue. These are pure-logic tests with no
camera/GPIO/model dependency — they should pass identically on a laptop and
on the Pi.

## 2. Integration testing (requires camera, can run off-Pi with a webcam)

```bash
python scripts/run_headless.py
```

Confirms: camera backend starts (falls back to OpenCV automatically if
`picamera2` is unavailable), frames flow through the full pipeline, no
exceptions over a multi-minute run. Watch the log output
(`logs/fire_detection.log`) for `ALARM:` lines and stage-by-stage
DEBUG-level detail if you raise `system.log_level` to `DEBUG`.

## 3. Decision-engine validation without real fire (recommended primary method)

**Never start an open flame indoors to test this system.** Use these
safer proxies instead, in order of preference:

1. **Recorded fire/smoke footage**: play known-positive fire/smoke video
   clips on a laptop screen or projector and point the camera at it.
   Note: this is explicitly one of the **hard negatives** the false-positive
   filter is designed to reject ("fire video playing on a projector") — if
   your decision engine *does* alarm on played-back footage, that tells you
   the false-positive filter's color/flicker/motion checks need retuning,
   not that the detector is "working great." Use this primarily to validate
   Stage 1 (raw detection) in isolation by temporarily lowering
   `require_all_stages` or inspecting `stage_results` in the logs, not to
   validate the full pipeline's alarm decision.
2. **A controlled candle or lighter flame**, briefly, in a well-ventilated
   space, with someone present and an extinguisher on hand, at a safe
   distance from the camera and any combustibles — this is real (small)
   flame and is the right way to validate the full Stage 1-5 path end to
   end, including that `min_average_confidence` and the 8-frame requirement
   are actually achievable for genuine small flame at your camera's
   distance/angle.
3. **For smoke**: a few seconds of incense or a smoke pen, again in a
   ventilated space.
4. **For false-positive rejection**: walk through the hard-negative list in
   `configs/dataset.yaml:negative_hard_examples` — shine a phone flashlight
   at the camera, wear bright orange clothing in frame, turn on/off stage
   lighting, point a laser pointer at the lens, etc. — and confirm **no**
   alarm fires (check `/api/events` and the logs for any AlarmEvent at all,
   even sub-critical "warning" severity).

## 4. Soak test (before any real deployment)

Run continuously for **at least 24 hours** (the spec's stated runtime
requirement) under realistic room conditions (people present, lights
on/off cycling, projector in occasional use):

```bash
bash scripts/run_dashboard.sh &
# let it run, periodically check:
curl -s http://localhost:8000/api/status | python3 -m json.tool
vcgencmd measure_temp
vcgencmd get_throttled     # must report throttled=0x0 the whole time
```

Acceptance:
- [ ] No crash/restart over 24h (`journalctl -u fire-detection.service` clean)
- [ ] CPU utilization stayed below 80% (sampled via `/api/status`)
- [ ] No thermal throttling (`vcgencmd get_throttled` stays `0x0`)
- [ ] Zero false alarms over the period (check `/api/events`)
- [ ] FPS stayed within the 10-15 target band

## 5. Power measurement

Use an inline USB-C power meter between the official supply and the Pi.
Record min/avg/max watts over a 1-hour window during active inference,
including one deliberate alarm trigger (siren+relay draw briefly more).
Confirm max stays under the 15W budget — see `docs/wiring_diagram.md`
"Power budget sanity check" for expected component-level draws.

## 6. Regression testing after model updates

Before promoting a newly trained/exported model to `configs/model.yaml:active_model`:

```bash
yolo val model=<new-weights>.pt data=datasets/processed/data.yaml imgsz=416
bash scripts/benchmark_models.sh   # confirm latency/CPU/memory still meet targets
pytest tests/ -v                    # confirm no regression in decision-engine logic
```

Then run the hard-negative walkthrough (Section 3.4) again with the new
model before going live — a retrained detector can shift the false-positive
profile even if the decision-engine code didn't change.
