# User Manual

## What this system does

A camera mounted in your seminar hall / classroom / auditorium continuously
watches for visible signs of fire or smoke. If it sees something consistent
with real fire or smoke for several seconds in a row (not a single flicker),
it sounds a local siren, logs the event with a saved photo and video clip,
and can notify staff over the network.

It does **not** replace your building's certified smoke detectors or fire
alarm system — think of it as an early-warning camera that can catch things
(like a smoldering podium short, or a curtain starting to catch a spark)
visually, before smoke reaches a ceiling-mounted detector.

## Dashboard

Open `http://<device-ip>:8000` in a browser on the same network.

- **Live view (top left)** — the camera feed with red boxes drawn around
  anything currently being evaluated as a possible fire/smoke source, with
  the predicted class and confidence.
- **System Telemetry (top right)** — CPU load, RAM use, temperature, and
  current frame rate of the device. If "Alarm" shows "ACTIVE", the siren is
  sounding (or was auto-silenced after timeout but not yet acknowledged).
- **Acknowledge & Reset Alarm button** — once staff have verified/handled
  the situation, press this to silence the siren and clear the active-alarm
  state. The event stays permanently in the history log either way.
- **Detection History (bottom)** — every confirmed alarm event, with
  timestamp, class, severity, confidence, the affected zone (e.g. "near
  stage"), and links to the saved snapshot image and video clip.

## What happens when an alarm fires

1. The buzzer and relay-driven siren activate, and the status LED blinks.
2. A snapshot photo and a short video clip (covering ~5 seconds before and
   ~15 seconds after the trigger) are saved automatically.
3. The event is logged to the on-device database, visible in the dashboard.
4. If configured, a notification is sent (MQTT topic, and/or Firebase push,
   and/or a webhook to your building management system).
5. The siren auto-silences after 5 minutes if nobody acknowledges it, but
   the alarm is still logged as "active" until someone presses **Acknowledge
   & Reset** on the dashboard — this is intentional, so a transient power
   cycle or unattended hall doesn't silently clear a real event.

## What does NOT trigger an alarm

By design, the system ignores, and will not alarm on:

- Sunlight through windows, sunset light
- Orange/red/yellow stage lighting and LED indicators
- Projector beams, TV/projector screen content (including fire footage
  played on a screen — this is explicitly tested against, see
  `docs/testing_guide.md`)
- Mobile phone flashlights
- Decorative string lights
- Reflections and glare off glass/metal
- Bright orange clothing or fabric
- Lit candles (small, steady flame sources are filtered as a known
  false-positive class via flicker/temporal characteristics — note this
  means the system is *not* a candle-safety monitor)
- Laser pointers / stage lasers

If you believe the system *should* have alarmed on a real fire and didn't,
or alarmed incorrectly, see `docs/maintenance_manual.md` "Tuning false
positives/negatives" and contact whoever administers `configs/decision.yaml`
for your site.

## Daily use checklist for hall staff

- Confirm the dashboard is reachable and "Alarm: Normal" before each event.
- If "Alarm: ACTIVE" is showing from a previous unattended trigger, review
  the saved snapshot/clip in the history table before acknowledging — verify
  whether it was a real event, a near-miss worth noting, or a false alarm
  to report for tuning.
- Do not block the camera's view of the stage/podium/projector area with
  banners, signage, or equipment — this is the system's primary coverage
  zone.

## Who to contact

This is a building/IT-managed system. For false alarms, missed detections,
hardware issues (camera obstructed, siren not sounding), or to request a
coverage-zone change, contact your facility's AV/IT support team, who can
reference `docs/maintenance_manual.md` and `docs/deployment_guide.md`.
