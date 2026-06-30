# Future Improvements

## Near-term (high value, low-to-moderate effort)

- **On-device active learning loop**: when the decision engine has a
  near-miss (passes Stage 1-2 but fails Stage 3/4, or vice versa), save the
  clip to a `datasets/raw/review_queue/` folder for human labeling. This is
  the single highest-leverage way to close the gap between public-dataset
  pretraining and this exact venue's lighting/layout, without needing a
  second camera deployment cycle.
- **Per-venue auto-calibration wizard**: a guided CLI/dashboard flow that
  walks an installer through exposure/WB/zone-polygon calibration
  (`docs/optimization_guide.md` Section 4) instead of hand-editing YAML.
- **Smoldering-fire sub-mode**: the current motion-analysis stage (Stage 4)
  expects visible flicker/diffusion; a very early smoldering fire (e.g. a
  pinched cable under a podium) may show minimal motion before transitioning
  to visible flame. A slower, lower-confidence-threshold "watch" tier that
  doesn't alarm but flags for review could catch this earlier.
- **Multi-camera fusion within one hall**: if a hall has multiple Pi 5 units
  (e.g. one per wall), cross-referencing simultaneous detections from two
  angles would let the decision engine relax Stage 3/4 thresholds (faster
  alarm) when corroborated by a second camera, while keeping today's
  conservative single-camera thresholds when uncorroborated.
- **Central multi-room dashboard**: aggregate `/api/status` and `/api/events`
  from multiple Pi 5 units (one per hall/classroom) into a single
  building-wide view — today each unit's dashboard is standalone.

## Medium-term (meaningful architecture work)

- **YOLOv10n / YOLOv11n head-to-head on real trained weights**: the
  recommendation in `docs/model_comparison.md` is provisional pending real
  training; once a labeled indoor fire/smoke dataset exists, train all three
  candidates and let `docs/benchmark_report.md` numbers (not literature
  estimates) decide.
- **Hardware acceleration as an opt-in, not a requirement**: this system is
  explicitly scoped CPU-only per the project brief, but the ONNX Runtime
  execution-provider abstraction in `inference/detector.py` means adding an
  optional Hailo-8/Coral path later (for sites willing to add a HAT) is a
  config change (`providers=[...]`), not a rewrite — worth revisiting if a
  venue needs higher resolution/FPS than CPU-only can sustain.
- **Proper multi-object tracker upgrade**: the current `IoUTracker` is
  deliberately simple (greedy IoU matching). If false track-switching
  becomes an issue with multiple simultaneous fire sources or heavy
  occlusion, a lightweight Kalman-filter-based tracker (e.g. a trimmed
  ByteTrack) would improve track continuity without a large CPU cost
  increase.
- **Formal model card + per-class confusion matrix dashboard panel**:
  surface ongoing precision/recall per class (not just raw event counts) in
  the dashboard once enough operational data accumulates, so site staff can
  see drift (e.g. "thin_smoke recall dropped this month") proactively.

## Longer-term / research-flavored

- **Thermal/IR sensor fusion**: pairing the NoIR camera with a low-cost
  thermal sensor (e.g. MLX90640) as a second, independent modality for
  Stage 2 verification — heat signature confirmation would meaningfully
  strengthen the false-positive story for visually fire-colored but
  non-thermal sources (projected video, colored lighting), at added BOM
  cost and a new calibration/fusion problem.
- **Federated/shared model improvement across deployments**: if multiple
  venues run this system, a privacy-preserving way to share hard-negative
  patterns (not raw footage) across sites would compound the false-positive
  reduction work over time.
- **Formal certification path**: engage a fire protection engineering firm
  to evaluate whether (and how) a hardened version of this system could be
  certified as a supplementary/auxiliary detection device under a relevant
  standard, enabling deeper integration with building FACPs beyond the
  current "advisory relay" posture described in `docs/deployment_guide.md`.
