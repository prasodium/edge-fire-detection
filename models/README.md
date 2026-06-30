# models/

Reserved for custom model architecture code (e.g. a hand-written PyTorch
`nn.Module` if this project ever moves beyond off-the-shelf nano detector
architectures).

It is currently empty by design: per `docs/model_comparison.md`, this
project uses **off-the-shelf pretrained architectures** (YOLOv8n and
siblings) via the `ultralytics` package for training and plain ONNX Runtime
for inference (`inference/detector.py`) — there is no custom model
definition to maintain. `configs/model.yaml` is the model *registry*
(precision/resolution/weights-path variants); `training/train.py` is where
architecture selection happens (`--base-model` flag).

If you later add a genuinely custom architecture (e.g. a distilled student
network not expressible as a standard Ultralytics config), it belongs here.
