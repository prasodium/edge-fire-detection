from utils.config import load_config, reload_config


def test_load_config_returns_all_sections():
    cfg = reload_config()
    assert cfg.system["max_cpu_percent"] == 80
    assert cfg.camera["frame_rate"] == 15
    assert "classes" in cfg.model
    # classes count is >=2 rather than a fixed 15: configs/model.yaml may be in
    # DEMO MODE (2-class small_flame/smoke) or restored to the full production
    # 15-class taxonomy - see configs/model.yaml's "DEMO MODE" header comment.
    assert len(cfg.model["classes"]) >= 2
    assert "temporal_verification" in cfg.decision
    assert "gpio" in cfg.alarm


def test_active_model_spec_resolves():
    cfg = load_config()
    spec = cfg.active_model_spec()
    assert "weights" in spec
    assert spec["input_size"] in (320, 416, 640)


def test_config_is_cached():
    a = load_config()
    b = load_config()
    assert a is b
