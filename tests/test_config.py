from utils.config import load_config, reload_config


def test_load_config_returns_all_sections():
    cfg = reload_config()
    assert cfg.system["max_cpu_percent"] == 80
    assert cfg.camera["frame_rate"] == 15
    assert "classes" in cfg.model
    assert len(cfg.model["classes"]) == 15
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
