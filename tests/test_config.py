from tokenslim.config import Config, load_config


def test_defaults():
    c = load_config(env={})
    assert c.min_bytes == 200
    assert c.enabled is True
    assert c.model is None


def test_env_overrides_defaults():
    env = {
        "TOKENSLIM_MIN_BYTES": "50",
        "TOKENSLIM_ENABLED": "false",
        "TOKENSLIM_MODEL": "gpt-4o",
        "TOKENSLIM_TELEMETRY": "1",
    }
    c = load_config(env=env)
    assert c.min_bytes == 50
    assert c.enabled is False
    assert c.model == "gpt-4o"
    assert c.telemetry is True


def test_enabled_compressors_parsed_as_tuple():
    c = load_config(env={"TOKENSLIM_ENABLED_COMPRESSORS": "json-minify, passthrough"})
    assert c.enabled_compressors == ("json-minify", "passthrough")


def test_per_call_overrides_win_over_env():
    c = load_config(env={"TOKENSLIM_MIN_BYTES": "50"}, min_bytes=999)
    assert c.min_bytes == 999


def test_unknown_env_vars_ignored():
    c = load_config(env={"TOKENSLIM_BOGUS": "x", "OTHER": "y"})
    assert c == Config()


def test_merged_ignores_none():
    base = Config(min_bytes=10)
    assert base.merged(min_bytes=None).min_bytes == 10
    assert base.merged(min_bytes=20).min_bytes == 20
