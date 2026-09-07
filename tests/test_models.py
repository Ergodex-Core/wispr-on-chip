import pytest

from wispr_on_chip.models import MODELS, get_model


@pytest.mark.parametrize("name", sorted(MODELS))
def test_param_counts_match_published(name):
    m = MODELS[name]
    assert m.reference_params
    rel = abs(m.total_params - m.reference_params) / m.reference_params
    assert rel < 0.005, f"{name}: {m.total_params} vs published {m.reference_params}"


def test_stage_order_and_kinds():
    m = get_model("whisper-tiny")
    kinds = [s.kind for s in m.stages()]
    assert kinds[0] == "stem"
    assert kinds[1:5] == ["encoder"] * 4
    assert kinds[5:9] == ["decoder"] * 4
    assert kinds[-1] == "head"


def test_llm_has_no_encoder_and_untied_embed():
    m = get_model("llama-3.1-8b")
    kinds = [s.kind for s in m.stages()]
    assert "encoder" not in kinds and "stem" not in kinds
    assert kinds[0] == "embed" and kinds[-1] == "head"
    assert m.d_kv == 1024  # 8 kv heads x 128


def test_decoder_stage_accounts_cross_kv_projection_per_window():
    m = get_model("whisper-base")
    dec = [s for s in m.stages() if s.kind == "decoder"][0]
    d = m.d_model
    assert dec.window_macs == 2 * m.n_audio_ctx * d * d
    assert dec.linear_macs_per_item == dec.params - 2 * d * d
    assert dec.kv_elements_per_stream == 2 * m.n_text_ctx * d + 2 * m.n_audio_ctx * d


def test_get_model_aliases():
    assert get_model("large-v3") is MODELS["whisper-large-v3"]
    with pytest.raises(KeyError):
        get_model("nope")
