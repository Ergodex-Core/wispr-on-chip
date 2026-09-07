import json

from wispr_on_chip.cli import main


def test_models_and_processes(capsys):
    assert main(["models"]) == 0
    assert "whisper-large-v3" in capsys.readouterr().out
    assert main(["processes"]) == 0
    assert "n5" in capsys.readouterr().out


def test_plan_text_and_ascii(capsys):
    assert main(["plan", "--model", "whisper-small", "--ascii"]) == 0
    out = capsys.readouterr().out
    assert "FITS" in out and "performance" in out and "E=encoder" in out


def test_plan_json(capsys):
    assert main(["plan", "--model", "tiny", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["fits"] is True


def test_plan_die_auto_mux(capsys):
    assert main(["plan", "--model", "llama-3.1-8b", "--process", "n6", "--target", "die", "--weight-bits", "3", "--auto-mux", "--streams", "1"]) == 0
    assert "single die" in capsys.readouterr().out


def test_sweep(capsys):
    assert main(["sweep"]) == 0
    out = capsys.readouterr().out
    assert "whisper-tiny" in out and "whisper-large-v3-turbo" in out


def test_bad_model(capsys):
    assert main(["plan", "--model", "nope"]) == 2


def test_svg_written(tmp_path, capsys):
    out = tmp_path / "fp.svg"
    assert main(["plan", "--model", "base", "--svg", str(out)]) == 0
    assert out.read_text().startswith("<svg")
