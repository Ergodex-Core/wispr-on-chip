"""Run the Verilator e2e set on Modal: one container per clip (or per shard), results copied back.

  modal run tests/e2e/modal/modal_e2e.py --set default_full [--clips a,b,c] [--cpus 8]

The image is built once (Java + sbt + Verilator + the repo with committed weights, sbt deps and firtool
pre-fetched, Test/compile done). Each call runs E2ESpec on a clip list under its own E2E_DIR and returns
the rtl_tokens.txt / rtl_stats.json files, which the caller writes into out/e2e/<set>/<clip>/.
Requires: MODAL token configured; the repo checked out locally (this file is executed locally).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_here = Path(__file__).resolve()
REPO = _here.parents[3] if len(_here.parents) > 3 and (_here.parents[3] / "build.sbt").exists() else Path("/repo")
APP_NAME = "whisper-si-e2e"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("default-jdk-headless", "verilator", "build-essential", "curl", "git", "ccache", "make", "g++", "perl")
    .run_commands(
        "mkdir -p /opt && cd /opt && curl -sSL -o sbt.tgz https://github.com/sbt/sbt/releases/download/v1.10.7/sbt-1.10.7.tgz && tar xzf sbt.tgz && rm sbt.tgz",
        "verilator --version",
    )
    .env({"PATH": "/opt/sbt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "WHISPER_SI_ROOT": "/repo",
          "SBT_OPTS": "-Xmx6g -Xss64m"})
    .add_local_dir(str(REPO / "src"), "/repo/src", copy=True)
    .add_local_dir(str(REPO / "project"), "/repo/project", copy=True, ignore=["target", "project"])
    .add_local_file(str(REPO / "build.sbt"), "/repo/build.sbt", copy=True)
    .add_local_dir(str(REPO / "weights"), "/repo/weights", copy=True)
    .add_local_dir(str(REPO / "out" / "vectors"), "/repo/out/vectors", copy=True)
    # compile once, fetch firtool once (a tiny elaboration), warm the sbt/coursier caches
    .run_commands(
        "cd /repo && sbt -batch Test/compile",
        "cd /repo && WHISPER_SIM_THREADS=1 MATMUL_CASES=rand_1_5x32x32_raw sbt -batch 'testOnly whisper.MatmulEngineSpec' | tail -3",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(cpu=8.0, memory=16384, timeout=6 * 3600)
def run_clips(set_name: str, clips: list[str], files: dict[str, bytes], threads: int = 4) -> dict[str, dict[str, str]]:
    """files: {"<clip>/mel.hex": bytes, "<clip>/meta.json": bytes, ...}. Returns {clip: {name: text}}."""
    import subprocess

    d = Path("/tmp/e2e") / set_name
    for rel, data in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (d / "clips.txt").write_text("\n".join(clips) + "\n")
    env = dict(os.environ, E2E_DIR=str(d), WHISPER_SIM_THREADS=str(threads), E2E_POLL="4096")
    log = subprocess.run(["sbt", "-batch", "testOnly whisper.E2ESpec"], cwd="/repo", env=env, capture_output=True, text=True)
    out = {}
    for c in clips:
        r = {}
        for name in ("rtl_tokens.txt", "rtl_stats.json"):
            f = d / c / name
            if f.exists():
                r[name] = f.read_text()
        r["log"] = "\n".join(l for l in log.stdout.splitlines() if "tokens," in l or "FAILED" in l or "Error" in l or "Tests:" in l)
        out[c] = r
    return out


@app.local_entrypoint()
def main(set: str = "default_full", clips: str = "", threads: int = 4, per_container: int = 1):
    d = REPO / "out" / "e2e" / set
    all_clips = [l.strip() for l in open(d / "clips.txt") if l.strip()]
    sel = clips.split(",") if clips else all_clips
    groups = [sel[i:i + per_container] for i in range(0, len(sel), per_container)]

    def payload(group):
        files = {}
        for c in group:
            for name in ("mel.hex", "meta.json", "golden_tokens.txt"):
                files[f"{c}/{name}"] = (d / c / name).read_bytes()
        return files

    print(f"launching {len(groups)} containers for {len(sel)} clips of {set}")
    results = list(run_clips.starmap([(set, g, payload(g), threads) for g in groups]))
    for res in results:
        for c, r in res.items():
            for name, txt in r.items():
                if name != "log":
                    (d / c / name).write_text(txt)
            print(c, r.get("log", "")[:300])
    print("done; now run: uv run python tests/e2e/run_e2e.py --set", set.replace("_full", ""), "--frames full --compare-only")
