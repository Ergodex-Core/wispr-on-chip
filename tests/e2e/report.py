"""Render the e2e results (out/e2e/<set>_<frames>/results.json) as a markdown table for docs/status.md.
Usage: uv run python tests/e2e/report.py default_full long_full"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.data import REPO  # noqa: E402


def main():
    for name in sys.argv[1:]:
        p = REPO / "out" / "e2e" / name / "results.json"
        if not p.exists():
            print(f"(no results for {name})"); continue
        r = json.load(open(p))
        ok = [x for x in r["rows"] if x["status"] == "ok"]
        cyc = [x["cycles"] for x in ok if x.get("cycles")]
        secs = [x["sim_seconds"] for x in ok if x.get("sim_seconds")]
        print(f"\n### e2e set `{name}` — {r['ran']}/{r['n']} clips ran, golden-vs-RTL token identity "
              f"{r['identical']}/{r['ran']}, RTL WER vs ground truth {100 * r['wer']:.2f} %\n")
        print("| clip | CPU fp32 text | RTL text | tokens = golden | text = CPU | WER | cycles | sim wall (s) | cycles/s |")
        print("|---|---|---|---|---|---|---|---|---|")
        for x in r["rows"]:
            if x["status"] != "ok":
                print(f"| {x['uid']} | | | {x['status']} | | | | | |"); continue
            cps = (x["cycles"] / x["sim_seconds"]) if x.get("cycles") and x.get("sim_seconds") else 0
            print(f"| {x['uid']} | {x['cpu_text'].strip()} | {x['rtl_text'].strip()} | {'yes' if x['identical'] else 'NO'} | "
                  f"{'yes' if x['cpu_match'] else 'no'} | {100 * x['wer']:.1f} % | {x['cycles']:,} | {x['sim_seconds']:.0f} | {cps:,.0f} |")
        if cyc:
            print(f"\nMean cycles/clip {sum(cyc) / len(cyc):,.0f}; mean sim wall {sum(secs) / len(secs):,.0f} s.")


if __name__ == "__main__":
    main()
