"""Collect Yosys/ABC results from /home/user/synth/<TOP>/{stat.txt,yosys.log} and write
docs/paper/synth_numbers.tex + a markdown table for docs/status.md."""
import re, sys, os, glob

ROOT = "/home/user/synth"
ORDER = ["SystolicArray", "SystolicArray4", "MatmulEngine", "VectorUnit", "Attention", "KVCache", "Sequencer", "Sampler", "WeightStore"]
LABEL = {"SystolicArray": "Systolic array (32$\\times$32 MAC, 8 rows/stage)", "SystolicArray4": "\\quad variant: 4 rows/stage (not in total)", "MatmulEngine": "Matmul engine (array + requant + control)",
         "VectorUnit": "Vector unit (16 lanes)", "Attention": "Attention (softmax, divider)", "KVCache": "KV cache logic (transposer)",
         "Sequencer": "Sequencer", "Sampler": "Sampler", "WeightStore": "Weight store mux (247 banks)"}

def parse(top):
    st = os.path.join(ROOT, top, "stat.txt"); lg = os.path.join(ROOT, top, "yosys.log")
    if not (os.path.exists(st) and os.path.exists(lg)): return None
    s = open(st).read()
    area = float(re.search(r"Chip area for (?:top )?module.*?:\s*([\d.]+)", s).group(1))
    cells = int(re.search(r"Number of cells:\s*(\d+)", s).group(1))
    flops = sum(int(n) for c, n in re.findall(r"^\s+\$?\\?(\S*DFF\S*)\s+(\d+)\s*$", s, re.M))
    ds = re.findall(r"Delay =\s*([\d.]+)\s*ps", open(lg, errors="ignore").read())
    delay = float(ds[-1]) if ds else float("nan")
    return dict(area_um2=area, cells=cells, flops=flops, delay_ps=delay)

res = {t: parse(t) for t in ORDER}
rows = []; md = ["| block | cells | flops | area (mm²) | critical path (ps) | est. Fmax (GHz) |", "|---|---|---|---|---|---|"]
tot_area = 0.0
for t in ORDER:
    r = res[t]
    if not r: continue
    fmax = 1000.0 / (r["delay_ps"] + 60.0)   # + setup/clk-to-q allowance
    rows.append(f"{LABEL[t]} & \\num{{{r['cells']}}} & \\num{{{r['flops']}}} & {r['area_um2']/1e6:.3f} & {r['delay_ps']:.0f} \\\\")
    md.append(f"| {t} | {r['cells']:,} | {r['flops']:,} | {r['area_um2']/1e6:.3f} | {r['delay_ps']:.0f} | {fmax:.2f} |")
    if t not in ("SystolicArray", "SystolicArray4"): tot_area += r["area_um2"]   # array is inside MatmulEngine
sa = res.get("SystolicArray"); allmax = max((r["delay_ps"] for t, r in res.items() if r and t != "SystolicArray4"), default=float("nan"))
def fmax(d): return f"\\SI{{{1000.0/(d+60.0):.2f}}}{{GHz}}"
tex = [f"\\newcommand{{\\areaLogicTotal}}{{{tot_area/1e6:.2f}}}",
       f"\\newcommand{{\\fmaxSA}}{{{fmax(sa['delay_ps']) if sa else 'TBD'}}}",
       f"\\newcommand{{\\fmaxAll}}{{{fmax(allmax)}}}",
       "\\newcommand{\\synthRows}{" + "\n".join(rows) + "}",
       "\\newcommand{\\synthDiscussion}{" + (open(sys.argv[1]).read().strip() if len(sys.argv) > 1 else "") + "}"]
open("/home/user/wispr-on-chip/docs/paper/synth_numbers.tex", "w").write("\n".join(tex) + "\n")
print("\n".join(md)); print(f"logic total (excl. array double count): {tot_area/1e6:.3f} mm^2 ; slowest path {allmax:.0f} ps")
