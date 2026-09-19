"""Render the run charts as SVG, straight from the loop's own state files.

    python3 tools/render_charts.py                    # all runs found
    python3 tools/render_charts.py --runs final_01

Writes docs/img/<run>-attempts-{light,dark}.svg and
       docs/img/<run>-tree-{light,dark}.svg

No dependencies: numpy is the project's only runtime requirement and charts are
not runtime. Two files per chart rather than one theme-aware file, because the
README pairs them in a <picture> element - which is the only mechanism GitHub
honours for light/dark image switching.

Palette and mark specs follow the validated reference palette: categorical slots
1-3 for the stage ribbon (validated all-pairs in both modes), status-critical for
failures, >=8px markers with a 2px surface ring, hairline solid gridlines.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

THEME = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        draft="#2a78d6", improve="#eb6834", debug="#1baf7a",
        critical="#d03b3b", dot="#2a78d6", dim="#c3c2b7"),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        draft="#3987e5", improve="#d95926", debug="#199e70",
        critical="#d03b3b", dot="#3987e5", dim="#383835"),
}
FONT = "system-ui,-apple-system,'Segoe UI',sans-serif"
BASELINE = 0.6016          # FM baseline on validation


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def load(run):
    p = ROOT / "work" / "runs" / run / "state.json"
    return json.loads(p.read_text())["nodes"]


# --------------------------------------------------------------- attempts chart
def attempts_svg(run, nodes, mode):
    c = THEME[mode]
    n = len(nodes)
    W, H = 900, 460
    L, R, T, B = 62, 22, 54, 118         # margins; B holds ribbon + legend
    pw, ph = W - L - R, H - T - B

    lo, hi = 0.45, 0.615
    def x(i):  return L + (pw * (i + 0.5) / n)
    def y(v):  return T + ph * (1 - (v - lo) / (hi - lo))

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" role="img" '
         f'aria-label="Validation score per attempt for run {esc(run)}">',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>']

    s.append(f'<text x="{L}" y="24" font-family="{FONT}" font-size="15" '
             f'font-weight="600" fill="{c["ink"]}">Validation score per attempt '
             f'&#8212; {esc(run)}</text>')
    scored = [d for d in nodes if d.get("score") is not None]
    s.append(f'<text x="{L}" y="42" font-family="{FONT}" font-size="12" '
             f'fill="{c["ink2"]}">{len(scored)} of {n} attempts produced a '
             f'score. Stage shown in the band below.</text>')

    # gridlines + y ticks
    for v in (0.45, 0.50, 0.55, 0.60):
        yy = round(y(v), 1)
        s.append(f'<line x1="{L}" y1="{yy}" x2="{L+pw}" y2="{yy}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{L-8}" y="{yy+4}" text-anchor="end" '
                 f'font-family="{FONT}" font-size="11" fill="{c["muted"]}" '
                 f'style="font-variant-numeric:tabular-nums">{v:.2f}</text>')

    # FM baseline reference line, directly labelled
    yb = round(y(BASELINE), 1)
    s.append(f'<line x1="{L}" y1="{yb}" x2="{L+pw}" y2="{yb}" '
             f'stroke="{c["ink2"]}" stroke-width="1"/>')
    s.append(f'<text x="{L+pw}" y="{yb-7}" text-anchor="end" '
             f'font-family="{FONT}" font-size="11" font-weight="600" '
             f'fill="{c["ink2"]}">FM baseline {BASELINE:.4f}</text>')

    # failure ticks: a picket fence along the floor makes a dead zone visible
    fy = T + ph + 10
    for i, d in enumerate(nodes):
        if d.get("score") is None:
            s.append(f'<line x1="{round(x(i),1)}" y1="{fy}" '
                     f'x2="{round(x(i),1)}" y2="{fy+13}" '
                     f'stroke="{c["critical"]}" stroke-width="2"/>')

    # scored attempts: one series, so position carries it and no legend is owed
    best = max(scored, key=lambda d: d["score"]) if scored else None
    for i, d in enumerate(nodes):
        v = d.get("score")
        if v is None:
            continue
        s.append(f'<circle cx="{round(x(i),1)}" cy="{round(y(v),1)}" r="4.5" '
                 f'fill="{c["dot"]}" stroke="{c["surface"]}" '
                 f'stroke-width="2"/>')
    if best:
        bi = nodes.index(best)
        bx, by = x(bi), y(best["score"])
        # keep the value label clear of the baseline caption in the far corner
        anchor = "end" if bx > L + pw - 60 else "middle"
        s.append(f'<text x="{round(bx,1)}" y="{round(by-12,1)}" '
                 f'text-anchor="{anchor}" font-family="{FONT}" font-size="11" '
                 f'font-weight="600" fill="{c["ink"]}" '
                 f'style="font-variant-numeric:tabular-nums">'
                 f'{best["score"]:.4f}</text>')

    # x ticks, every 5th attempt
    for i in range(0, n, 5):
        s.append(f'<text x="{round(x(i),1)}" y="{fy+30}" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="10" fill="{c["muted"]}">'
                 f'{i}</text>')

    # stage ribbon: 2px surface gaps do the separating, not strokes
    ry, rh = fy + 38, 9
    for i, d in enumerate(nodes):
        col = c[d["stage"]]
        x0 = L + pw * i / n
        w = pw / n - 2
        s.append(f'<rect x="{round(x0,1)}" y="{ry}" width="{round(max(w,1),1)}" '
                 f'height="{rh}" rx="1.5" fill="{col}"/>')

    s.append(f'<text x="{L+pw/2}" y="{ry+rh+16}" text-anchor="middle" '
             f'font-family="{FONT}" font-size="11" fill="{c["ink2"]}">'
             f'attempt</text>')

    # legend: always present for the ribbon's three categories, plus the
    # failure mark. Light-mode aqua is sub-3:1, so these labels are the relief.
    lx = L
    for label, col in (("draft", c["draft"]), ("improve", c["improve"]),
                       ("debug", c["debug"])):
        s.append(f'<rect x="{lx}" y="{H-22}" width="10" height="10" rx="2" '
                 f'fill="{col}"/>')
        s.append(f'<text x="{lx+15}" y="{H-13}" font-family="{FONT}" '
                 f'font-size="11" fill="{c["ink2"]}">{label}</text>')
        lx += 26 + 7 * len(label)
    s.append(f'<line x1="{lx+3}" y1="{H-23}" x2="{lx+3}" y2="{H-11}" '
             f'stroke="{c["critical"]}" stroke-width="2"/>')
    s.append(f'<text x="{lx+11}" y="{H-13}" font-family="{FONT}" '
             f'font-size="11" fill="{c["ink2"]}">failed (no score)</text>')

    s.append("</svg>")
    return "\n".join(s)


# ------------------------------------------------------------------ tree chart
def tree_svg(run, nodes, mode):
    """Node-link view of the solution tree. Depth is x, siblings stack in y."""
    c = THEME[mode]
    by_id = {d["id"]: d for d in nodes}
    kids = {}
    roots = []
    for d in nodes:
        p = d["parent_id"]
        if p is None:
            roots.append(d["id"])
        else:
            kids.setdefault(p, []).append(d["id"])

    depth, order = {}, []

    def walk(i, dep):
        depth[i] = dep
        order.append(i)
        for k in kids.get(i, []):
            walk(k, dep + 1)

    for r in roots:
        walk(r, 0)

    rows = {i: n for n, i in enumerate(order)}
    maxdep = max(depth.values()) if depth else 0
    W = 900
    RH = 19
    T, B, L = 62, 58, 40
    H = T + B + RH * len(order)
    colw = (W - L - 210) / max(maxdep + 1, 1)

    def px(i): return L + depth[i] * colw
    def py(i): return T + rows[i] * RH + RH / 2

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" role="img" '
         f'aria-label="Solution tree for run {esc(run)}">',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>']
    s.append(f'<text x="{L}" y="24" font-family="{FONT}" font-size="15" '
             f'font-weight="600" fill="{c["ink"]}">Solution tree '
             f'&#8212; {esc(run)}</text>')
    widest = max(((len(v), k) for k, v in kids.items()), default=(0, None))
    sub = (f'Depth is left to right. Node {widest[1]} carries {widest[0]} '
           f'children.') if widest[0] > 3 else 'Depth is left to right.'
    s.append(f'<text x="{L}" y="42" font-family="{FONT}" font-size="12" '
             f'fill="{c["ink2"]}">{esc(sub)}</text>')

    # edges first, so nodes sit on top of them
    for p, cs in kids.items():
        for k in cs:
            x0, y0, x1, y1 = px(p) + 5, py(p), px(k) - 5, py(k)
            mx = (x0 + x1) / 2
            s.append(f'<path d="M{x0:.1f},{y0:.1f} C{mx:.1f},{y0:.1f} '
                     f'{mx:.1f},{y1:.1f} {x1:.1f},{y1:.1f}" fill="none" '
                     f'stroke="{c["dim"]}" stroke-width="1"/>')

    best = max((d for d in nodes if d.get("score") is not None),
               key=lambda d: d["score"], default=None)
    for i in order:
        d = by_id[i]
        v = d.get("score")
        fill = c["dot"] if v is not None else c["critical"]
        s.append(f'<circle cx="{px(i):.1f}" cy="{py(i):.1f}" r="4.5" '
                 f'fill="{fill}" stroke="{c["surface"]}" stroke-width="2"/>')
        lbl = f'#{i} {d["stage"]}'
        if v is not None:
            lbl += f'  {v:.4f}'
        weight = "600" if best and i == best["id"] else "400"
        ink = c["ink"] if best and i == best["id"] else c["ink2"]
        if best and i == best["id"]:
            lbl += "   best"
        s.append(f'<text x="{px(i)+11:.1f}" y="{py(i)+4:.1f}" '
                 f'font-family="{FONT}" font-size="11" font-weight="{weight}" '
                 f'fill="{ink}" stroke="{c["surface"]}" stroke-width="3" '
                 f'paint-order="stroke fill" '
                 f'style="font-variant-numeric:tabular-nums">'
                 f'{esc(lbl)}</text>')

    s.append(f'<rect x="{L}" y="{H-20}" width="10" height="10" rx="5" '
             f'fill="{c["dot"]}"/>')
    s.append(f'<text x="{L+15}" y="{H-11}" font-family="{FONT}" font-size="11" '
             f'fill="{c["ink2"]}">scored</text>')
    s.append(f'<rect x="{L+80}" y="{H-20}" width="10" height="10" rx="5" '
             f'fill="{c["critical"]}"/>')
    s.append(f'<text x="{L+95}" y="{H-11}" font-family="{FONT}" '
             f'font-size="11" fill="{c["ink2"]}">failed</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    a = ap.parse_args()
    runs = a.runs or sorted(
        p.parent.name for p in (ROOT / "work" / "runs").glob("*/state.json")
        if p.parent.name.startswith("final_"))
    OUT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        nodes = load(run)
        for mode in ("light", "dark"):
            for name, fn in (("attempts", attempts_svg), ("tree", tree_svg)):
                p = OUT / f"{run}-{name}-{mode}.svg"
                p.write_text(fn(run, nodes, mode))
                print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
