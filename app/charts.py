"""Small server-side SVG charts (no JavaScript, no chart library): SHAP waterfall, LIME bars, trend lines."""
from __future__ import annotations

from html import escape

POS, NEG = "var(--pos)", "var(--neg)"


def _fmt(v: float) -> str:
    return f"{v:+.3f}"


def waterfall(features, shap_vals, values, base: float, top: int = 10, width: int = 640) -> str:
    """SHAP waterfall: start at the model's average output, add each feature's push, end at this prediction."""
    idx = sorted(range(len(shap_vals)), key=lambda i: -abs(shap_vals[i]))[:top]
    rest = sum(v for i, v in enumerate(shap_vals) if i not in idx)
    items = [(f"{features[i]} = {values[i]:.3g}", shap_vals[i]) for i in idx]
    if abs(rest) > 1e-9:
        items.append((f"{len(shap_vals) - len(idx)} other features", rest))
    final = base + sum(shap_vals)
    path = [base] + [base + sum(v for _, v in items[:k + 1]) for k in range(len(items))]
    lo, hi = min(path), max(path)
    pad_v = (hi - lo) * 0.08 or 0.05
    lo, hi = lo - pad_v, hi + pad_v
    lab_w, pad, row = 250, 10, 24
    plot_w = width - lab_w - 70
    h = (len(items) + 2) * row + 20
    x = lambda v: lab_w + (v - lo) / (hi - lo or 1) * plot_w
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="SHAP waterfall">']
    y = 10
    out.append(f'<text x="{lab_w - 8}" y="{y + 15}" text-anchor="end" class="lbl">average output E[f(x)]</text>'
               f'<line x1="{x(base)}" x2="{x(base)}" y1="{y}" y2="{h - 10}" class="guide"/>'
               f'<text x="{x(base) + 4}" y="{y + 15}" class="val">{base:.3f}</text>')
    cur = base
    for name, v in items:
        y += row
        a, b = sorted((x(cur), x(cur + v)))
        col = POS if v > 0 else NEG
        out.append(f'<text x="{lab_w - 8}" y="{y + 15}" text-anchor="end" class="lbl">{escape(name)}</text>'
                   f'<rect x="{a:.1f}" y="{y + 3}" width="{max(b - a, 1.5):.1f}" height="{row - 8}" fill="{col}" rx="2"/>'
                   f'<text x="{b + 4:.1f}" y="{y + 15}" class="val">{_fmt(v)}</text>')
        cur += v
    y += row
    out.append(f'<text x="{lab_w - 8}" y="{y + 15}" text-anchor="end" class="lbl strong">this prediction f(x)</text>'
               f'<line x1="{x(final)}" x2="{x(final)}" y1="{y}" y2="{y + row - 4}" class="final"/>'
               f'<text x="{x(final) + 4}" y="{y + 15}" class="val strong">{final:.3f}</text></svg>')
    return "".join(out)


def bars(features, weights, top: int = 10, width: int = 640, title: str = "LIME") -> str:
    idx = sorted(range(len(weights)), key=lambda i: -abs(weights[i]))[:top]
    m = max([abs(weights[i]) for i in idx] + [1e-9])
    lab_w, row, gutter = 230, 24, 58
    half = (width - lab_w - 2 * gutter) / 2
    mid = lab_w + gutter + half
    h = len(idx) * row + 20
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="{escape(title)} weights">',
           f'<line x1="{mid}" x2="{mid}" y1="4" y2="{h - 6}" class="guide"/>']
    for k, i in enumerate(idx):
        y = 8 + k * row
        v = weights[i]
        w = abs(v) / m * half
        xa = mid if v > 0 else mid - w
        col = POS if v > 0 else NEG
        tx = mid + w + 4 if v > 0 else mid - w - 4
        anchor = "start" if v > 0 else "end"
        out.append(f'<text x="{lab_w - 8}" y="{y + 15}" text-anchor="end" class="lbl">{escape(features[i])}</text>'
                   f'<rect x="{xa:.1f}" y="{y + 3}" width="{max(w, 1.5):.1f}" height="{row - 8}" fill="{col}" rx="2"/>'
                   f'<text x="{tx:.1f}" y="{y + 15}" text-anchor="{anchor}" class="val">{v:+.4f}</text>')
    out.append("</svg>")
    return "".join(out)


def trend(points: list[tuple[str, float]], width: int = 640, height: int = 160, pct: bool = True,
          label: str = "") -> str:
    if not points:
        return '<p class="muted">No retraining runs yet.</p>'
    vals = [v for _, v in points]
    lo, hi = min(vals + [0]), max(vals + [1e-9])
    pad_l, pad_b = 46, 26
    pw, ph = width - pad_l - 20, height - pad_b - 14
    n = len(points)
    X = lambda i: pad_l + (i / max(n - 1, 1)) * pw
    Y = lambda v: 10 + ph - (v - lo) / ((hi - lo) or 1) * ph
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, (_, v) in enumerate(points))
    f = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.3f}")
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{escape(label)}">',
           f'<line x1="{pad_l}" x2="{width - 20}" y1="{10 + ph}" y2="{10 + ph}" class="guide"/>',
           f'<text x="{pad_l - 6}" y="{Y(hi) + 4}" text-anchor="end" class="val">{f(hi)}</text>',
           f'<text x="{pad_l - 6}" y="{Y(lo) + 4}" text-anchor="end" class="val">{f(lo)}</text>',
           f'<path d="{path}" fill="none" stroke="var(--accent)" stroke-width="2"/>']
    for i, (name, v) in enumerate(points):
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" fill="var(--accent)"><title>{escape(name)}: '
                   f'{f(v)}</title></circle><text x="{X(i):.1f}" y="{height - 6}" text-anchor="middle" '
                   f'class="val">{escape(name)}</text>')
    out.append("</svg>")
    return "".join(out)
