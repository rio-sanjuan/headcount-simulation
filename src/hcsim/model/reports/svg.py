"""Plain SVG charts for reports: points and lines on linear axes, with no plotting library.

A chart is a list of `Series` on shared axes, with a legend, axis labels, and optionally the
identity line, for plotting a fit against the truth. The output depends only on the data, so a
report's figures are as reproducible as its tables.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal
from xml.sax.saxutils import escape

WIDTH: Final = 640
HEIGHT: Final = 420
_LEFT, _RIGHT, _TOP, _BOTTOM = 72, 180, 44, 56
PALETTE: Final = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")


class ChartError(ValueError):
    """Data a chart can't draw."""


@dataclass(frozen=True)
class Series:
    name: str
    x: Sequence[float]
    y: Sequence[float]
    style: Literal["points", "line"] = "points"

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y) or not self.x:
            raise ChartError(f"series {self.name!r} needs one y per x, and at least one point")
        if not all(math.isfinite(v) for v in (*self.x, *self.y)):
            raise ChartError(f"series {self.name!r} has values that aren't finite")


def ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values covering [low, high], about `count` of them."""
    step = 10 ** math.floor(math.log10((high - low) / count))
    step *= next(f for f in (1, 2, 5, 10) if (high - low) / (step * f) <= count)
    first = math.ceil(low / step - 1e-9) * step
    out = []
    k = 0
    while first + k * step <= high + step * 1e-9:
        out.append(round(first + k * step, 12))
        k += 1
    return out


def _label(value: float) -> str:
    return f"{value:.3g}"


def _bounds(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    if low == high:
        low, high = low - 0.5, high + 0.5
    pad = (high - low) * 0.05
    return low - pad, high + pad


def chart(
    series: Sequence[Series],
    *,
    title: str,
    x_label: str,
    y_label: str,
    diagonal: bool = False,
) -> str:
    """The series as one SVG chart. With `diagonal`, both axes share a range and the identity
    line is drawn, so points on it are fits that match the truth."""
    if not series:
        raise ChartError("a chart needs at least one series")
    if len(series) > len(PALETTE):
        raise ChartError(f"a chart holds at most {len(PALETTE)} series")
    xs = [float(v) for s in series for v in s.x]
    ys = [float(v) for s in series for v in s.y]
    if diagonal:
        x0, x1 = _bounds(xs + ys)
        y0, y1 = x0, x1
    else:
        (x0, x1), (y0, y1) = _bounds(xs), _bounds(ys)
    plot_w, plot_h = WIDTH - _LEFT - _RIGHT, HEIGHT - _TOP - _BOTTOM

    def px(x: float) -> float:
        return _LEFT + (x - x0) / (x1 - x0) * plot_w

    def py(y: float) -> float:
        return _TOP + plot_h - (y - y0) / (y1 - y0) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif" font-size="12">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="{WIDTH / 2:.1f}" y="24" text-anchor="middle" font-size="14">'
        f"{escape(title)}</text>",
    ]
    for t in ticks(x0, x1):
        parts.append(
            f'<line x1="{px(t):.1f}" y1="{_TOP}" x2="{px(t):.1f}" y2="{_TOP + plot_h}" '
            'stroke="#e6e6e6"/>'
        )
        parts.append(
            f'<text x="{px(t):.1f}" y="{_TOP + plot_h + 16}" text-anchor="middle">'
            f"{_label(t)}</text>"
        )
    for t in ticks(y0, y1):
        parts.append(
            f'<line x1="{_LEFT}" y1="{py(t):.1f}" x2="{_LEFT + plot_w}" y2="{py(t):.1f}" '
            'stroke="#e6e6e6"/>'
        )
        parts.append(
            f'<text x="{_LEFT - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{_label(t)}</text>'
        )
    parts.append(
        f'<rect x="{_LEFT}" y="{_TOP}" width="{plot_w}" height="{plot_h}" fill="none" '
        'stroke="#333333"/>'
    )
    if diagonal:
        parts.append(
            f'<line x1="{px(x0):.1f}" y1="{py(y0):.1f}" x2="{px(x1):.1f}" y2="{py(y1):.1f}" '
            'stroke="#999999" stroke-dasharray="4 3"/>'
        )
    for color, s in zip(PALETTE, series, strict=False):
        points = [(px(float(x)), py(float(y))) for x, y in zip(s.x, s.y, strict=True)]
        if s.style == "line":
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>' for x, y in points]
    for k, (color, s) in enumerate(zip(PALETTE, series, strict=False)):
        y = _TOP + 12 + 18 * k
        parts.append(
            f'<rect x="{WIDTH - _RIGHT + 16}" y="{y - 9}" width="10" height="10" fill="{color}"/>'
        )
        parts.append(f'<text x="{WIDTH - _RIGHT + 32}" y="{y}">{escape(s.name)}</text>')
    parts.append(
        f'<text x="{_LEFT + plot_w / 2:.1f}" y="{HEIGHT - 14}" text-anchor="middle">'
        f"{escape(x_label)}</text>"
    )
    middle = _TOP + plot_h / 2
    parts.append(
        f'<text x="18" y="{middle:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {middle:.1f})">{escape(y_label)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
