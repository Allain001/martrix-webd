from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager


def configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "Noto Serif SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "Noto Serif CJK JP",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    resolved = [name for name in preferred_fonts if name in installed]
    if not resolved:
        resolved = [font.name for font in font_manager.fontManager.ttflist if "Noto" in font.name]
    mpl.rcParams["font.family"] = resolved + ["DejaVu Sans"]
    mpl.rcParams["font.sans-serif"] = resolved + ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
