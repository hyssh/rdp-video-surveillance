"""Pillow renderer for fake Windows-style RDP frames (1280x720).

Every drawn text/button/field is also recorded as a UI element with a 0-1 bbox, so the
element overlay in the UI lines up with the picture (same shape OmniParser returns).
"""
import os
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
TASKBAR_H = 48

_SANS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
_BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
_MONO = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Supplemental/Courier New.ttf", "C:/Windows/Fonts/consola.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"]
_FONTS: Dict[Tuple[str, int], Any] = {}


def font(size: int, kind: str = "sans"):
    key = (kind, size)
    if key not in _FONTS:
        loaded = None
        for path in {"sans": _SANS, "bold": _BOLD, "mono": _MONO}[kind]:
            if os.path.exists(path):
                try:
                    loaded = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
        _FONTS[key] = loaded or ImageFont.load_default(size)
    return _FONTS[key]


COLORS = {"red": (241, 76, 76), "green": (35, 209, 139), "yellow": (229, 229, 16), "cyan": (41, 184, 219), "gray": (140, 140, 140)}
FG_LIGHT, FG_DARK = (27, 27, 27), (204, 204, 204)
BLUE = (0, 103, 192)


def _fit(text: str, fnt, max_w: int) -> str:
    if fnt.getlength(text) <= max_w:
        return text
    while text and fnt.getlength(text + "...") > max_w:
        text = text[:-1]
    return text + "..."


def font_height(fnt) -> int:
    return int(sum(fnt.getmetrics()))


class Scene:
    def __init__(self) -> None:
        self.img = Image.new("RGB", (W, H))
        self.d = ImageDraw.Draw(self.img)
        self.elements: List[Dict[str, Any]] = []

    def record(self, kind: str, content: str, box: Tuple[float, float, float, float], interactive: bool) -> None:
        x1, y1, x2, y2 = box
        bbox = [round(max(0.0, min(1.0, v)), 4) for v in (x1 / W, y1 / H, x2 / W, y2 / H)]
        self.elements.append({"type": kind, "content": content, "interactivity": interactive, "bbox": bbox})

    def text(self, x: int, y: int, s: str, fnt, fill, max_w: int = 0, record: bool = True, interactive: bool = False, kind: str = "text") -> int:
        if max_w:
            s = _fit(s, fnt, max_w)
        self.d.text((x, y), s, font=fnt, fill=fill)
        if record and s.strip():
            x1, y1, x2, y2 = self.d.textbbox((x, y), s, font=fnt)
            self.record(kind, s, (x1 - 2, y1 - 2, x2 + 2, y2 + 2), interactive)
        return font_height(fnt)


def draw_desktop(sc: Scene) -> None:
    for y in range(H):
        t = y / H
        sc.d.line([(0, y), (W, y)], fill=(int(14 + 30 * t), int(70 + 70 * t), int(150 + 60 * t)))
    sc.d.ellipse((760, 80, 1260, 580), fill=(30, 105, 190))
    for i, label in enumerate(["Recycle Bin", "Microsoft Edge", "Notes"]):
        y = 24 + i * 88
        sc.d.rounded_rectangle((34, y, 72, y + 38), 6, fill=[(220, 220, 225), (40, 160, 220), (250, 210, 70)][i])
        sc.text(18, y + 44, label, font(12), (255, 255, 255), max_w=90, record=False)
        sc.record("icon", f"{label} desktop icon", (30, y - 2, 76, y + 60), True)


def draw_taskbar(sc: Scene, active_title: str, clock: str, date: str) -> None:
    top = H - TASKBAR_H
    sc.d.rectangle((0, top, W, H), fill=(28, 30, 42))
    for dx in (14, 25):
        for dy in (14, 25):
            sc.d.rectangle((dx, top + dy, dx + 8, top + dy + 8), fill=(0, 160, 240))
    sc.record("icon", "Start button", (10, top + 10, 38, top + 38), True)
    sc.d.rounded_rectangle((56, top + 8, 250, top + 40), 16, fill=(52, 55, 70))
    sc.text(76, top + 14, "Search", font(14), (190, 190, 200), record=False)
    sc.record("icon", "Taskbar search box", (56, top + 8, 250, top + 40), True)
    for i, (label, color) in enumerate([("File Explorer", (250, 200, 60)), ("Microsoft Edge", (40, 170, 230)), ("Terminal", (120, 120, 130))]):
        x = 270 + i * 46
        sc.d.rounded_rectangle((x, top + 10, x + 30, top + 38), 6, fill=color)
        sc.record("icon", f"{label} taskbar icon", (x - 2, top + 8, x + 32, top + 40), True)
    sc.d.rounded_rectangle((420, top + 6, 700, top + 42), 6, fill=(70, 74, 96))
    sc.d.line([(432, top + 41), (688, top + 41)], fill=(96, 165, 250), width=3)
    sc.text(432, top + 14, active_title, font(14), (255, 255, 255), max_w=250, interactive=True, kind="icon")
    sc.text(W - 118, top + 5, clock, font(14), (255, 255, 255))
    sc.text(W - 118, top + 24, date, font(13), (210, 210, 220))


def _body_color(theme: str):
    return ((12, 12, 12), FG_DARK) if theme == "dark" else ((255, 255, 255), FG_LIGHT)


def draw_body(sc: Scene, rect: Tuple[int, int, int, int], y: int, theme: str, items: List[Tuple]) -> None:
    x1, _, x2, y2 = rect
    _, fg = _body_color(theme)
    pad = 22
    inner_w = x2 - x1 - 2 * pad
    for item in items:
        kind = item[0]
        if y > y2 - 30:
            break
        if kind == "sp":
            y += item[1]
        elif kind == "h":
            y += sc.text(x1 + pad, y, item[1], font(24, "bold"), fg, max_w=inner_w) + 8
        elif kind == "t":
            color = COLORS.get(item[2], fg) if len(item) > 2 else fg
            y += sc.text(x1 + pad, y, item[1], font(16), color, max_w=inner_w) + 6
        elif kind == "b":
            y += sc.text(x1 + pad, y, item[1], font(16, "bold"), fg, max_w=inner_w) + 6
        elif kind == "code":
            color = COLORS.get(item[2], fg) if len(item) > 2 else fg
            y += sc.text(x1 + pad, y, item[1], font(15, "mono"), color, max_w=inner_w) + 4
        elif kind == "field":
            label, value = item[1], item[2]
            sc.text(x1 + pad, y + 4, label, font(16), fg, max_w=190)
            bx1, bx2 = x1 + pad + 200, min(x2 - pad, x1 + pad + 200 + 420)
            sc.d.rectangle((bx1, y, bx2, y + 30), fill=(255, 255, 255) if theme == "light" else (40, 40, 40), outline=(140, 140, 140))
            sc.text(bx1 + 8, y + 5, value, font(16), FG_LIGHT if theme == "light" else FG_DARK, max_w=bx2 - bx1 - 16, record=False)
            sc.record("icon", f"{label} input: {value}", (bx1, y, bx2, y + 30), True)
            y += 40
        elif kind == "btns":
            labels = item[1]
            left = len(item) > 2 and item[2] == "left"
            widths = [int(font(15).getlength(s)) + 40 for s in labels]
            bx = x1 + pad if left else x2 - pad - sum(widths) - 10 * (len(labels) - 1)
            for i, (label, bw) in enumerate(zip(labels, widths)):
                fill = BLUE if i == 0 else ((225, 225, 225) if theme == "light" else (60, 60, 60))
                sc.d.rounded_rectangle((bx, y, bx + bw, y + 32), 4, fill=fill)
                sc.d.text((bx + 20, y + 7), label, font=font(15), fill=(255, 255, 255) if i == 0 else fg)
                sc.record("icon", f"{label} button", (bx, y, bx + bw, y + 32), True)
                bx += bw + 10
            y += 44
        elif kind in ("row", "hrow"):
            cells = item[1]
            widths = item[2] if len(item) > 2 and item[2] else [100 // len(cells)] * len(cells)
            selected = kind == "row" and len(item) > 3 and item[3]
            if selected:
                sc.d.rectangle((x1 + pad - 6, y - 2, x2 - pad + 6, y + 24), fill=(204, 228, 247) if theme == "light" else (38, 79, 120))
            cx = x1 + pad
            for cell, pct in zip(cells, widths):
                cw = int(inner_w * pct / 100)
                sc.text(cx, y, str(cell), font(15, "bold" if kind == "hrow" else "sans"), fg, max_w=cw - 10, record=False)
                cx += cw
            sc.record("text", " | ".join(str(c) for c in cells), (x1 + pad, y - 2, x2 - pad, y + 24), False)
            if kind == "hrow":
                sc.d.line([(x1 + pad - 6, y + 26), (x2 - pad + 6, y + 26)], fill=(180, 180, 180))
                y += 4
            y += 28
        elif kind == "toggle":
            label, on = item[1], item[2]
            sc.text(x1 + pad, y + 2, label, font(16), fg, max_w=inner_w - 120)
            tx = x1 + pad + min(inner_w - 70, 520)
            sc.d.rounded_rectangle((tx, y, tx + 46, y + 24), 12, fill=(0, 140, 70) if on else (150, 150, 150))
            kx = tx + 24 if on else tx + 2
            sc.d.ellipse((kx, y + 2, kx + 20, y + 22), fill=(255, 255, 255))
            sc.record("icon", f"{label} toggle {'On' if on else 'Off'}", (tx, y, tx + 46, y + 24), True)
            y += 38
        elif kind == "progress":
            label, pct = item[1], item[2]
            shown = label if "%" in label else f"{label} {pct}%"
            y += sc.text(x1 + pad, y, shown, font(16), fg, max_w=inner_w) + 4
            bw = min(inner_w, 520)
            sc.d.rectangle((x1 + pad, y, x1 + pad + bw, y + 16), outline=(140, 140, 140), fill=(235, 235, 235) if theme == "light" else (45, 45, 45))
            sc.d.rectangle((x1 + pad, y, x1 + pad + int(bw * pct / 100), y + 16), fill=(35, 160, 70))
            sc.record("icon", f"progress bar {pct}%", (x1 + pad, y, x1 + pad + bw, y + 16), False)
            y += 28


def draw_window(sc: Scene, rect: Tuple[int, int, int, int], title: str, theme: str, toolbar: str = "", body: List[Tuple] = ()) -> None:
    x1, y1, x2, y2 = rect
    sc.d.rectangle((x1 + 8, y1 + 8, x2 + 8, y2 + 8), fill=(10, 40, 90))
    bg, _ = _body_color(theme)
    sc.d.rectangle(rect, fill=bg, outline=(90, 90, 100))
    bar = (32, 32, 32) if theme == "dark" else (240, 243, 249)
    barfg = (255, 255, 255) if theme == "dark" else FG_LIGHT
    sc.d.rectangle((x1, y1, x2, y1 + 34), fill=bar)
    sc.d.rounded_rectangle((x1 + 10, y1 + 9, x1 + 26, y1 + 25), 3, fill=BLUE)
    sc.text(x1 + 36, y1 + 8, title, font(15), barfg, max_w=x2 - x1 - 190)
    for i, name in enumerate(["Minimize", "Maximize", "Close"]):
        bx = x2 - 138 + i * 46
        cx, cy = bx + 23, y1 + 17
        if name == "Minimize":
            sc.d.line([(cx - 5, cy), (cx + 5, cy)], fill=barfg, width=1)
        elif name == "Maximize":
            sc.d.rectangle((cx - 5, cy - 5, cx + 5, cy + 5), outline=barfg)
        else:
            sc.d.line([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill=barfg, width=1)
            sc.d.line([(cx - 5, cy + 5), (cx + 5, cy - 5)], fill=barfg, width=1)
        sc.record("icon", f"{name} window button", (bx, y1, bx + 46, y1 + 34), True)
    y = y1 + 34
    if toolbar:
        sc.d.rectangle((x1, y, x2, y + 40), fill=(248, 248, 250) if theme == "light" else (45, 45, 45))
        sc.d.rounded_rectangle((x1 + 14, y + 6, x2 - 14, y + 34), 14, fill=(255, 255, 255) if theme == "light" else (30, 30, 30), outline=(190, 190, 195))
        sc.text(x1 + 30, y + 10, toolbar, font(15), FG_LIGHT if theme == "light" else FG_DARK, max_w=x2 - x1 - 70, kind="icon", interactive=True)
        y += 40
    draw_body(sc, rect, y + 18, theme, list(body))


def draw_dialog(sc: Scene, spec: Dict[str, Any]) -> None:
    w, h = spec.get("size", (560, 250))
    rect = ((W - w) // 2, (H - TASKBAR_H - h) // 2, (W + w) // 2, (H - TASKBAR_H + h) // 2)
    draw_window(sc, rect, spec["title"], "light", "", spec["body"])


def draw_toast(sc: Scene, title: str, message: str) -> None:
    box = (W - 380, H - TASKBAR_H - 110, W - 20, H - TASKBAR_H - 16)
    sc.d.rounded_rectangle(box, 8, fill=(44, 46, 60), outline=(90, 92, 110))
    sc.text(box[0] + 16, box[1] + 14, title, font(15, "bold"), (255, 255, 255), max_w=330)
    sc.text(box[0] + 16, box[1] + 42, message, font(14), (210, 210, 220), max_w=330)


def render_frame(spec: Dict[str, Any], clock: str, date: str) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    sc = Scene()
    draw_desktop(sc)
    win = spec["win"]
    rect = tuple(win.get("rect") or (110, 40, 1170, 640))
    draw_window(sc, rect, win["title"], win.get("theme", "light"), win.get("toolbar", ""), win["body"])
    if spec.get("dialog"):
        draw_dialog(sc, spec["dialog"])
    if spec.get("toast"):
        draw_toast(sc, *spec["toast"])
    draw_taskbar(sc, spec["dialog"]["title"] if spec.get("dialog") else win["title"], clock, date)
    return sc.img, sc.elements


def save_png(img: Image.Image, path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
    return os.path.getsize(path)
