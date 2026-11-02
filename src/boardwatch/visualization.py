from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from boardwatch.constants import CLASS_NAMES

COLORS = {
    "empty_cell": "#18a999",
    "white_circle_cell": "#2463eb",
    "black_cross_cell": "#e34b4b",
}
SHORT_LABELS = {"empty_cell": "E", "white_circle_cell": "O", "black_cross_cell": "X"}


def draw_detections(image: Image.Image, detections: list[dict], line_width: int | None = None) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font_size = max(9, min(13, round(min(output.size) / 42)))
    width = line_width or max(1, round(min(output.size) / 320))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()
    for item in detections:
        box = [round(value) for value in item["box"]]
        label = str(item["label"])
        if label not in CLASS_NAMES:
            continue
        color = COLORS[label]
        draw.rectangle(box, outline=color, width=width)
        confidence = round(float(item.get("score", 1.0)) * 100)
        title = f"{SHORT_LABELS[label]} {confidence}"
        text_y = max(0, box[1] - font_size - 3)
        left, top, right, bottom = draw.textbbox((box[0], text_y), title, font=font)
        draw.rectangle((left, top, right + 3, bottom + 2), fill=color)
        draw.text((left + 1, top), title, fill="white", font=font)
    return output
