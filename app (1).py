import io, os, re, zipfile
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import arabic_reshaper
from bidi.algorithm import get_display

from streamlit_drawable_canvas import st_canvas


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


def safe_filename(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(s).strip())


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def arabic_text_fix(text: str) -> str:
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)


def find_any_ttf_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_font_safe(font_path: str, size: int):
    if not font_path or not os.path.exists(font_path):
        raise ValueError(f"❌ Font path not found: {font_path}. Please upload a TTF font.")
    try:
        return ImageFont.truetype(font_path, size)
    except OSError as e:
        raise ValueError(f"❌ Could not open font: {font_path}. Error: {e}")


def fit_text_in_box(draw, text, font_path, box: Box, max_size=140, min_size=20, step=2):
    box_w = box.x2 - box.x1

    for size in range(max_size, min_size - 1, -step):
        font = load_font_safe(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

        if text_w <= box_w:
            x = box.x1 + (box_w - text_w) // 2
            y = box.y1
            return font, (x, y)

    font = load_font_safe(font_path, min_size)
    return font, (box.x1, box.y1)


def extract_box_from_canvas(json_data):
    if not json_data or "objects" not in json_data or not json_data["objects"]:
        return None

    for obj in json_data["objects"]:
        if obj.get("type") == "rect":
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
            height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

            x1 = int(left)
            y1 = int(top)
            x2 = int(left + width)
            y2 = int(top + height)

            if x2 > x1 and y2 > y1:
                return Box(x1, y1, x2, y2)

    return None


def generate_zip(template_bytes, csv_bytes, encoding, name_col, font_bytes, default_font, box, rgb, prefix):
    df = pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding)
    df.columns = df.columns.str.strip()

    if name_col not in df.columns:
        raise ValueError(f"❌ Column '{name_col}' not found. Available columns: {list(df.columns)}")

    names = df[name_col].dropna().astype(str).tolist()
    names = [n.strip() for n in names if str(n).strip()]
    if not names:
        raise ValueError("❌ No names found in CSV.")

    tmp_font = None
    if font_bytes and len(font_bytes) > 0:
        tmp_font = "/tmp/uploaded_font.ttf"
        with open(tmp_font, "wb") as f:
            f.write(font_bytes)
        font_path = tmp_font
    else:
        font_path = default_font

    if not font_path:
        raise ValueError("❌ No default font found. Please upload a TTF font.")

    base = Image.open(io.BytesIO(template_bytes)).convert("RGB")

    zip_buf = io.BytesIO()
    preview = None

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, name in enumerate(names):
            im = base.copy()
            draw = ImageDraw.Draw(im)

            name_to_draw = arabic_text_fix(name)
            font_fit, pos = fit_text_in_box(draw, name_to_draw, font_path, box)
            draw.text(pos, name_to_draw, fill=rgb, font=font_fit)

            out = io.BytesIO()
            im.save(out, format="PNG")
            zf.writestr(f"{prefix}{safe_filename(name)}.png", out.getvalue())

            if i == 0:
                preview = out.getvalue()

    zip_buf.seek(0)

    if tmp_font and os.path.exists(tmp_font):
        os.remove(tmp_font)

    return zip_buf.getvalue(), preview, len(names)


st.set_page_config(page_title="Certificate Generator", page_icon="✅", layout="wide")
st.title("Smart Certificate Automation System")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("Upload")
    template_file = st.file_uploader("Template (PNG/JPG)", type=["png", "jpg", "jpeg"])
    csv_file = st.file_uploader("Names CSV", type=["csv"])
    font_file = st.file_uploader("Font (TTF) - optional (Arabic recommended)", type=["ttf"])

    st.subheader("Settings")
    encoding = st.selectbox("CSV encoding", ["utf-16", "utf-8", "utf-8-sig", "cp1256"], index=0)
    name_col = st.text_input("Name column", value="name")

    st.subheader("Text Color")
    manual_color_hex = st.color_picker("Text color", value="#226622")
    prefix = st.text_input("Filename prefix", value="certificate_")

with c2:
    st.subheader("AI placement (Draw NAME box)")
    st.caption("ارسم مستطيل فوق مكان الاسم مرة وحدة. بعدها النظام بحسب حجم الخط تلقائيًا.")

    default_font = find_any_ttf_font()
    box = None

    if template_file:
        pil_img = Image.open(io.BytesIO(template_file.getvalue())).convert("RGB")

        st.write("✏️ ارسم Rect فوق مكان الاسم:")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#00AA00",
            background_image=pil_img,
            update_streamlit=True,
            height=pil_img.height,
            width=pil_img.width,
            drawing_mode="rect",
            key="canvas",
        )

        box = extract_box_from_canvas(canvas_result.json_data)

        if box:
            st.success(f"✅ NAME_BOX: ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
        else:
            st.warning("ارسم مستطيل واحد فوق مكان الاسم عشان نكمل.")

    st.divider()
    st.subheader("Generate")

    if st.button("🚀 Generate ZIP", use_container_width=True, type="primary"):
        if not template_file or not csv_file:
            st.error("بدنا Template + CSV.")
            st.stop()

        if not box:
            st.error("ارسم مربع الاسم أولاً (NAME_BOX).")
            st.stop()

        if not default_font and not font_file:
            st.error("❌ ما لقيت أي خط TTF. ارفعي خط (TTF).")
            st.stop()

        try:
            zip_bytes, preview, n = generate_zip(
                template_file.getvalue(),
                csv_file.getvalue(),
                encoding,
                name_col.strip(),
                font_file.getvalue() if font_file else None,
                default_font,
                box,
                hex_to_rgb(manual_color_hex),
                prefix.strip() or "certificate_",
            )

            st.success(f"✅ Generated {n} certificates.")

            if preview:
                st.image(preview, caption="Preview (first)", use_container_width=True)

            st.download_button(
                "⬇️ Download ZIP",
                data=zip_bytes,
                file_name="certificates.zip",
                mime="application/zip",
                use_container_width=True
            )

        except Exception as e:
            st.exception(e)
