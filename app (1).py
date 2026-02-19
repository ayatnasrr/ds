import io, os, re, zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import cv2
import pytesseract

import arabic_reshaper
from bidi.algorithm import get_display


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
    return ImageFont.truetype(font_path, size)


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


def ocr_words_with_boxes(pil_img: Image.Image):
    """
    Returns list of dicts: {text, x1,y1,x2,y2, conf}
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # تحسين بسيط
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    data = pytesseract.image_to_data(thr, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = float(data["conf"][i]) if str(data["conf"][i]).strip() != "-1" else -1
        if txt and conf >= 30:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append({"text": txt, "x1": x, "y1": y, "x2": x+w, "y2": y+h, "conf": conf})
    return out


def find_name_box_ai(pil_img: Image.Image):
    """
    Heuristic:
    - Find anchor words like: Name, Student, Participant, الطالب, اسم, المتدرب
    - Then pick a likely name region near that anchor (to the left/right/below depending)
    Returns Box or None
    """
    words = ocr_words_with_boxes(pil_img)
    if not words:
        return None

    anchors = {"name", "student", "participant", "trainee", "الطالب", "اسم", "المتدرب", "الطالبة"}
    # pick best anchor
    best = None
    for w in words:
        t = w["text"].lower()
        if t in anchors or any(a in t for a in anchors):
            if (best is None) or (w["conf"] > best["conf"]):
                best = w

    if not best:
        return None

    # create a box near the anchor:
    # assume name is on the same line but bigger area next to it.
    img_w, img_h = pil_img.size
    y_center = (best["y1"] + best["y2"]) // 2

    # try same-line words to the right (english) or left (arabic)
    same_line = [w for w in words if abs(((w["y1"]+w["y2"])//2) - y_center) <= 20]

    # Build candidate region: big horizontal band across center line
    band_y1 = max(0, best["y1"] - 10)
    band_y2 = min(img_h, best["y2"] + 30)

    # Start x after anchor (English assumption)
    x1 = min(img_w-1, best["x2"] + 10)
    x2 = min(img_w, x1 + int(img_w * 0.45))

    # If Arabic anchor appears, flip direction (name likely to the left)
    if any(ch in best["text"] for ch in ["ا","ل","ط","ب","س","م"]):
        x2 = max(1, best["x1"] - 10)
        x1 = max(0, x2 - int(img_w * 0.45))

    # sanity
    if x2 - x1 < 80:
        # fallback: center-ish box
        x1 = int(img_w * 0.2)
        x2 = int(img_w * 0.8)

    return Box(int(x1), int(band_y1), int(x2), int(band_y2))


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


# ---------------- UI ----------------
st.set_page_config(page_title="AI Certificate Generator", page_icon="✅", layout="wide")
st.title("AI Smart Certificate Automation System")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("Upload")
    template_file = st.file_uploader("Template (PNG/JPG)", type=["png", "jpg", "jpeg"])
    csv_file = st.file_uploader("Names CSV", type=["csv"])
    font_file = st.file_uploader("Font (TTF) - optional (Arabic recommended)", type=["ttf"])

    st.subheader("Settings")
    encoding = st.selectbox("CSV encoding", ["utf-16", "utf-8", "utf-8-sig", "cp1256"], index=0)
    name_col = st.text_input("Name column", value="name")
    manual_color_hex = st.color_picker("Text color", value="#226622")
    prefix = st.text_input("Filename prefix", value="certificate_")

with c2:
    st.subheader("AI auto-detect name position")
    box = None
    if template_file:
        pil_img = Image.open(io.BytesIO(template_file.getvalue())).convert("RGB")

        with st.spinner("Running AI (OCR) to find name area..."):
            box = find_name_box_ai(pil_img)

        if box:
            st.success(f"✅ AI NAME_BOX: ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
            # show overlay preview
            vis = pil_img.copy()
            d = ImageDraw.Draw(vis)
            d.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 255, 0), width=4)
            st.image(vis, caption="AI detected name area (green box)", use_container_width=True)
        else:
            st.error("❌ AI couldn't detect name area. (We can add manual fallback if you want.)")

    st.divider()
    st.subheader("Generate")
    default_font = find_any_ttf_font()

    if st.button("🚀 Generate ZIP", use_container_width=True, type="primary"):
        if not template_file or not csv_file:
            st.error("بدنا Template + CSV.")
            st.stop()

        if not box:
            st.error("AI didn't detect box. Upload another template or we add manual fallback.")
            st.stop()

        if not default_font and not font_file:
            st.error("❌ ما لقيت أي خط TTF. ارفعي خط (TTF).")
            st.stop()

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
