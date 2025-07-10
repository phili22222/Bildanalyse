import streamlit as st
import cv2
import numpy as np
import pytesseract
import os
import sys
from pyzbar.pyzbar import decode
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import pandas as pd
import io
import math

# ----------------------------------------------------------
# Thumbnail-Größe (0.40 = 40 %)
# ----------------------------------------------------------
IMG_SCALE = 0.40

# ----------------------------------------------------------
# Dynamischer Pfad zu Tesseract
# ----------------------------------------------------------
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS  # type: ignore
else:
    base_path = os.path.dirname(__file__)

tesseract_dir = os.path.join(base_path, "Mitarbeiter", "Tesseract-OCR")
pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_dir, "tesseract.exe")
os.environ["TESSDATA_PREFIX"] = os.path.join(tesseract_dir, "tessdata")

# ----------------------------------------------------------
# Standard-Gewichte und -Normalisierungen
# ----------------------------------------------------------
DEFAULT_WEIGHTS = {
    "Allgemein-kontrast": 0.25,
    "Text-kontrast":      0.25,
    "Schärfe":            0.20,
    "Texterkennung":      0.10,
    "Rauschen":           0.15,
}
DEFAULT_NORMALIZATION = {
    "Allgemein-kontrast": (10, 80),
    "Text-kontrast":      (0.1, 0.7),
    "Schärfe":            (10, 80),
    "Texterkennung":      (0, 100),
    "Rauschen":           (2.5, 4.5),
}
BLUE_NORMALIZATION = {
    "Allgemein-kontrast": (10, 40),
    "Text-kontrast":      (0.1, 0.35),
    "Schärfe":            (10, 40),
    "Texterkennung":      (0, 100),
    "Rauschen":           (2.5, 4.5),
}
YELLOW_NORMALIZATION = {
    "Allgemein-kontrast": (10, 45),
    "Text-kontrast":      (0.1, 0.4),
    "Schärfe":            (10, 40),
    "Texterkennung":      (0, 100),
    "Rauschen":           (2.5, 4.5),
}
ANTHRAZIT_NORMALIZATION = {
    "Allgemein-kontrast": (10, 65),
    "Text-kontrast":      (0.1, 0.57),
    "Schärfe":            (10, 55),
    "Texterkennung":      (0, 100),
    "Rauschen":           (2.5, 4.5),
}
BLACK_NORMALIZATION = {
    "Allgemein-kontrast": (10, 65),
    "Text-kontrast":      (0.1, 0.57),
    "Schärfe":            (10, 55),
    "Texterkennung":      (0, 100),
    "Rauschen":           (2.5, 4.5),
}
COLOR_PRESETS = {
    "Blau":      {"weights": DEFAULT_WEIGHTS, "norm": BLUE_NORMALIZATION},
    "Gelb":      {"weights": DEFAULT_WEIGHTS, "norm": YELLOW_NORMALIZATION},
    "Anthrazit": {"weights": DEFAULT_WEIGHTS, "norm": ANTHRAZIT_NORMALIZATION},
    "Schwarz":   {"weights": DEFAULT_WEIGHTS, "norm": BLACK_NORMALIZATION},
}

# ----------------------------------------------------------
# Bildverarbeitungsfunktionen
# ----------------------------------------------------------
def preprocess(image: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return cv2.fastNlMeansDenoising(gray_eq, None, 10, 7, 21)

def rms_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))

def michelson_contrast(gray: np.ndarray) -> float:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg, bg = gray[thresh == 255], gray[thresh == 0]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    return float((np.mean(fg) - np.mean(bg)) / (np.mean(fg) + np.mean(bg)))

def tenengrad_schaerfe(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))

def texterkennung_confidence(gray: np.ndarray) -> float:
    cfg  = "--oem 3 --psm 3 -l eng+deu"
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=cfg)
    conf = [int(c) for c in data["conf"] if str(c).lstrip("-+").isdigit() and int(c) >= 0]
    return float(np.mean(conf)) if conf else 0.0

def barcode_present(image: np.ndarray) -> bool:
    if decode(image):
        return True
    val, *_ = cv2.QRCodeDetector().detectAndDecode(image)
    return bool(val)

def entropy_noise(gray: np.ndarray) -> float:
    ent = entropy(img_as_ubyte(gray / 255.0), disk(5))
    return float(np.mean(ent))

def _normalize(val: float, vmin: float, vmax: float) -> float:
    return float(np.clip((val - vmin) / (vmax - vmin) * 100, 0, 100))

def compute_score(normed: dict[str, float], weights: dict[str, float]) -> float:
    raw = sum(normed[k] * weights[k] for k in weights)
    return float(np.clip(raw * 1.3, 0, 100))

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="Plakettenanalyse", layout="wide")
st.title("Analyse der Plakettenlesbarkeit")

farbe = st.selectbox("Bitte Plakettenfarbe auswählen:", list(COLOR_PRESETS.keys()))
preset       = COLOR_PRESETS[farbe]
WEIGHTS      = preset["weights"]
NORMALIZATION = preset["norm"]

uploaded_files = st.file_uploader(
    "Bilder hochladen",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if uploaded_files:
    results          = []
    images_for_excel = []

    for f in uploaded_files:
        st.markdown("---")
        raw_bytes = f.read()
        img_buf   = io.BytesIO(raw_bytes)
        img_arr   = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        image     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        proc      = preprocess(image)

        r_raw = entropy_noise(gray)
        r_disp = _normalize(r_raw, *NORMALIZATION["Rauschen"])
        r_int  = 100.0 - r_disp

        raw = {
            "Allgemein-kontrast": rms_contrast(proc),
            "Text-kontrast":      michelson_contrast(proc),
            "Schärfe":            tenengrad_schaerfe(proc),
            "Texterkennung":      texterkennung_confidence(proc),
            "Rauschen":           r_disp,
        }
        normed = {
            k: (_normalize(v, *NORMALIZATION[k]) if k not in ["Texterkennung", "Rauschen"] else v)
            for k, v in raw.items()
        }
        normed_for_score = normed.copy()
        normed_for_score["Rauschen"] = r_int

        gesamt  = compute_score(normed_for_score, WEIGHTS)
        barcode = barcode_present(image)

        results.append({
            "Datei":              f.name,
            "Allgemein-kontrast": normed["Allgemein-kontrast"],
            "Text-kontrast":      normed["Text-kontrast"],
            "Schärfe":            normed["Schärfe"],
            "Texterkennung":      normed["Texterkennung"],
            "Rauschen":           normed["Rauschen"],
            "Barcode erkannt":    "Ja" if barcode else "Nein",
            "Gesamt-Score":       gesamt,
        })
        images_for_excel.append({"name": f.name, "buf": img_buf})

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption=f.name, channels="BGR", width=500)
        with col2:
            st.markdown("### Einzelmetriken")
            for k, v in normed.items():
                st.progress(v / 100, text=f"{k}: {v:.1f}")
            st.markdown(f"**Barcode erkannt:** {'Ja' if barcode else 'Nein'}")
            st.markdown(f"### Gesamt-Score: **{gesamt:.1f} / 100**")

    # ------------------------------------------------------
    # Excel-Export
    # ------------------------------------------------------
    if results:
        st.markdown("---")
        st.markdown("### Excel-Export")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            # ----- Scores-Sheet -----
            df.to_excel(writer, index=False, sheet_name="Scores")
            ws_scores = writer.sheets["Scores"]
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                ws_scores.set_column(i, i, max_len)
            ws_scores.freeze_panes(1, 1)

            # ----- Bilder-Sheet -----
            wb = writer.book
            ws = wb.add_worksheet("Bilder")
            ws.set_column(2, 2, 45)           # Spalte C (Dateiname) breiter

            row = 0
            for img in images_for_excel:
                # Bildabmessungen
                h_px, w_px = cv2.imdecode(
                    np.frombuffer(img["buf"].getvalue(), dtype=np.uint8),
                    cv2.IMREAD_COLOR
                ).shape[:2]
                h_pts = h_px * IMG_SCALE * 0.75
                ws.set_row(row, h_pts)        # Zeile passend hochziehen

                # Bild in Spalte A (0), Dateiname in Spalte C (2)
                ws.insert_image(row, 0, img["name"], {
                    "image_data": img["buf"],
                    "x_scale": IMG_SCALE,
                    "y_scale": IMG_SCALE,
                })
                ws.write(row, 2, img["name"])
                row += 1                      # direkt nächste Zeile

        st.download_button(
            "Scores + Bilder als Excel herunterladen",
            buffer.getvalue(),
            file_name="plaketten_scores_bilder.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
