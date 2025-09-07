import io
import os
from typing import Optional, Tuple, List

import httpx
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from pydantic import BaseModel, AnyHttpUrl
from PIL import Image
from google.cloud import vision

# Tunables via env (safe defaults)
TARGET_WH = float(os.getenv("HEADSHOT_TARGET_WH", "0.78"))  # W:H ~ 35:45
MARGIN_PCT = float(os.getenv("HEADSHOT_MARGIN_PCT", "0.12"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

app = FastAPI(title="Checklist Headshot Service", version="1.0.0")


# ---------------------------
# Helpers: bytes -> PIL.Image
# ---------------------------

def _is_pdf_bytes(data: bytes, content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return data[:5] == b"%PDF-"

def _pil_from_pdf_first_page(pdf_bytes: bytes) -> Image.Image:
    # Render first page at decent resolution
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("Empty PDF")
    page = doc.load_page(0)
    # Scale ~200 DPI
    mat = fitz.Matrix(200/72, 200/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img

def _pil_from_image_bytes(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    # Normalize
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        # Drop alpha on white
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img


# ---------------------------
# Helpers: Vision face detect
# ---------------------------

def _largest_face_box_wxh(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns (x, y, w, h) or None
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    content = buf.getvalue()

    client = vision.ImageAnnotatorClient()  # uses ADC on Cloud Run
    vimg = vision.Image(content=content)
    resp = client.face_detection(image=vimg, max_results=5)

    if resp.error and resp.error.message:
        raise RuntimeError(f"Vision error: {resp.error.message}")

    anns = resp.face_annotations or []
    if not anns:
        return None

    w, h = image.size

    boxes: List[Tuple[int, int, int, int]] = []
    for fa in anns:
        poly = fa.fd_bounding_poly or fa.bounding_poly
        verts = list(poly.vertices or [])
        if not verts:
            continue
        xs = [max(0, min(int(v.x or 0), w)) for v in verts]
        ys = [max(0, min(int(v.y or 0), h)) for v in verts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bw, bh = max(1, xmax - xmin), max(1, ymax - ymin)
        boxes.append((xmin, ymin, bw, bh))

    if not boxes:
        return None

    # pick largest area
    boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
    return boxes[0]


# ---------------------------
# Helpers: crop to passport-ish
# ---------------------------

def _crop_to_ratio_center(img: Image.Image, target_wh: float) -> Image.Image:
    W, H = img.size
    cur = W / H
    if cur > target_wh:
        # too wide, reduce width
        new_w = int(round(H * target_wh))
        x = (W - new_w) // 2
        box = (x, 0, x + new_w, H)
    else:
        # too tall, reduce height
        new_h = int(round(W / target_wh))
        y = (H - new_h) // 2
        box = (0, y, W, y + new_h)
    return img.crop(box)

def _crop_around_face(img: Image.Image, box: Tuple[int, int, int, int], margin_pct: float, target_wh: float) -> Image.Image:
    W, H = img.size
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2

    # expand with margin
    exp_w = bw * (1 + 2 * margin_pct)
    exp_h = bh * (1 + 2 * margin_pct)

    # adjust to target ratio
    cur = exp_w / exp_h
    if cur < target_wh:
        exp_w = exp_h * target_wh
    else:
        exp_h = exp_w / target_wh

    # top-left
    x0 = int(round(cx - exp_w / 2))
    y0 = int(round(cy - exp_h / 2))

    # clamp
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x0 + exp_w > W: x0 = max(0, int(W - exp_w))
    if y0 + exp_h > H: y0 = max(0, int(H - exp_h))

    x1 = int(round(x0 + exp_w))
    y1 = int(round(y0 + exp_h))
    # final clamp
    x1 = min(x1, W)
    y1 = min(y1, H)

    return img.crop((x0, y0, x1, y1))


def _final_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    # mildly downscale huge images for “checklist” use (keeps bandwidth down)
    max_side = 1200
    W, H = img.size
    if max(W, H) > max_side:
        if W >= H:
            new_w = max_side
            new_h = int(H * (new_w / W))
        else:
            new_h = max_side
            new_w = int(W * (new_h / H))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------
# Endpoints
# ---------------------------

class UrlIn(BaseModel):
    url: AnyHttpUrl

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Checklist Headshot Service: /health, POST /process (multipart file) or POST /process-url (JSON {url})"

@app.post("/process-url")
async def process_url(payload: UrlIn):
    # Fetch bytes from URL
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(str(payload.url))
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Fetch failed: {r.status_code}")
            content_type = r.headers.get("content-type", "")
            data = r.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch error: {e}")

    return await _process_common(data, content_type)

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    data = await file.read()
    content_type = file.content_type or ""
    return await _process_common(data, content_type)

async def _process_common(data: bytes, content_type: str):
    try:
        if _is_pdf_bytes(data, content_type):
            img = _pil_from_pdf_first_page(data)
        else:
            img = _pil_from_image_bytes(data)

        box = _largest_face_box_wxh(img)
        if box:
            cropped = _crop_around_face(img, box, margin_pct=MARGIN_PCT, target_wh=TARGET_WH)
        else:
            cropped = _crop_to_ratio_center(img, target_wh=TARGET_WH)

        jpeg = _final_jpeg(cropped, JPEG_QUALITY)
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as e:
        # Log-ish detail in JSON, but don’t leak internals
        return JSONResponse(status_code=500, content={"error": "processing_failed", "detail": str(e)})
