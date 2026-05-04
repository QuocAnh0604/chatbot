# extract_text.py

import os
import gc
import re
import cv2
import time
import shutil
import warnings

os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["FLAGS_use_mkldnn"] = "0"

warnings.filterwarnings("ignore", category=UserWarning, module="gdown")
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*__path__.*")

import numpy as np
from PIL import Image
import pymupdf as fitz
from doctr.models import ocr_predictor
from .preprocess import preprocess, pdf_to_image, imread_unicode, imwrite_unicode

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def load_vietocr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        return Predictor(config)


def load_doctr_detector():
    try:
        detector = ocr_predictor(
            det_arch='db_mobilenet_v3_large',
            pretrained=True,
            assume_straight_pages=False
        )
        print("[INFO] DocTR Detection đã khởi động")
        return detector
    except Exception as e:
        print(f"[ERROR] Load DocTR detector thất bại: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LINE DETECTION — DocTR line-level
# ══════════════════════════════════════════════════════════════════════════════

def extract_line_bboxes_doctr(img_np: np.ndarray, doctr_detector) -> list[tuple]:
    """
    Phát hiện bbox các dòng text bằng DocTR (line-level).
    Trả về list (x, y, w, h) đã sắp xếp theo reading order.
    """
    if doctr_detector is None:
        return []
    try:
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB) if len(img_np.shape) == 2 else img_np

        result = doctr_detector([img_rgb])
        h_img, w_img = img_np.shape[:2]

        bboxes = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    geo = line.geometry
                    
                    if len(geo) == 2:
                        # Format thẳng: ((xmin, ymin), (xmax, ymax))
                        (x0, y0), (x1, y1) = geo
                        x = int(x0 * w_img)
                        y = int(y0 * h_img)
                        w_box = int((x1 - x0) * w_img)
                        h_box = int((y1 - y0) * h_img)
                    elif len(geo) == 3:
                        # Format nghiêng: ((cx, cy), (w, h), alpha)
                        (cx, cy), (bw, bh), alpha = geo
                        box_points = cv2.boxPoints(((cx * w_img, cy * h_img), (bw * w_img, bh * h_img), alpha))
                        pts = np.int32(box_points)
                        x = int(pts[:, 0].min())
                        y = int(pts[:, 1].min())
                        w_box = int(pts[:, 0].max()) - x
                        h_box = int(pts[:, 1].max()) - y
                    elif len(geo) == 4:
                        # Format Polygon 4 điểm
                        pts = np.array([[p[0] * w_img, p[1] * h_img] for p in geo], dtype=np.int32)
                        x = int(pts[:, 0].min())
                        y = int(pts[:, 1].min())
                        w_box = int(pts[:, 0].max()) - x
                        h_box = int(pts[:, 1].max()) - y
                    else:
                        continue

                    # Lọc bbox quá nhỏ / nhiễu
                    if h_box < 8 or w_box < 20:
                        continue
                    bboxes.append((x, y, w_box, h_box))

        return _sort_reading_order(bboxes)

    except Exception as e:
        print(f"[WARN] DocTR detect lỗi: {e}")
        return []


def _sort_reading_order(bboxes: list[tuple], y_tolerance: int = 18) -> list[tuple]:
    """Sắp xếp bbox theo thứ tự đọc: từ trên xuống, trái sang phải."""
    if not bboxes:
        return []

    bboxes_sorted = sorted(bboxes, key=lambda b: b[1])
    groups, current = [], [bboxes_sorted[0]]

    for box in bboxes_sorted[1:]:
        if abs(box[1] - current[-1][1]) <= y_tolerance:
            current.append(box)
        else:
            groups.append(sorted(current, key=lambda b: b[0]))
            current = [box]

    groups.append(sorted(current, key=lambda b: b[0]))
    return [b for group in groups for b in group]


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT BATCH HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _predict_batch(pil_images: list, predictor) -> list:
    results = []
    for i in range(0, len(pil_images), BATCH_SIZE):
        chunk = pil_images[i:i + BATCH_SIZE]
        try:
            results.extend(predictor.predict_batch(chunk))
        except Exception:
            for img in chunk:
                try:
                    results.append(predictor.predict(img))
                except Exception:
                    results.append("")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN TEXT
# ══════════════════════════════════════════════════════════════════════════════

_CORRECTION_MAP = {
    "KHẮCN":         "KH&CN",
    "KHÁCN":         "KH&CN",
    "KH8CN":         "KH&CN",
    "đôi tượng":     "đối tượng",
    "bô sung":       "bổ sung",
    "thâm quyên":    "thẩm quyền",
    "quyên tác giả": "quyền tác giả",
    "kí":            "ký",
    "?Quy định":     '"Quy định',
    "hà nội?":       'Hà Nội"',
}


def clean_ocr_text(text: str) -> str:
    """
    Clean nhẹ, giữ nguyên structure để phục vụ chunking:
    - KHÔNG xoá dòng
    - KHÔNG filter noise mạnh
    - CHỈ normalize + fix OCR phổ biến
    """

    # ── Fix lỗi OCR phổ biến ───────────────────────
    for wrong, right in _CORRECTION_MAP.items():
        text = text.replace(wrong, right)

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        l = line.strip()

        # giữ marker trang
        if re.match(r'^===Trang \d+===$', l):
            cleaned.append(l)
            continue

        if not l:
            continue

        # normalize khoảng trắng
        l = re.sub(r'\s+', ' ', l)

        # bỏ ký tự rác nhẹ đầu/cuối (KHÔNG đụng '-', KHÔNG xóa '.' cuối câu)
        l = re.sub(r'^[|\\/._]\s*', '', l)
        l = re.sub(r'\s*[|\\/]$', '', l)

        # fix lỗi OCR phổ biến về dấu
        l = l.replace(" ? ", " – ")
        l = l.replace(" ,", ",")
        l = l.replace(" .", ".")
        l = l.replace(" ;", ";")
        l = l.replace(" :", ":")

        cleaned.append(l)

    result = "\n".join(cleaned)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PARSE PAGES
# ══════════════════════════════════════════════════════════════════════════════

def parse_pages(text: str) -> list[dict]:
    pattern = r'===Trang (\d+)==='
    parts   = re.split(pattern, text)

    pages = []
    for i in range(1, len(parts), 2):
        pages.append({
            "page": int(parts[i]),
            "content": parts[i + 1].strip() if i + 1 < len(parts) else ""
        })
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════════

def process_image_folder(folder_path: str, predictor, doctr_detector) -> str | None:
    """
    Với mỗi ảnh trong folder:
      1. DocTR Detection → bbox các dòng text (line-level)
      2. Crop từng bbox
      3. VietOCR recognize từng crop
    """
    if not os.path.exists(folder_path):
        return None

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
    ])
    if not image_files:
        return None

    PAD_X = 4
    PAD_Y = 3

    total_pages   = len(image_files)
    final_content = ""

    for i, filename in enumerate(image_files, 1):
        img_np = imread_unicode(os.path.join(folder_path, filename))
        if img_np is None:
            continue

        # ── DocTR detect (line-level) ──────────────────────────────────────
        bboxes = extract_line_bboxes_doctr(img_np, doctr_detector)
        if not bboxes:
            print(f" Trang {i}/{total_pages}: không detect được bbox, bỏ qua.")
            continue

        h_img, w_img = img_np.shape[:2]

        # ── Crop từng line bbox ────────────────────────────────────────────
        if len(img_np.shape) == 2:
            crops = [
                Image.fromarray(
                    img_np[
                        max(0, y - PAD_Y) : min(h_img, y + h + PAD_Y),
                        max(0, x - PAD_X) : min(w_img, x + w + PAD_X)
                    ]
                )
                for (x, y, w, h) in bboxes
            ]
        else:
            crops = [
                Image.fromarray(
                    cv2.cvtColor(
                        img_np[
                            max(0, y - PAD_Y) : min(h_img, y + h + PAD_Y),
                            max(0, x - PAD_X) : min(w_img, x + w + PAD_X)
                        ],
                        cv2.COLOR_BGR2RGB
                    )
                )
                for (x, y, w, h) in bboxes
            ]

        # ── VietOCR recognize ─────────────────────────────────────────────
        print(f" Đang OCR trang {i}/{total_pages} ({len(crops)} dòng)...", end=" ", flush=True)
        t0    = time.perf_counter()
        texts = _predict_batch(crops, predictor)
        print(f"xong ({time.perf_counter() - t0:.2f}s)")

        text = "\n".join(t.strip() for t in texts if t.strip())

        final_content += f"\n===Trang {i}===\n"
        final_content += text.strip() + "\n" if text.strip() else "[TRANG TRỐNG]\n"

    gc.collect()
    return clean_ocr_text(final_content)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS INPUT FILE
# ══════════════════════════════════════════════════════════════════════════════

def process_input_file(file_path: str, predictor, doctr_detector) -> str | None:
    if not os.path.exists(file_path):
        return None

    ext       = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(os.path.dirname(current_dir), "data")
    temp_dir    = os.path.join(data_dir, "img_dir")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        if ext == ".pdf":
            doc         = fitz.open(file_path)
            total_pages = len(doc)
            print(f"[INFO] PDF có {total_pages} trang.")

            for page_num in range(total_pages):
                print(f" Đang preprocessing trang {page_num + 1}/{total_pages}...", end=" ", flush=True)

                img = pdf_to_image(file_path, page_num=page_num, dpi=300)
                if img is None:
                    print("bỏ qua (lỗi đọc trang)")
                    continue

                processed = preprocess(img)
                if processed is None:
                    print("bỏ qua (lỗi preprocess)")
                    continue

                save_path = os.path.join(temp_dir, f"{base_name}_p{page_num + 1:03d}.png")
                imwrite_unicode(save_path, processed)
                print("xong")

        elif ext in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}:
            print(" Đang preprocessing ảnh...", end=" ", flush=True)

            img = imread_unicode(file_path)
            if img is None:
                return None

            processed = preprocess(img)
            if processed is None:
                return None

            save_path = os.path.join(temp_dir, f"{base_name}_p001.png")
            imwrite_unicode(save_path, processed)
            print("xong")

        else:
            print(f"[ERROR] Định dạng không hỗ trợ: {ext}")
            return None

    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý file: {e}")
        return None

    raw_text = process_image_folder(temp_dir, predictor, doctr_detector)

    if not raw_text or not raw_text.strip():
        return None

    return raw_text


# ── Save TXT ──────────────────────────────────────────────────────────────────

def save_text_to_txt(text: str, output_path: str) -> bool:
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    predictor      = load_vietocr()
    doctr_detector = load_doctr_detector()

    test_file   = r"D:\KLTN\data\plus_document\699242.pdf"
    output_file = r"D:\KLTN\raw_ocr\result_ocr_preprocess\699242.txt"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[INFO] Bắt đầu OCR (Có Preprocess) cho file: {test_file}")

    result = process_input_file(test_file, predictor=predictor, doctr_detector=doctr_detector)

    if result:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[SUCCESS] Đã ghi kết quả OCR (Preprocess) ra file: {output_file}")
        except Exception as e:
            print(f"[ERROR] Có lỗi khi ghi file: {e}")
    else:
        print("[WARN] OCR không tạo ra kết quả để ghi")

# ── Batch mode (đã comment) ───────────────────────────────────────────
#
# if __name__ == "__main__":
#     INPUT_FOLDER  = r"D:\KLTN\data\plus_document"
#     OUTPUT_FOLDER = r"D:\KLTN\data\raw_ocr"
#
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     predictor       = load_vietocr()
#     doctr_detector  = load_doctr_detector()
#
#     pdf_files = [
#         f for f in os.listdir(INPUT_FOLDER)
#         if f.lower().endswith(".pdf")
#     ]
#
#     if not pdf_files:
#         print("[INFO] Không tìm thấy file PDF nào.")
#     else:
#         print(f"[INFO] Tìm thấy {len(pdf_files)} file PDF.\n")
#
#     for idx, filename in enumerate(pdf_files, 1):
#         file_path   = os.path.join(INPUT_FOLDER, filename)
#         output_name = os.path.splitext(filename)[0] + ".txt"
#         output_path = os.path.join(OUTPUT_FOLDER, output_name)
#
#         if os.path.exists(output_path):
#             print(f"[{idx}/{len(pdf_files)}] Bỏ qua (đã có): {output_name}")
#             continue
#
#         print(f"[{idx}/{len(pdf_files)}] Đang xử lý: {filename}")
#         result = process_input_file(file_path, predictor, doctr_detector)
#
#         if result:
#             try:
#                 with open(output_path, "w", encoding="utf-8") as f:
#                     f.write(result)
#                 print(f"  → Đã ghi: {output_name}\n")
#             except Exception as e:
#                 print(f"  → Lỗi khi ghi file: {e}\n")
#         else:
#             print(f"  → OCR thất bại, bỏ qua.\n")
#
#     print("[INFO] Hoàn tất!")