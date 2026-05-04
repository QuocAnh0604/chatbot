# preprocess.py
# Pipeline tối giản cho PDF pháp quy có con dấu:
# deskew → remove red/blue → crop → grayscale

import cv2
import numpy as np
from scipy.ndimage import rotate as nd_rotate
import os
import pymupdf as fitz  # PyMuPDF — render PDF + đếm trang
from PIL import Image

# POPPLER_PATH không còn cần thiết khi dùng PyMuPDF


# ══════════════════════════════════════════════════════════════════════════════
# UNICODE FILE I/O (fix đường dẫn tiếng Việt trên Windows)
# ══════════════════════════════════════════════════════════════════════════════

def imread_unicode(path: str):
    """cv2.imread hỗ trợ đường dẫn Unicode/tiếng Việt trên Windows."""
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img) -> bool:
    """cv2.imwrite hỗ trợ đường dẫn Unicode/tiếng Việt trên Windows."""
    ext = os.path.splitext(path)[1]
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(path)
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# TRIM BORDER — xóa rìa ảnh trước deskew để tránh artifact vệt dọc/ngang
# ══════════════════════════════════════════════════════════════════════════════

def crop_margins(img, top: int = 10, bottom: int = 10,
                 left: int = 10, right_ratio: float = 0.05):
    """
    Crop 4 rìa ảnh:
    - top/bottom/left: số pixel cố định (loại bóng scan, artifact deskew)
    - right_ratio: tỉ lệ % chiều rộng (loại sidebar watermark)
    """
    h, w = img.shape[:2]
    x_right = int(w * (1.0 - right_ratio))
    return img[top:h - bottom, left:x_right]


# ══════════════════════════════════════════════════════════════════════════════
# DESKEW
# ══════════════════════════════════════════════════════════════════════════════

def _find_score(arr, angle):
    data = nd_rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    return np.sum((hist[1:] - hist[:-1]) ** 2)


def detect_major_rotation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rotations = {
        0:   binary,
        90:  cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE),
        -90: cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE),
    }
    scores = {a: np.var(np.sum(r, axis=1)) for a, r in rotations.items()}
    return max(scores, key=scores.get)


def deskew_image(binary_img):
    angles = np.arange(-5, 6, 1)
    scores = [_find_score(binary_img, a) for a in angles]
    return angles[int(np.argmax(scores))]


# ══════════════════════════════════════════════════════════════════════════════
# REMOVE STAMPS & SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════

def remove_all_stamps_and_signatures(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask đỏ
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    b, g, r = cv2.split(img)
    red_dominant = (
        (r.astype(int) - g.astype(int) > 40) &
        (r.astype(int) - b.astype(int) > 40)
    )
    mask_red_dominant = (red_dominant.astype(np.uint8)) * 255
    mask_red = cv2.bitwise_or(mask_red, mask_red_dominant)

    # Dùng kernel nhỏ hơn để tránh mask lan vào vùng text
    kernel   = np.ones((2, 2), np.uint8)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)

    if np.sum(mask_red > 0) == 0:
        return img

    # inpaintRadius=1 để khôi phục vùng bị che mà không blur lan ra text xung quanh
    return cv2.inpaint(img, mask_red, inpaintRadius=1, flags=cv2.INPAINT_TELEA)


# ══════════════════════════════════════════════════════════════════════════════
# CROP WHITESPACE
# ══════════════════════════════════════════════════════════════════════════════

def crop_whitespace_np(img_np, padding=30):
    try:
        img      = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        gray_pil = img.convert("L")
        inv_mask = gray_pil.point(lambda p: 0 if p < 220 else 1).point(lambda p: 1 - p)
        bbox     = inv_mask.getbbox()

        if bbox is None:
            return img_np

        l, u, r, d = bbox
        w, h = img.size
        l, u = max(0, l - padding), max(0, u - padding)
        r, d = min(w, r + padding), min(h, d + padding)

        cropped = img.crop((l, u, r, d))
        return cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)

    except Exception:
        return img_np


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(image_input):
    img = imread_unicode(image_input) if isinstance(image_input, str) else image_input.copy()
    if img is None:
        return None

    # Trim rìa trước — loại bóng scan / đường kẻ mép gây artifact sau deskew
    img = crop_margins(img)

    # Deskew lớn (90°) — dùng cv2.rotate, không mất chất lượng
    major_angle = detect_major_rotation(img)
    if major_angle == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif major_angle == -90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Deskew nhỏ — chỉ rotate khi lệch đáng kể, dùng INTER_LANCZOS4 để giữ nét
    gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_tmp = cv2.threshold(gray_tmp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    angle = deskew_image(bin_tmp)

    if abs(angle) > 0.5:
        h, w = img.shape[:2]
        M   = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LANCZOS4,          # giữ nét tốt hơn INTER_LINEAR mặc định
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    # Remove dấu đỏ (inpaint để khôi phục text bị che)
    img = remove_all_stamps_and_signatures(img)

    # Crop
    img = crop_whitespace_np(img, padding=50)

    # Grayscale (KHÔNG denoise — tránh làm mờ nét chữ)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# ══════════════════════════════════════════════════════════════════════════════
# PDF UTILS — dùng PyMuPDF thay pdf2image/Poppler
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_image(pdf_path: str, page_num: int = 0, dpi: int = 300):
    """Render một trang PDF thành numpy array (BGR) dùng PyMuPDF."""
    try:
        doc  = fitz.open(pdf_path)
        page = doc[page_num]

        # matrix scale tương đương DPI (72 dpi là mặc định của fitz)
        zoom   = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=matrix, alpha=False)
        doc.close()

        # pix.samples là raw RGB bytes
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"[LỖI render trang {page_num + 1}] {e}")
        return None


def process_full_pdf(pdf_path: str, output_dir: str, dpi: int = 300):
    """Preprocess toàn bộ trang PDF và lưu ảnh ra thư mục."""
    os.makedirs(output_dir, exist_ok=True)

    doc         = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    for page_num in range(total_pages):
        img = pdf_to_image(pdf_path, page_num, dpi)
        if img is None:
            continue

        processed = preprocess(img)
        if processed is None:
            continue

        out_path = os.path.join(output_dir, f"page_{page_num + 1:02d}.png")
        imwrite_unicode(out_path, processed)


if __name__ == "__main__":
    SOURCE_DIR = r"D:\KLTN\data\plus_document"
    OUTPUT_ROOT = r"D:\KLTN\data\img_preprocess"

    # Lấy tất cả file PDF trong thư mục nguồn
    pdf_files = [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"[WARN] Không tìm thấy file PDF nào trong: {SOURCE_DIR}")
    else:
        print(f"[INFO] Tìm thấy {len(pdf_files)} file PDF. Bắt đầu xử lý...\n")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(SOURCE_DIR, pdf_name)

        # Tên folder = tên file PDF (bỏ phần mở rộng .pdf)
        folder_name = os.path.splitext(pdf_name)[0]
        output_folder = os.path.join(OUTPUT_ROOT, folder_name)

        print(f"[→] Xử lý: {pdf_name}")
        print(f"    Lưu ảnh vào: {output_folder}")

        process_full_pdf(pdf_path, output_folder, dpi=300)

        print(f"    [✓] Hoàn thành: {pdf_name}\n")

    print("[INFO] Đã xử lý xong tất cả file PDF.")