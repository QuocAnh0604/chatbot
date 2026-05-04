# vector_build_legal.py

import re
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from underthesea import sent_tokenize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE, FAISS_INDEX_PATH


# =========================
# TOKEN ESTIMATION
# =========================
CHARS_PER_TOKEN = 4   # tiếng Việt ~3–5 chars/token; lấy 4 (conservative)
MAX_TOKENS      = 200 # giới hạn tối đa mỗi chunk (model limit = 256)


def _estimate_tokens(text: str) -> int:
    """Ước tính số token bằng cách chia số ký tự cho CHARS_PER_TOKEN."""
    return len(text) // CHARS_PER_TOKEN


# =========================
# BUILD OVERLAP
# =========================
def _build_overlap(sents: list[str], max_overlap_tokens: int) -> tuple[list[str], int]:
    """
    Lấy các câu cuối của sents sao cho tổng token ước tính <= max_overlap_tokens.
    Trả về (overlap_sents, overlap_tok).
    """
    overlap_sents: list[str] = []
    overlap_tok = 0
    for s in reversed(sents):
        t = _estimate_tokens(s)
        if overlap_tok + t <= max_overlap_tokens:
            overlap_sents.insert(0, s)
            overlap_tok += t
        else:
            break
    return overlap_sents, overlap_tok


# =========================
# SPLIT LARGE CHUNK + OVERLAP THEO CÂU (kiểm soát theo token)
# =========================
def normalize_legal_text(text: str) -> str:
    # Fix "1.Quyết định" -> "1. Quyết định"
    text = re.sub(r'(\n\d+)\.(\S)', r'\1. \2', text)

    # Fix "a)Nội dung" -> "a) Nội dung"
    text = re.sub(r'(\n[a-zđ]\))(\S)', r'\1 \2', text)

    return text
def split_large_chunk(
    chunk: str,
    max_tokens: int = MAX_TOKENS,
    overlap_ratio: float = 0.15,
) -> list[str]:
    """
    Tách chunk thành các phần <= max_tokens (token ước tính).
    - Không bao giờ cắt giữa câu.
    - Overlap lấy các câu cuối của chunk trước sao cho tổng <= max_tokens * overlap_ratio.
    - Câu đơn dài hơn max_tokens: giữ nguyên (không thể cắt giữa câu),
      nhưng vẫn carry-over overlap cho câu tiếp theo.
    """
    max_overlap_tok = int(max_tokens * overlap_ratio)

    # Chunk đủ ngắn → trả về ngay, không cần tách
    if _estimate_tokens(chunk) <= max_tokens:
        return [chunk]

    sentences = sent_tokenize(chunk)
    if not sentences:
        # Fallback: hard-cut theo chars nếu sent_tokenize không hoạt động
        max_chars = max_tokens * CHARS_PER_TOKEN
        return [chunk[i: i + max_chars] for i in range(0, len(chunk), max_chars)]

    chunks:        list[str] = []
    current_sents: list[str] = []
    current_tok:   int       = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        sent_tok = _estimate_tokens(sent)

        # ── Câu đơn vượt max_tokens ─────────────────────────────────────────
        if sent_tok > max_tokens:
            if current_sents:
                chunks.append(" ".join(current_sents))

            # 🔥 HARD SPLIT theo chars
            max_chars = max_tokens * CHARS_PER_TOKEN
            sub_chunks = [
                sent[i:i + max_chars]
                for i in range(0, len(sent), max_chars)
            ]

            chunks.extend(sub_chunks)

            current_sents = []
            current_tok   = 0
            continue

        # ── Còn đủ chỗ trong buffer ─────────────────────────────────────────
        if current_tok + sent_tok <= max_tokens:
            current_sents.append(sent)
            current_tok += sent_tok

        # ── Buffer đầy → flush + tạo chunk mới với overlap ──────────────────
        else:
            if current_sents:
                chunks.append(" ".join(current_sents))

            overlap_sents, overlap_tok = _build_overlap(current_sents, max_overlap_tok)
            current_sents = overlap_sents + [sent]
            current_tok   = overlap_tok + sent_tok

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


# =========================
# PARENT-CHILD CHUNKING
# Cấu trúc: Điều > Mục (1. ; 2.) > Khoản (a) ; b))
# =========================

DIEU_PATTERN  = re.compile(r"(?=Điều\s+\d+[\.:])", re.DOTALL)

# FIX: yêu cầu \n trước digit → tránh match '1.' trong 'Điều 1. Tiêu đề'
MUC_PATTERN   = re.compile(r"(?=\n\d+\.\s+[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])", re.DOTALL)

KHOAN_PATTERN = re.compile(r"(?=(?<!\w)[a-zđ]\)\s+\S)", re.DOTALL)


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[str]:
    parts = pattern.split(text)
    return [p.strip() for p in parts if p.strip()]


def _extract_dieu_header(dieu_text: str) -> str:
    """
    Trích tiêu đề Điều: lấy dòng đầu tiên, không giới hạn ký tự.
    Không dùng regex để tránh match sang nội dung dòng kế tiếp.
    """
    first_line = dieu_text.split("\n")[0].strip()
    return first_line if first_line else dieu_text.strip()


def _truncate_at_sentence(text: str, max_tokens: int) -> str:
    """
    Truncate parent_content tại ranh giới câu sao cho token ước tính <= max_tokens.
    Không bao giờ cắt giữa câu.
    """
    if _estimate_tokens(text) <= max_tokens:
        return text

    sentences = sent_tokenize(text)
    result_sents: list[str] = []
    result_tok = 0

    for s in sentences:
        t = _estimate_tokens(s)
        if result_tok + t <= max_tokens:
            result_sents.append(s)
            result_tok += t
        else:
            break

    return " ".join(result_sents).strip() if result_sents else text[: max_tokens * CHARS_PER_TOKEN]


def _process_dieu(dieu_text: str, page_num: int, make_doc, docs: list):
    dieu_header = _extract_dieu_header(dieu_text)
    muc_parts   = _split_by_pattern(dieu_text, MUC_PATTERN)

    # ── Không có Mục → chunk thẳng toàn bộ Điều ────────────────────────────
    if not muc_parts or len(muc_parts) == 1:
        for idx, c in enumerate(split_large_chunk(dieu_text)):
            docs.append(make_doc(c, {
                "page":  page_num,
                "dieu":  dieu_header,
                "level": "dieu",
                "chunk": idx + 1,
            }))
        return

    # ── Có Mục → tách dieu_intro và muc_list ────────────────────────────────
    # muc_parts[0] có thể là phần intro của Điều (trước mục 1.)
    # hoặc đã là mục 1. luôn
    first_is_muc = bool(re.match(r"\d+\.\s+", muc_parts[0]))
    dieu_intro   = "" if first_is_muc else muc_parts[0]
    muc_list     = muc_parts if first_is_muc else muc_parts[1:]

    # ✅ Tạo chunk riêng cho Điều (dieu_intro nếu có, không thì chỉ dieu_header)
    # Chunk này giúp query "Điều X ..." có thể retrieve được
    dieu_chunk_content = (dieu_intro.strip() or dieu_header)
    docs.append(make_doc(dieu_chunk_content, {
        "page":  page_num,
        "dieu":  dieu_header,
        "level": "dieu",
        "chunk": 1,
    }))

    # ── Xử lý từng Mục ──────────────────────────────────────────────────────
    for muc_text in muc_list:
        muc_label_match = re.match(r"(\d+\.)\s+", muc_text)
        muc_label       = muc_label_match.group(1) if muc_label_match else ""

        # parent của Mục = dieu_header (để embed biết chunk này thuộc Điều nào)
        muc_parent = dieu_header

        khoan_parts = _split_by_pattern(muc_text, KHOAN_PATTERN)

        # ── Không có Khoản → chunk thẳng Mục ────────────────────────────────
        if not khoan_parts or len(khoan_parts) == 1:
            for idx, c in enumerate(split_large_chunk(muc_text)):
                docs.append(make_doc(c, {
                    "page":           page_num,
                    "dieu":           dieu_header,
                    "muc":            muc_label,
                    "level":          "muc",
                    "chunk":          idx + 1,
                    "parent_content": muc_parent,
                }))
            continue

        # ── Có Khoản → tách muc_intro và khoan_list ─────────────────────────
        first_is_khoan = bool(re.match(r"[a-zđ]\)\s+", khoan_parts[0]))
        muc_intro      = "" if first_is_khoan else khoan_parts[0]
        khoan_list     = khoan_parts if first_is_khoan else khoan_parts[1:]

        # parent của Khoản = dieu_header + muc_intro (nếu có)
        # lấy toàn bộ, không giới hạn
        khoan_parent_raw = (dieu_header + ". " + muc_intro).strip(". ")
        khoan_parent     = khoan_parent_raw

        # ── Xử lý từng Khoản ─────────────────────────────────────────────────
        for khoan_text in khoan_list:
            khoan_label_match = re.match(r"([a-zđ]\))\s+", khoan_text)
            khoan_label       = khoan_label_match.group(1) if khoan_label_match else ""

            for idx, c in enumerate(split_large_chunk(khoan_text)):
                docs.append(make_doc(c, {
                    "page":           page_num,
                    "dieu":           dieu_header,
                    "muc":            muc_label,
                    "khoan":          khoan_label,
                    "level":          "khoan",
                    "chunk":          idx + 1,
                    "parent_content": khoan_parent,
                }))
def merge_broken_lines(lines: list[str]) -> list[str]:
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not buffer:
            buffer = line
            continue

        # Nếu dòng trước chưa kết thúc câu → nối
        if not re.search(r'[.:;?!]$', buffer):
            # Nếu dòng mới là bullet / structure → không nối
            if re.match(r'^(\d+\.|[a-zđà-ỹ]\)|Điều\s+\d+|Khoản\s+\d+|Mục\s+\d+|Điểm\s+[a-zđà-ỹ])', line):
                merged.append(buffer)
                buffer = line
            else:
                buffer += " " + line
        else:
            merged.append(buffer)
            buffer = line

    if buffer:
        merged.append(buffer)

    return merged


def split_text_to_docs(text: str) -> list[Document]:
    """
    Parent-Child Chunking theo cấu trúc pháp lý:
      - Level 0 (Parent):     Điều X. <tiêu đề>
      - Level 1 (Child):      Mục 1. / 2. ... (nếu có)
      - Level 2 (Grandchild): Khoản a) / b) ... (nếu có)

    Carry-over: nội dung đầu trang không bắt đầu bằng Điều
    → ghép vào Điều cuối trang trước, giữ page_num trang hiện tại.

    Metadata keys:
      chunk_id        – ID duy nhất (chunk_0001, chunk_0002, ...)
      page            – số trang thực tế của chunk
      dieu            – tiêu đề Điều cha (ngắn gọn, <= 100 chars)
      muc             – nhãn Mục (nếu có)
      khoan           – nhãn Khoản (nếu có)
      level           – "dieu" | "muc" | "khoan" | "raw"
      chunk           – thứ tự chunk trong cùng node (khi bị split_large)
      parent_content  – nội dung node cha, truncate tại ranh giới câu
    """
    page_pattern = r"===Trang\s+(\d+)===\s*\n?(.*?)(?=(===Trang\s+\d+===)|$)"
    page_matches  = re.findall(page_pattern, text, re.DOTALL)

    pages = (
        [(int(p), c.strip()) for p, c, _ in page_matches if c.strip()]
        if page_matches
        else [(1, text.strip())]
    )

    processed_pages = []
    for p_num, p_text in pages:
        lines = p_text.splitlines()
        merged = merge_broken_lines(lines)
        processed_pages.append((p_num, "\n".join(merged)))
    pages = processed_pages

    docs: list[Document] = []
    chunk_counter = 0

    def make_doc(content: str, meta: dict) -> Document:
        nonlocal chunk_counter
        chunk_counter += 1
        meta["chunk_id"] = f"chunk_{chunk_counter:04d}"
        return Document(page_content=content, metadata=meta)

    # KIỂM TRA XEM VĂN BẢN CÓ ĐIỀU KHOẢN KHÔNG?
    has_dieu = bool(re.search(r"Điều\s+\d+", text))

    if not has_dieu:
        # ======================================================================
        # CHẾ ĐỘ 1: VĂN BẢN THƯỜNG (KHÔNG CÓ ĐIỀU KHOẢN)
        # Tách hoàn toàn theo Token (200 tokens, overlap ~30 tokens)
        # ======================================================================
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            
            # split_large_chunk đã được cấu hình mặc định MAX_TOKENS=200, overlap=0.15
            chunks = split_large_chunk(page_text)
            for idx, c in enumerate(chunks):
                docs.append(make_doc(c, {
                    "page":  page_num,
                    "level": "raw",
                    "chunk": idx + 1,
                }))
        return docs

    # ==========================================================================
    # CHẾ ĐỘ 2: VĂN BẢN PHÁP LUẬT (CÓ ĐIỀU KHOẢN)
    # Bóc tách theo cấu trúc Điều > Mục > Khoản
    # ==========================================================================
    carry_dieu_text: str | None = None
    carry_page_num:  int        = 1

    for page_num, page_text in pages:

        dieu_parts = _split_by_pattern(page_text, DIEU_PATTERN)
        if not dieu_parts:
            continue

        starts_with_dieu = bool(re.match(r"Điều\s+\d+", dieu_parts[0]))

        if not starts_with_dieu:
            orphan = dieu_parts[0]

            if carry_dieu_text is not None:
                # Ghép orphan vào Điều cuối trang trước
                _process_dieu(carry_dieu_text + " " + orphan, page_num, make_doc, docs)
                carry_dieu_text = None
            else:
                # Trang bắt đầu không có Điều mà cũng không có carry -> raw
                for idx, c in enumerate(split_large_chunk(orphan)):
                    docs.append(make_doc(c, {
                        "page":  page_num,
                        "level": "raw",
                        "chunk": idx + 1,
                    }))

            remaining = dieu_parts[1:]
        else:
            # Trang bắt đầu bằng Điều → flush carry cũ trước
            if carry_dieu_text is not None:
                _process_dieu(carry_dieu_text, carry_page_num, make_doc, docs)
                carry_dieu_text = None
            remaining = dieu_parts

        for i, dieu_text in enumerate(remaining):
            if not re.match(r"Điều\s+\d+", dieu_text):
                for idx, c in enumerate(split_large_chunk(dieu_text)):
                    docs.append(make_doc(c, {
                        "page":  page_num,
                        "level": "raw",
                        "chunk": idx + 1,
                    }))
                continue

            is_last = (i == len(remaining) - 1)

            if is_last:
                carry_dieu_text = dieu_text
                carry_page_num  = page_num
            else:
                _process_dieu(dieu_text, page_num, make_doc, docs)

    # Flush Điều cuối cùng của toàn bộ văn bản
    if carry_dieu_text is not None:
        _process_dieu(carry_dieu_text, carry_page_num, make_doc, docs)

    return docs


# =========================
# BUILD VECTORSTORE
# =========================
def build_vectorstore(text: str):
    if not text:
        print("[ERROR] Nội dung văn bản rỗng.")
        return None

    print("\n🚀 Bắt đầu build vector (parent-child chunking)...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    docs = split_text_to_docs(text)

    if not docs:
        print("[WARN] Không có chunk.")
        return None

    level_counts: dict[str, int] = {}
    for d in docs:
        lvl = d.metadata.get("level", "?")
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    print(f"[INFO] Tổng số chunks: {len(docs)}")
    for lvl, cnt in sorted(level_counts.items()):
        print(f"       ├─ level={lvl}: {cnt} chunks")

    try:
        vectorstore = FAISS.from_documents(docs, embeddings)

        folder = os.path.dirname(FAISS_INDEX_PATH)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        vectorstore.save_local(FAISS_INDEX_PATH)
        print("✅ Build vectorstore thành công!")
        return vectorstore

    except Exception as e:
        print(f"[ERROR] FAISS: {e}")
        return None


# =========================
# MAIN — 2 chế độ: debug chunking hoặc build FAISS
# =========================
def main():
    file_path = input("Nhập đường dẫn file txt: ").strip().strip('"')

    if not os.path.exists(file_path):
        print(f"[ERROR] File không tồn tại: {file_path}")
        return

    mode = input("Chế độ: [1] Debug chunking  [2] Build FAISS  (mặc định: 1): ").strip()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if mode == "2":
        result = build_vectorstore(text)
        print("[SUCCESS]" if result else "[FAILED]")
        return

    # ── Chế độ debug chunking ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("   KIỂM TRA CHUNKING — PARENT-CHILD + CARRY-OVER")
    print("=" * 70)

    docs = split_text_to_docs(text)

    if not docs:
        print("[WARN] Không tạo được chunk nào.")
        return

    level_counts: dict[str, int] = {}
    token_counts: list[int]      = []
    for d in docs:
        lvl = d.metadata.get("level", "?")
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
        token_counts.append(_estimate_tokens(d.page_content))

    over_limit = [t for t in token_counts if t > MAX_TOKENS]

    print(f"\n[INFO] Tổng số chunks  : {len(docs)}")
    for lvl, cnt in sorted(level_counts.items()):
        print(f"       ├─ level={lvl}: {cnt} chunks")
    print(f"\n[TOKEN] max={max(token_counts)}  avg={sum(token_counts)//len(token_counts)}  "
          f"vượt {MAX_TOKENS} tokens: {len(over_limit)} chunks")

    print("\n" + "─" * 70)
    sep = "─" * 70

    for i, doc in enumerate(docs, 1):
        meta   = doc.metadata
        level  = meta.get("level",    "?")
        page   = meta.get("page",     "?")
        dieu   = meta.get("dieu",     "")
        muc    = meta.get("muc",      "")
        khoan  = meta.get("khoan",    "")
        chunk  = meta.get("chunk",    "")
        cid    = meta.get("chunk_id", "")
        tok    = _estimate_tokens(doc.page_content)
        flag   = " ⚠️ OVER" if tok > MAX_TOKENS else ""

        breadcrumb_parts = [f"level={level}", f"trang={page}", f"~{tok}tok{flag}"]
        if dieu:  breadcrumb_parts.append(f"dieu='{dieu[:50]}'")
        if muc:   breadcrumb_parts.append(f"muc='{muc}'")
        if khoan: breadcrumb_parts.append(f"khoan='{khoan}'")
        if chunk: breadcrumb_parts.append(f"chunk={chunk}")

        print(f"\n[{cid}] {' | '.join(breadcrumb_parts)}")
        print(f"  {doc.page_content.replace(chr(10), '↵ ')}")

        parent = meta.get("parent_content", "")
        if parent:
            print(f"  [parent]: {parent[:120]}{'...' if len(parent) > 120 else ''}")

        print(sep)

        if i % 20 == 0:
            cont = input(f"\n  -- {i}/{len(docs)} chunks. Tiếp tục? (Enter / 'q'): ").strip()
            if cont.lower() == "q":
                break

    print(f"\n[DONE] {len(docs)} chunks.")


if __name__ == "__main__":
    main()