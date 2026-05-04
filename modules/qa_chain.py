#QA_CHAIN.py
import os
from collections import deque

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.rerank import rerank_documents
from modules.question_classifier import is_global_question

from config.settings import (
    FAISS_INDEX_PATH, EMBEDDING_MODEL, EMBEDDING_DEVICE,
    GEMINI_API_KEY, GEMINI_MODEL_NAME
)


def format_docs(docs):
    formatted = []
    for doc in docs:
        content = doc.page_content.replace("\n", " ")
        page    = doc.metadata.get("page", "N/A")
        formatted.append(f"[Thông tin tại Trang {page}]: {content}")
    return "\n\n".join(formatted)


def format_chat_history(history: deque) -> str:
    """
    Chuyển deque history thành chuỗi để nhúng vào prompt.
    Mỗi phần tử là ("user" | "bot", "nội dung").
    """
    if not history:
        return "Chưa có lịch sử hội thoại."

    lines = []
    for role, content in history:
        prefix = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def load_qa_chain():
    print("[INFO] Đang khởi tạo RAG Chain...")

    if not GEMINI_API_KEY:
        print("[ERROR] Thiếu GEMINI_API_KEY.")
        return None

    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"[ERROR] Không tìm thấy Vector Store tại: {FAISS_INDEX_PATH}")
        return None

    try:
        embeddings  = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
        )
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"[ERROR] Không load được FAISS: {e}")
        return None

    # ── Small-to-Big: load toàn bộ docs từ docstore một lần ──────────────────
    all_docs = list(vectorstore.docstore._dict.values())
    print(f"[INFO] Đã load {len(all_docs)} docs vào bộ nhớ cho Small-to-Big retrieval.")

    # Index nhanh: dieu → list[Document] để expand O(1)
    from collections import defaultdict
    dieu_index: dict[str, list] = defaultdict(list)
    for doc in all_docs:
        dieu_val = doc.metadata.get("dieu", "")
        if dieu_val:
            dieu_index[dieu_val].append(doc)

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.4,
        max_output_tokens=3072
    )

    template = """
Bạn là trợ lý AI chuyên về văn bản pháp quy. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp trong phần CONTEXT.

YÊU CẦU QUAN TRỌNG:
1. Chỉ trả lời dựa trên thông tin trong CONTEXT. Không được tự ý bổ sung kiến thức bên ngoài hoặc bịa đặt.
2. Nếu tìm thấy thông tin, hãy trích dẫn số trang một cách chính xác. Ví dụ: "...theo quy định (Trang 5)".
3. Nếu không tìm thấy thông tin trong CONTEXT, hãy nói: "Xin lỗi, tài liệu hiện tại không chứa thông tin bạn cần."
4. Nếu câu hỏi có liên quan đến lịch sử hội thoại (ví dụ: "từ số 9", "cái đó", "giải thích thêm"...),
   hãy dựa vào LỊCH SỬ HỘI THOẠI để hiểu đúng ý người dùng.

XỬ LÝ DỮ LIỆU NHIỄU:
- Dữ liệu CONTEXT được trích xuất từ OCR nên có thể chứa các từ nhiễu vô nghĩa (ví dụ: 'nos', 'STA', 'MANN',...).
- Hãy chủ động khôi phục ý nghĩa của câu dựa trên ngữ cảnh nếu từ ngữ chỉ bị lỗi nhẹ.
- Loại bỏ hoàn toàn các thực thể hoặc ký tự vô nghĩa ra khỏi quá trình suy luận.

--- CONTEXT BẮT ĐẦU ---
{context}
--- CONTEXT KẾT THÚC ---

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI CHI TIẾT:
"""
    prompt    = PromptTemplate(
        template=template,
        input_variables=[ "context", "question"]
    )
    rag_chain = prompt | llm

    # maxlen=6: 3 lượt × (1 user + 1 bot) = 6 phần tử
    chat_history: deque = deque(maxlen=6)

    def _calc_k(pct: float, floor: int, ceil: int) -> int:
        total = vectorstore.index.ntotal
        return max(floor, min(ceil, round(total * pct)))

    def _small_to_big(query: str, k_retrieve: int, top_n: int) -> list:
        """
        Small-to-Big Retrieval:
          1. Retrieve k chunk nhỏ (khoản/mục) bằng similarity search.
          2. Lấy tập dieu duy nhất từ kết quả.
          3. Mở rộng: lấy toàn bộ chunk cùng Điều từ dieu_index.
          4. Rerank tập mở rộng, giữ top_n.
        """
        # Bước 1: Retrieve chunk nhỏ
        seed_docs = vectorstore.similarity_search(query, k=k_retrieve)
        if not seed_docs:
            return []

        # Bước 2: Tập hợp các Điều liên quan
        dieu_set = {d.metadata.get("dieu", "") for d in seed_docs if d.metadata.get("dieu", "")}
        print(f"[S2B] Seed chunks: {len(seed_docs)} | Điều liên quan: {len(dieu_set)}")

        # Bước 3: Mở rộng sang tất cả chunk cùng Điều
        expanded: list = []
        seen_ids: set  = set()
        for dieu_val in dieu_set:
            for doc in dieu_index.get(dieu_val, []):
                cid = doc.metadata.get("chunk_id", id(doc))
                if cid not in seen_ids:
                    expanded.append(doc)
                    seen_ids.add(cid)

        # Fallback: nếu expand rỗng (chunk level raw không có dieu)
        if not expanded:
            expanded = seed_docs

        print(f"[S2B] Expanded pool: {len(expanded)} chunks → rerank → top {top_n}")

        # Bước 4: Rerank tập mở rộng
        reranked = rerank_documents(query, expanded, top_n=top_n)
        if not reranked:
            print("[WARN] Rerank trả rỗng, dùng seed docs gốc.")
            reranked = seed_docs[:top_n]

        return reranked

    def ask_bot(query: str) -> str:
        history_str = format_chat_history(chat_history)

        # ── Global question: không cần Small-to-Big, lấy rộng trực tiếp ───────
        if is_global_question(query):
            k    = _calc_k(pct=0.30, floor=10, ceil=9999)
            docs = vectorstore.similarity_search(query, k=k)
            if not docs:
                return "Không tìm thấy tài liệu liên quan đến câu hỏi của bạn."
            context_str = format_docs(docs)

        # ── Normal question: Small-to-Big + Rerank ───────────────────────────
        else:
            k_retrieve = _calc_k(pct=0.12, floor=10, ceil=9999)  # seed nhỏ
            top_n      = max(5, round(k_retrieve * 0.5))        # giữ sau rerank
            reranked   = _small_to_big(query, k_retrieve, top_n)
            context_str = format_docs(reranked)

        try:
            response = rag_chain.invoke({
                "chat_history": history_str,
                "context"     : context_str,
                "question"    : query,
            })
            answer = response.content

            chat_history.append(("user", query))
            chat_history.append(("bot",  answer))

            return answer

        except Exception as e:
            return f"Lỗi khi gọi Gemini API: {e}"

    print("[SUCCESS] RAG Chain đã sẵn sàng!")
    return ask_bot