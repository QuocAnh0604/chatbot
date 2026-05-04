# app.py — Giao diện Streamlit cho hệ thống RAG văn bản pháp quy
import os
import sys
import tempfile
import streamlit as st

# ── Đảm bảo import từ thư mục gốc ──────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chatbot Văn Bản Pháp Quy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
}

/* ── Chat messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
    animation: fadeSlideUp 0.3s ease;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    animation: fadeSlideUp 0.3s ease;
}

.bubble-user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff;
    padding: 12px 18px;
    border-radius: 20px 20px 4px 20px;
    max-width: 72%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 15px rgba(102,126,234,0.35);
}

.bubble-bot {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e2e8f0;
    padding: 14px 18px;
    border-radius: 20px 20px 20px 4px;
    max-width: 82%;
    font-size: 0.95rem;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin: 0 8px;
    align-self: flex-end;
}
.avatar-user { background: linear-gradient(135deg,#667eea,#764ba2); }
.avatar-bot  { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2); }

/* ── Typing indicator ── */
.typing-indicator {
    display: flex;
    gap: 5px;
    padding: 14px 18px;
    background: rgba(255,255,255,0.08);
    border-radius: 20px 20px 20px 4px;
    width: fit-content;
    border: 1px solid rgba(255,255,255,0.1);
}
.dot {
    width: 8px; height: 8px;
    background: #a78bfa;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.6; }
    40%            { transform: translateY(-8px); opacity: 1; }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Header ── */
.page-header {
    text-align: center;
    padding: 24px 0 8px;
}
.page-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.page-header p {
    color: rgba(255,255,255,0.5);
    font-size: 0.9rem;
    margin-top: 6px;
}

/* ── Status badges ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 8px;
}
.status-ready   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.status-loading { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.status-error   { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.status-idle    { background: rgba(255,255,255,0.05); color: #94a3b8; border: 1px solid rgba(255,255,255,0.1); }

/* ── Input area ── */
.stChatInputContainer {
    background: rgba(255,255,255,0.05) !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    padding: 12px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SESSION STATE — khởi tạo 1 lần
# ════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "chat_history":   [],   # list of {"role": "user"|"bot", "content": str}
        "rag_system":     None, # instance LegalRAGSystem
        "doc_loaded":     False,
        "doc_name":       "",
        "status":         "idle",   # idle | loading | ready | error
        "status_msg":     "Chưa có tài liệu nào được tải lên.",
        "models_loaded":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════════
# LOAD MODELS — chỉ load 1 lần, cache theo session
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_rag_system():
    """Load VietOCR + DocTR vào bộ nhớ (1 lần duy nhất)."""
    from modules.extract_text import load_vietocr, load_doctr_detector
    predictor       = load_vietocr()
    doctr_detector = load_doctr_detector()
    return predictor, doctr_detector


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ Chatbot Pháp Quy")
    st.markdown("---")

    # ── Status ──────────────────────────────────────────────────
    s = st.session_state.status
    if s == "ready":
        badge = f'<div class="status-badge status-ready">✅ Sẵn sàng — {st.session_state.doc_name}</div>'
    elif s == "loading":
        badge = '<div class="status-badge status-loading">⏳ Đang xử lý tài liệu...</div>'
    elif s == "error":
        badge = f'<div class="status-badge status-error">❌ {st.session_state.status_msg}</div>'
    else:
        badge = '<div class="status-badge status-idle">📂 Chưa có tài liệu</div>'
    st.markdown(badge, unsafe_allow_html=True)

    st.markdown("---")

    # ── Upload file ──────────────────────────────────────────────
    st.markdown("### 📄 Tải lên văn bản")
    uploaded_file = st.file_uploader(
        label="Chọn file PDF hoặc ảnh",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Hỗ trợ PDF và ảnh scan (PNG, JPG).",
        key="file_uploader",
    )

    if uploaded_file and not st.session_state.doc_loaded:
        if st.button(" Xử lý & Nạp tài liệu", use_container_width=True, type="primary"):
            st.session_state.status     = "loading"
            st.session_state.status_msg = "Đang xử lý tài liệu..."
            st.session_state.doc_loaded = False
            st.session_state.chat_history = []

            with st.spinner("Đang tải mô hình OCR và xử lý tài liệu... (có thể mất vài phút)"):
                try:
                    from modules.extract_text import process_input_file
                    from modules.vector_build import build_vectorstore
                    from modules.qa_chain import load_qa_chain

                    # Ghi file tạm
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Load models (cached)
                    predictor, doctr_detector = _load_rag_system()

                    # OCR
                    raw_text = process_input_file(tmp_path, predictor, doctr_detector)
                    os.unlink(tmp_path)

                    if not raw_text:
                        raise RuntimeError("OCR thất bại, không trích xuất được văn bản.")

                    # Build vector DB
                    vector_db = build_vectorstore(raw_text)
                    if not vector_db:
                        raise RuntimeError("Tạo Vector DB thất bại.")

                    # Load QA chain
                    chat_fn = load_qa_chain()
                    if not chat_fn:
                        raise RuntimeError("Không thể khởi tạo QA chain (kiểm tra GEMINI_API_KEY).")

                    # Lưu vào session
                    st.session_state.rag_system  = chat_fn
                    st.session_state.doc_loaded  = True
                    st.session_state.doc_name    = uploaded_file.name
                    st.session_state.status      = "ready"
                    st.session_state.status_msg  = f"Sẵn sàng — {uploaded_file.name}"

                except Exception as e:
                    st.session_state.status     = "error"
                    st.session_state.status_msg = str(e)
                    st.session_state.doc_loaded = False

            st.rerun()

    elif st.session_state.doc_loaded:
        if st.button("🔄 Tải lên tài liệu khác", use_container_width=True):
            st.session_state.doc_loaded   = False
            st.session_state.doc_name     = ""
            st.session_state.status       = "idle"
            st.session_state.status_msg   = "Chưa có tài liệu nào được tải lên."
            st.session_state.chat_history = []
            st.session_state.rag_system   = None
            st.rerun()

    st.markdown("---")

    # ── Thông tin ────────────────────────────────────────────────
    st.markdown("### ℹ️ Hướng dẫn")
    st.markdown("""
1. **Tải lên** file PDF hoặc ảnh scan văn bản pháp quy.
2. Nhấn **Xử lý & Nạp tài liệu** và chờ OCR hoàn tất.
3. **Đặt câu hỏi** về nội dung văn bản trong ô bên dưới.
4. Chatbot sẽ trả lời dựa **chính xác** trên tài liệu đã tải.
""")

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <h1>⚖️ Chatbot Văn Bản Pháp Quy</h1>
    <p>Hệ thống hỏi đáp thông minh dựa trên RAG — Hỗ trợ văn bản pháp luật tiếng Việt</p>
</div>
""", unsafe_allow_html=True)

# ── Vùng chat ────────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: rgba(255,255,255,0.3);">
            <div style="font-size:4rem; margin-bottom:16px;">📜</div>
            <div style="font-size:1.1rem; font-weight:500; color:rgba(255,255,255,0.5);">
                Tải lên một văn bản pháp quy để bắt đầu hỏi đáp
            </div>
            <div style="font-size:0.85rem; margin-top:8px;">
                Hỗ trợ: Thông tư, Nghị định, Quyết định, Kế hoạch...
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Render lịch sử hội thoại
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="bubble-user">{msg["content"]}</div>
                    <div class="avatar avatar-user">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-bot">
                    <div class="avatar avatar-bot">⚖️</div>
                    <div class="bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Input câu hỏi ────────────────────────────────────────────────────────────
if st.session_state.doc_loaded:
    placeholder_text = "Nhập câu hỏi về văn bản pháp quy..."
else:
    placeholder_text = "Vui lòng tải lên tài liệu trước khi đặt câu hỏi..."

user_input = st.chat_input(
    placeholder=placeholder_text,
    disabled=not st.session_state.doc_loaded,
    key="chat_input",
)

if user_input and st.session_state.doc_loaded:
    # Thêm câu hỏi người dùng vào lịch sử
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Hiển thị typing indicator + gọi bot
    with st.spinner("⚖️ Đang tra cứu văn bản..."):
        try:
            answer = st.session_state.rag_system(user_input)
        except Exception as e:
            answer = f"❌ Lỗi khi gọi API: {e}"

    st.session_state.chat_history.append({"role": "bot", "content": answer})
    st.rerun()
