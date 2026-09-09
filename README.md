# Chatbot Hỏi Đáp Tài Liệu (RAG-based Document QA Chatbot)

Hệ thống chatbot hỏi đáp dựa trên tài liệu PDF, ứng dụng kiến trúc **Retrieval-Augmented Generation (RAG)**. Người dùng tải lên file PDF, hệ thống trích xuất và mã hóa nội dung thành cơ sở tri thức dạng vector, sau đó trả lời câu hỏi bằng cách truy xuất ngữ cảnh liên quan và sinh câu trả lời thông qua mô hình ngôn ngữ lớn (LLM).

## Kiến trúc tổng quan

Quy trình được chia thành hai giai đoạn chính, triển khai qua bốn module độc lập nhưng liên kết chặt chẽ:

1. **Giai đoạn tiền xử lý và xây dựng cơ sở tri thức**
2. **Giai đoạn hỏi đáp**

![Sơ đồ luồng xử lý](./flow-diagram.png)

---

## Giai đoạn 1: Tiền xử lý và xây dựng cơ sở tri thức

### 1. Module tiền xử lý tài liệu
- Người dùng tải file PDF lên qua giao diện.
- Hệ thống tách tài liệu thành các trang ảnh riêng lẻ.
- Áp dụng các kỹ thuật cải thiện chất lượng ảnh:
  - Chuyển đổi định dạng
  - Chuẩn hóa kích thước
  - Tăng độ tương phản
  - Khử nhiễu
  - Chỉnh sửa góc nghiêng (deskew)
- Mục tiêu: đảm bảo điều kiện tối ưu cho bước OCR tiếp theo.

### 2. Module trích xuất văn bản (OCR)
- Các trang ảnh đã tiền xử lý được đưa vào engine OCR để chuyển đổi thành văn bản thuần túy.
- Có thể kết hợp thêm metadata: vị trí khối văn bản, thứ tự đọc, thông tin bảng biểu (nếu engine hỗ trợ).
- Văn bản từ các trang được tổng hợp và làm sạch:
  - Chuẩn hóa encoding
  - Xử lý ký tự Unicode
  - Loại bỏ header/footer lặp lại
- Kết quả: một khối văn bản liền mạch, sạch sẽ.

### 3. Module xây dựng cơ sở dữ liệu tri thức
- Văn bản được chia nhỏ thành các **chunk** với kích thước phù hợp, sử dụng một trong các phương pháp:
  - Cố định kích thước (fixed-size)
  - Đệ quy theo cấu trúc (recursive)
  - Dựa trên ngữ nghĩa (semantic chunking)
  - Kèm phần chồng lấp (overlap) để giữ ngữ cảnh
- Mỗi đoạn văn bản được mã hóa thành vector ngữ nghĩa bằng **mô hình embedding**.
- Vector cùng metadata liên quan (tên file, số trang, id đoạn, tiêu đề phần...) được lưu vào **cơ sở dữ liệu vector**, sẵn sàng cho tìm kiếm theo độ tương đồng.

---

## Giai đoạn 2: Module hỏi đáp

Kích hoạt mỗi khi người dùng gửi câu hỏi:

1. **Embedding câu hỏi**: câu hỏi được chuyển thành vector embedding tương ứng.
2. **Truy xuất (Retrieval)**: hệ thống tìm kiếm trong cơ sở dữ liệu vector để lấy ra top-k đoạn văn bản có độ tương đồng cao nhất.
3. **Xếp hạng lại (Reranking)**: các kết quả top-k được đưa qua mô hình reranking (cross-encoder) để sắp xếp lại theo mức độ liên quan thực sự với câu hỏi, chọn ra tập đoạn văn bản phù hợp nhất (top-n).
4. **Xây dựng prompt**: các đoạn văn bản được chọn kết hợp với câu hỏi gốc, lịch sử hội thoại và hướng dẫn hệ thống (system instructions) để tạo thành prompt hoàn chỉnh.
5. **Sinh câu trả lời (Generation)**: prompt được đưa vào LLM để sinh câu trả lời tự nhiên, chính xác, có thể kèm trích dẫn nguồn tham chiếu.

Nhờ cơ chế truy xuất và bổ sung ngữ cảnh từ tài liệu thực tế, hệ thống đảm bảo câu trả lời vừa thông minh vừa trung thực với dữ liệu đầu vào (giảm ảo giác - hallucination).

---

## Cấu trúc thư mục (đề xuất)

```
chatbot/
├── data/                   # Tài liệu PDF đầu vào
├── src/
│   ├── preprocessing/      # Tách trang, tiền xử lý ảnh
│   ├── ocr/                # Trích xuất văn bản (OCR + làm sạch)
│   ├── knowledge_base/     # Chunking, embedding, lưu trữ vector DB
│   ├── retrieval/          # Truy vấn top-k, reranking
│   ├── generation/         # Xây dựng prompt, gọi LLM
│   └── interface/          # Giao diện người dùng (UI/API)
├── requirements.txt
└── README.md
```

> Cập nhật lại cấu trúc trên cho khớp với cấu trúc thực tế của repo.

## Cài đặt

```bash
git clone https://github.com/QuocAnh0604/chatbot.git
cd chatbot
pip install -r requirements.txt
```

## Cách sử dụng

1. Chạy giao diện hệ thống.
2. Tải file PDF lên để xây dựng cơ sở tri thức.
3. Đặt câu hỏi liên quan đến nội dung tài liệu và nhận câu trả lời từ chatbot.

## Công nghệ sử dụng

- OCR engine để trích xuất văn bản từ ảnh
- Mô hình embedding để mã hóa văn bản/câu hỏi thành vector
- Cơ sở dữ liệu vector để lưu trữ và truy vấn theo độ tương đồng
- Mô hình reranking (cross-encoder) để xếp hạng lại kết quả truy xuất
- LLM để sinh câu trả lời cuối cùng

## Đóng góp

Mọi đóng góp, báo lỗi hoặc đề xuất tính năng đều được hoan nghênh qua Issues/Pull Requests trên repo.

## License

Chưa xác định — cập nhật theo giấy phép bạn muốn sử dụng cho dự án.
