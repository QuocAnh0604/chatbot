def is_global_question(question: str) -> bool:
    q = question.lower()

    GLOBAL_KEYWORDS = [
        "bao nhiêu điều",
        "bao nhiêu chương",
        "bao nhiêu mục",
        "có mấy điều",
        "có mấy chương",
        "có mấy mục",
        "tổng số điều",
        "tổng số chương",
        "toàn bộ",
        "liệt kê",
        "danh sách",
        "gồm những",
        "gồm bao nhiêu",
        "cấu trúc văn bản",
        "tóm tắt toàn văn",
        "nội dung chính",
        "điều nào",
    ]

    return any(key in q for key in GLOBAL_KEYWORDS)
