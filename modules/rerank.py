import torch
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import List
from sentence_transformers import CrossEncoder

# Singleton — load model 1 lần duy nhất, tái dùng cho mọi query
_reranker_instance = None

class Reranker:
    def __init__(self, model_name: str = "namdp-ptit/ViRanker", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Giới hạn max_length là 8194 token theo mô hình
        self.model = CrossEncoder(model_name, device=self.device, max_length=8194)

    def rerank(self, query: str, docs: List, top_n: int = 5) -> List:
        if not docs:
            return []

        # Giới hạn embedding là 256 token, nên ta có thể đưa toàn bộ nội dung chunk vào reranker 
        # (tokenizer của CrossEncoder sẽ tự động truncate dựa vào max_length=8194 nếu vượt quá)
        pairs = [
            (query, d.page_content)
            for d in docs
        ]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"[WARN] Rerank thất bại: {e}, dùng docs gốc.")
            return docs[:top_n]

        for d, s in zip(docs, scores):
            d.metadata["rerank_score"] = float(s)

        ranked = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        return ranked[:top_n]


def rerank_documents(query: str, docs: List, top_n: int = 5) -> List:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance.rerank(query, docs, top_n)