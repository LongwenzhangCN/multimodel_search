import faiss
import numpy as np
import json
import os

class FaissIndexManager:
    def __init__(self, dimension, metric="cosine"):
        self.dimension = dimension
        self.metric = metric
        self.index = faiss.IndexFlatIP(dimension)
        self.mappings = []

    def add_vectors(self, vectors, mappings):
        vectors = np.array(vectors).astype('float32')
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.mappings.extend(mappings)

    def search(self, query_vector, top_k=5):
        query_vector = np.array([query_vector]).astype('float32')
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        return indices[0], distances[0]

    def save(self, index_path, mapping_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.mappings, f, ensure_ascii=False, indent=2)
        print(f"✅ 索引已保存: {index_path} (共 {self.index.ntotal} 条)")

    def load(self, index_path, mapping_path):
        if not os.path.exists(index_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
        print(f"✅ 索引已加载: {index_path} (共 {self.index.ntotal} 条)")
        return True