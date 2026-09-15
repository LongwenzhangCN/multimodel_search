import os
import hashlib
from utils.faiss_utils import FaissIndexManager

class IndexManager:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_index_path(self, image_dir, model_key):
        folder_name = os.path.basename(os.path.normpath(image_dir))
        hash_id = hashlib.md5(f"{folder_name}_{model_key}".encode()).hexdigest()[:8]
        index_id = f"{folder_name}_{model_key}_{hash_id}"
        index_dir = os.path.join(self.cache_dir, index_id)
        os.makedirs(index_dir, exist_ok=True)
        return {
            "index_file": os.path.join(index_dir, "index.faiss"),
            "mapping_file": os.path.join(index_dir, "mapping.json"),
            "index_id": index_id
        }

    def index_exists(self, image_dir, model_key):
        paths = self.get_index_path(image_dir, model_key)
        return os.path.exists(paths["index_file"]) and os.path.exists(paths["mapping_file"])

    def load_index(self, image_dir, model_key, feature_dim):
        paths = self.get_index_path(image_dir, model_key)
        if not self.index_exists(image_dir, model_key):
            return None, None
        manager = FaissIndexManager(feature_dim, "cosine")
        if manager.load(paths["index_file"], paths["mapping_file"]):
            return manager, paths["index_id"]
        return None, None

    def save_index(self, image_dir, model_key, manager):
        paths = self.get_index_path(image_dir, model_key)
        manager.save(paths["index_file"], paths["mapping_file"])
        return paths["index_id"]

    def list_cached_indices(self):
        indices = []
        if not os.path.exists(self.cache_dir):
            return indices
        for folder in os.listdir(self.cache_dir):
            folder_path = os.path.join(self.cache_dir, folder)
            if os.path.isdir(folder_path):
                index_file = os.path.join(folder_path, "index.faiss")
                if os.path.exists(index_file):
                    import faiss
                    idx = faiss.read_index(index_file)
                    parts = folder.rsplit('_', 2)
                    indices.append({"id": folder, "folder_name": parts[0] if len(parts)>=1 else folder, "model_key": parts[1] if len(parts)>=2 else "", "size": idx.ntotal})
        return indices