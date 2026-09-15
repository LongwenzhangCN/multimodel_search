import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"


# ========== Qwen 配置 ==========
# 直接指向包含 config.json 和 model-*.safetensors 的目录
QWEN_MODEL_PATH = "/root/autodl-tmp/models/qwen/Qwen2-VL-7B-Instruct"

# ========== 模型本地路径配置 ==========
# ️ 请确保运行 download_models.py 后，这些路径真实存在
MODEL_BASE_DIR = "/root/autodl-tmp/multimodal_search/models"

MODELS = {
    "clip_en": {
        "local_path": os.path.join(MODEL_BASE_DIR, "clip-vit-large-patch14"),
        "feature_dim": 768,
        "type": "clip",
        "display_name": "CLIP (英文)"
    },
    "clip_zh": {
        "local_path": os.path.join(MODEL_BASE_DIR, "chinese-clip-vit-large-patch14"),
        "feature_dim": 768,
        "type": "clip",
        "display_name": "CLIP (中文)"
    },
    "dino_vit": {
        "local_path": os.path.join(MODEL_BASE_DIR, "dinov2-large"), # 使用 large 节省显存
        "feature_dim": 1024,
        "type": "dinov3",
        "backbone": "vit",
        "display_name": "DINOv2 (ViT)"
    },
    # "dino_convnext": {
    #     "local_path": os.path.join(MODEL_BASE_DIR, "convnext-large-224-22k"),
    #     "feature_dim": 1536,
    #     "type": "dinov3",
    #     "backbone": "convnext",
    #     "display_name": "DINOv3 (ConvNeXt)"
    # }
}

# ========== 路径配置 ==========
DEFAULT_IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
BASE_INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
