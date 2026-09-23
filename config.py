import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"


# ========== Qwen 模型配置 ==========
QWEN_MODEL_BASE_DIR = "/root/autodl-tmp/models"

QWEN_MODELS = {
    "qwen2_vl_7b": {
        "path": "/root/autodl-tmp/models/qwen/Qwen2-VL-7B-Instruct",  # 你之前的模型
        "display_name": "Qwen2-VL-7B",
        "model_class": "Qwen2VLForConditionalGeneration",
    },
    "qwen3_vl_8b": {  # 新增这个
        "path": "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct",
        "display_name": "Qwen3-VL-8B",
        "model_class": "Qwen3VLForConditionalGeneration",
    }
}

DEFAULT_QWEN_MODEL = "qwen2_vl_7b"  # 默认使用旧模型，启动后再切换
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
