import os
from huggingface_hub import hf_hub_download

# 设置国内镜像（必须）
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 禁用代理
os.environ['ALL_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

base_dir = "/home/cfw/Project/multimodal_search/models"

# 要下载的模型列表
models = {
    "chinese-clip-vit-large-patch14": {
        "repo": "OFA-Sys/chinese-clip-vit-large-patch14",
        "files": ["pytorch_model.bin", "config.json", "preprocessor_config.json", "vocab.txt"]
    },
    "clip-vit-large-patch14": {
        "repo": "openai/clip-vit-large-patch14",
        "files": ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
    },
    "dinov2-large": {
        "repo": "facebook/dinov2-large",
        "files": ["model.safetensors", "config.json", "preprocessor_config.json"]
    },
    "convnext-large-224-22k": {
        "repo": "facebook/convnext-large-224-22k",
        "files": ["model.safetensors", "config.json", "preprocessor_config.json"]
    }
}

print(" 开始下载模型（带进度条）...\n")

for model_name, config in models.items():
    print(f"\n{'='*60}")
    print(f"📦 下载: {model_name}")
    print(f"{'='*60}")
    
    target_dir = os.path.join(base_dir, model_name)
    os.makedirs(target_dir, exist_ok=True)
    
    for filename in config["files"]:
        filepath = os.path.join(target_dir, filename)
        
        # 检查是否已存在且大小合理
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000000:  # > 1MB
            print(f"✅ {filename:30s} 已存在，跳过")
            continue
        
        print(f"⏳ 下载中: {filename:30s} ...")
        try:
            hf_hub_download(
                repo_id=config["repo"],
                filename=filename,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                force_download=True  # 强制重新下载
            )
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✅ {filename:30s} 完成 ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ {filename:30s} 失败: {e}")

print("\n🎉 所有模型下载完成！")