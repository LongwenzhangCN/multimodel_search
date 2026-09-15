cd /home/cfw/Project/multimodal_search

# 1. 创建 models 目录
mkdir -p models

# 2. 临时禁用代理并下载
export ALL_PROXY=''
export HTTP_PROXY=''
export HTTPS_PROXY=''

# 3. 使用 git 下载（最稳定）
cd models
git lfs install
git clone https://hf-mirror.com/openai/clip-vit-large-patch14
git clone https://hf-mirror.com/OFA-Sys/chinese-clip-vit-large-patch14
git clone https://hf-mirror.com/facebook/dinov2-large
git clone https://hf-mirror.com/facebook/convnext-large-224-22k

echo "✅ 下载完成！"