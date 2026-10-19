import os
import sys
import glob
import time
import traceback
import gradio as gr
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS, DEFAULT_IMAGE_DIR, CACHE_DIR
from utils.model_manager import ModelManager
from unified_search.index_manager import IndexManager

import shutil


# 全局状态管理
model_manager = ModelManager()
index_manager = IndexManager(CACHE_DIR)
loaded_models_state = {}  # 跟踪已加载的模型

def check_status():
    """检查所有模型文件状态"""
    status = "📂 **模型文件状态：**\n\n"
    for k, v in MODELS.items():
        p = v["local_path"]
        config_exists = os.path.exists(os.path.join(p, "config.json"))
        if config_exists:
            # 检查大文件
            weight_files = [f for f in os.listdir(p) if f.endswith(('.bin', '.safetensors'))]
            total_size = sum(os.path.getsize(os.path.join(p, f)) for f in weight_files) / 1024 / 1024
            status += f"✅ **{v['display_name']}** (`{k}`)\n"
            status += f"   路径: `{p}`\n"
            status += f"   权重: {len(weight_files)} 个文件, 总计 {total_size:.1f} MB\n"
            if k in loaded_models_state:
                status += f"   状态:  已加载到显存\n"
            else:
                status += f"   状态:  未加载\n"
        else:
            status += f"❌ **{v['display_name']}** (`{k}`)\n"
            status += f"   路径: `{p}` (配置文件缺失)\n"
        status += "\n"
    return status

def load_selected_model(model_key, log_output):
    """加载选中的模型"""
    if not model_key:
        return "⚠️ 请先选择一个模型", loaded_models_state
    
    if model_key not in MODELS:
        return f"❌ 未知模型: {model_key}", loaded_models_state
    
    model_config = MODELS[model_key]
    log_text = f"🔄 正在加载模型: {model_config['display_name']} ({model_key})\n"
    log_text += f"📂 路径: {model_config['local_path']}\n"
    
    try:
        # 检查是否已加载
        if model_key in loaded_models_state:
            log_text += f"✅ 模型已在显存中，无需重复加载\n"
            return log_text, loaded_models_state
        
        # 检查文件是否存在
        if not os.path.exists(os.path.join(model_config['local_path'], "config.json")):
            raise FileNotFoundError(f"模型配置文件不存在: {model_config['local_path']}")
        
        log_text += f"⏳ 开始加载模型到显存...\n"
        start_time = time.time()
        
        # 加载模型
        encoder = model_manager.load_model(model_key, model_config)
        
        if encoder is None:
            raise Exception("模型加载失败，encoder 为 None")
        
        elapsed = time.time() - start_time
        loaded_models_state[model_key] = {
            "config": model_config,
            "load_time": elapsed
        }
        
        log_text += f"✅ 模型加载成功！\n"
        log_text += f"️ 耗时: {elapsed:.2f} 秒\n"
        log_text += f" 特征维度: {model_config['feature_dim']}\n"
        log_text += f"💾 显存占用: 已加载到 GPU\n"
        
        return log_text, loaded_models_state
        
    except Exception as e:
        error_msg = f"❌ 模型加载失败:\n{str(e)}\n\n堆栈跟踪:\n{traceback.format_exc()}"
        log_text += error_msg
        return log_text, loaded_models_state

def unload_model(model_key, log_output):
    """卸载模型释放显存"""
    if model_key not in loaded_models_state:
        return f"️ 模型 {model_key} 未加载", loaded_models_state
    
    try:
        model_manager.unload_model(model_key)
        del loaded_models_state[model_key]
        import torch
        torch.cuda.empty_cache()
        
        log_text = f"✅ 模型 {model_key} 已卸载，显存已释放"
        return log_text, loaded_models_state
    except Exception as e:
        return f"❌ 卸载失败: {e}", loaded_models_state

def build_index(image_dir, model_key, progress=gr.Progress()):
    """构建索引（只针对已加载的模型）"""
    log_text = ""
    
    # 验证目录
    if not os.path.isdir(image_dir):
        return f"❌ 错误: 图片目录不存在\n路径: {image_dir}"
    
    # 验证模型
    if not model_key:
        return "⚠️ 请先选择并加载一个模型"
    
    if model_key not in loaded_models_state:
        return f"❌ 模型 {model_key} 未加载，请先点击【加载模型】按钮"
    
    if model_key not in MODELS:
        return f"❌ 未知模型: {model_key}"
    
    model_config = MODELS[model_key]
    log_text += f"🔧 开始构建索引\n"
    log_text += f"📂 图片目录: {image_dir}\n"
    log_text += f"🤖 模型: {model_config['display_name']}\n"
    log_text += f"📏 特征维度: {model_config['feature_dim']}\n\n"
    
    # 扫描图片
    log_text += "🔍 扫描图片文件...\n"
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_paths:
        return log_text + "️ 未找到支持的图片格式 (jpg, jpeg, png, bmp, webp)"
    
    log_text += f"✅ 找到 {len(image_paths)} 张图片\n\n"
    
    # 检查索引是否已存在
    if index_manager.index_exists(image_dir, model_key):
        log_text += f"⚠️ 索引已存在，将跳过构建\n"
        index_path = index_manager.get_index_path(image_dir, model_key)
        log_text += f"📂 索引ID: {index_path['index_id']}\n"
        return log_text + "\n💡 如需重建，请先删除缓存目录中的对应文件夹"
    
    # 获取编码器
    encoder = model_manager.get_encoder(model_key)
    if encoder is None:
        return log_text + "❌ 获取模型编码器失败"
    
    # 创建索引管理器
    from utils.faiss_utils import FaissIndexManager
    mgr = FaissIndexManager(model_config["feature_dim"], "cosine")
    
    # 构建索引
    log_text += " 开始提取特征并构建索引...\n\n"
    start_time = time.time()
    failed_count = 0
    
    try:
        for i, img_path in enumerate(image_paths):
            try:
                progress((i + 1) / len(image_paths), desc=f"处理中: {i+1}/{len(image_paths)}")
                
                # 提取特征
                feat = encoder.encode_image(img_path)
                
                # 添加到索引
                mgr.add_vectors([feat], [{"path": img_path, "name": os.path.basename(img_path)}])
                
            except Exception as e:
                failed_count += 1
                log_text += f"⚠️ 跳过图片 {i+1}: {os.path.basename(img_path)}\n"
                log_text += f"   错误: {str(e)[:100]}...\n"
                continue
        
        # 保存索引
        elapsed = time.time() - start_time
        idx_id = index_manager.save_index(image_dir, model_key, mgr)
        
        log_text += f"\n✅ 索引构建完成！\n"
        log_text += f"📊 统计信息:\n"
        log_text += f"   总图片数: {len(image_paths)}\n"
        log_text += f"   成功: {len(image_paths) - failed_count}\n"
        log_text += f"   失败: {failed_count}\n"
        log_text += f"   耗时: {elapsed:.2f} 秒\n"
        log_text += f"   平均速度: {len(image_paths)/elapsed:.1f} 张/秒\n"
        log_text += f"📂 索引ID: {idx_id}\n"
        log_text += f"💾 索引大小: {mgr.index.ntotal} 个向量\n"
        
    except Exception as e:
        log_text += f"\n❌ 索引构建失败:\n{str(e)}\n\n堆栈跟踪:\n{traceback.format_exc()}"
    
    return log_text

def list_cached_indices():
    """列出所有缓存的索引"""
    indices = index_manager.list_cached_indices()
    if not indices:
        return "📂 暂无缓存索引"
    
    log_text = "📂 **已缓存的索引：**\n\n"
    for idx in indices:
        log_text += f"📁 **{idx['folder_name']}**\n"
        log_text += f"   模型: {idx['model_key']}\n"
        log_text += f"   索引ID: {idx['id']}\n"
        log_text += f"   向量数: {idx['size']} 个\n\n"
    
    return log_text

def text_search(query, image_dir, model_key, top_k):
    """文本搜索图片"""
    log_text = ""
    
    if not query.strip():
        return [], "⚠️ 请输入搜索文本"
    
    if model_key not in ["clip_en", "clip_zh"]:
        return [], "⚠️ 文搜图仅支持 CLIP 模型（英文或中文）"
    
    if model_key not in loaded_models_state:
        return [], f"❌ 模型 {model_key} 未加载，请先加载模型"
    
    model_config = MODELS[model_key]
    log_text += f"🔍 文本搜索\n"
    log_text += f"查询: '{query}'\n"
    log_text += f"模型: {model_config['display_name']}\n\n"
    
    try:
        # 加载索引
        mgr, _ = index_manager.load_index(image_dir, model_key, model_config["feature_dim"])
        if mgr is None:
            return [], f"❌ 索引不存在\n请先在【索引管理】页面构建 {model_config['display_name']} 的索引"
        
        # 获取编码器
        encoder = model_manager.get_encoder(model_key)
        if encoder is None:
            return [], "❌ 获取编码器失败"
        
        # 编码文本
        log_text += "⏳ 编码查询文本...\n"
        feat = encoder.encode_text(query)
        
        # 搜索
        log_text += "⏳ 搜索相似图片...\n"
        indices, distances = mgr.search(feat, int(top_k))
        
        # 整理结果
        results = []
        info_text = log_text + f"\n✅ 找到 {len(indices)} 个结果：\n\n"
        
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            if idx != -1 and idx < len(mgr.mappings):
                mapping = mgr.mappings[idx]
                if os.path.exists(mapping["path"]):
                    results.append(Image.open(mapping["path"]))
                    info_text += f"{rank}. **{mapping['name']}**\n"
                    info_text += f"   相似度: {dist:.4f}\n"
                    info_text += f"   路径: {mapping['path']}\n\n"
        
        if not results:
            return [], info_text + "⚠️ 未找到匹配的图片"
        
        return results, info_text
        
    except Exception as e:
        error_msg = f"❌ 搜索失败:\n{str(e)}\n\n堆栈跟踪:\n{traceback.format_exc()}"
        return [], error_msg

def image_search(query_image, image_dir, model_key, top_k):
    """图像搜索相似图片"""
    log_text = ""
    
    if query_image is None:
        return [], "⚠️ 请上传查询图片"
    
    if model_key not in MODELS:
        return [], "❌ 请选择有效的模型"
    
    if model_key not in loaded_models_state:
        return [], f"❌ 模型 {model_key} 未加载，请先加载模型"
    
    model_config = MODELS[model_key]
    log_text += f"🔍 图像搜索\n"
    log_text += f"模型: {model_config['display_name']}\n\n"
    
    try:
        # 加载索引
        mgr, _ = index_manager.load_index(image_dir, model_key, model_config["feature_dim"])
        if mgr is None:
            return [], f"❌ 索引不存在\n请先在【索引管理】页面构建 {model_config['display_name']} 的索引"
        
        # 获取编码器
        encoder = model_manager.get_encoder(model_key)
        if encoder is None:
            return [], "❌ 获取编码器失败"
        
        # 保存查询图片
        temp_path = "/tmp/query_img.jpg"
        query_image.save(temp_path)
        
        # 编码图像
        log_text += "⏳ 编码查询图片...\n"
        feat = encoder.encode_image(temp_path)
        
        # 搜索
        log_text += "⏳ 搜索相似图片...\n"
        indices, distances = mgr.search(feat, int(top_k))
        
        # 整理结果
        results = []
        info_text = log_text + f"\n✅ 找到 {len(indices)} 个结果：\n\n"
        
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            if idx != -1 and idx < len(mgr.mappings):
                mapping = mgr.mappings[idx]
                if os.path.exists(mapping["path"]):
                    results.append(Image.open(mapping["path"]))
                    info_text += f"{rank}. **{mapping['name']}**\n"
                    info_text += f"   相似度: {dist:.4f}\n"
                    info_text += f"   路径: {mapping['path']}\n\n"
        
        if not results:
            return [], info_text + "️ 未找到相似的图片"
        
        return results, info_text
        
    except Exception as e:
        error_msg = f"❌ 搜索失败:\n{str(e)}\n\n堆栈跟踪:\n{traceback.format_exc()}"
        return [], error_msg

import shutil

def get_unique_filepath(filepath):
    """处理目标文件夹中的文件名冲突"""
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{ext}"
        counter += 1
    return filepath

def deduplicate_images(model_key, source_dir, duplicate_dir, threshold, progress=gr.Progress()):
    """
    基于特征相似度的图片去重（支持递归子目录，并保留原有目录结构）
    """
    log_text = ""
    
    # 1. 验证模型
    if not model_key:
        return "⚠️ 请先选择一个模型", []
    if model_key not in loaded_models_state:
        return f"❌ 模型 {model_key} 未加载，请先在【索引管理】页面加载模型", []
    
    model_config = MODELS[model_key]
    
    # 2. 验证目录
    if not os.path.isdir(source_dir):
        return f"❌ 源目录不存在: {source_dir}", []
    
    os.makedirs(duplicate_dir, exist_ok=True)
    
    # 3. 递归扫描图片 (使用 os.walk 支持多层子目录)
    log_text += f"🔍 正在递归扫描目录: {source_dir}\n"
    image_paths = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                image_paths.append(os.path.join(root, f))
    
    if not image_paths:
        return "⚠️ 源目录中未找到支持的图片格式", []
    
    log_text += f"✅ 共找到 {len(image_paths)} 张图片（包含子目录）\n"
    log_text += f"🤖 使用模型: {model_config['display_name']}\n"
    log_text += f"🎯 相似度阈值: {threshold}\n\n"
    log_text += "⏳ 开始提取特征并去重...\n\n"
    
    # 4. 初始化 Faiss 索引
    from utils.faiss_utils import FaissIndexManager
    mgr = FaissIndexManager(model_config["feature_dim"], "cosine")
    encoder = model_manager.get_encoder(model_key)
    
    kept_count = 0
    dup_count = 0
    error_count = 0
    
    start_time = time.time()
    
    # 5. 遍历去重
    for i, img_path in enumerate(image_paths):
        try:
            progress((i + 1) / len(image_paths), desc=f"去重处理中: {i+1}/{len(image_paths)}")
            
            # 提取特征
            feat = encoder.encode_image(img_path)
            
            if mgr.index.ntotal == 0:
                # 第一张图片，直接保留
                mgr.add_vectors([feat], [{"path": img_path, "name": os.path.basename(img_path)}])
                kept_count += 1
            else:
                # 在已保留的图片中搜索最相似的一张 (Top-1)
                indices, distances = mgr.search(feat, k=1)
                max_sim = distances[0]
                
                if max_sim >= threshold:
                    # 判定为重复，移动到目标文件夹
                    # 核心修改：计算相对路径，保持原有的子目录结构
                    rel_path = os.path.relpath(img_path, source_dir)
                    target_path = os.path.join(duplicate_dir, rel_path)
                    
                    # 处理文件名冲突
                    target_path = get_unique_filepath(target_path)
                    
                    # 创建对应的子目录并移动
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.move(img_path, target_path)
                    
                    dup_count += 1
                    log_text += f"🗑️ [重复] {rel_path} (相似度: {max_sim:.4f})\n"
                else:
                    # 判定为新图片，保留并加入索引
                    mgr.add_vectors([feat], [{"path": img_path, "name": os.path.basename(img_path)}])
                    kept_count += 1
                    
        except Exception as e:
            error_count += 1
            log_text += f"️ [错误] 跳过 {os.path.basename(img_path)}: {str(e)[:50]}...\n"
            continue
    
    elapsed = time.time() - start_time
    
    # 6. 生成总结报告
    log_text += f"\n✅ 去重完成！\n"
    log_text += f"📊 统计信息:\n"
    log_text += f"   总图片数: {len(image_paths)}\n"
    log_text += f"   ✅ 保留 (唯一): {kept_count}\n"
    log_text += f"   ️ 判定重复并移走: {dup_count}\n"
    log_text += f"   ⚠️ 处理失败跳过: {error_count}\n"
    log_text += f"   ️ 总耗时: {elapsed:.2f} 秒\n"
    log_text += f"   💾 重复图片已保存至: `{duplicate_dir}` (保持原目录结构)\n"
    
    return log_text

# ========== Gradio 界面 ==========

# 注意：Gradio 6.0 中 theme 参数移到了 launch() 方法中
with gr.Blocks(title="多模态搜索系统") as demo:
    gr.Markdown("# 🔍 多模态图像搜索系统")
    gr.Markdown("**功能**：支持 CLIP (中英文) + DINO (ViT)")
    
    # 全局日志区域
    with gr.Accordion("📋 系统状态", open=False):
        status_markdown = gr.Markdown(check_status())
        refresh_btn = gr.Button("🔄 刷新状态")
        refresh_btn.click(check_status, outputs=[status_markdown])
    
    with gr.Tabs():
        # ========== 页面1：索引管理 ==========
        with gr.TabItem("️ 索引管理"):
            gr.Markdown("### 📂 模型加载与索引构建")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 模型选择
                    model_dropdown = gr.Dropdown(
                        choices=[(v["display_name"], k) for k, v in MODELS.items()],
                        label="选择模型",
                        info="从列表中选择一个模型"
                    )
                    
                    # 模型操作按钮
                    with gr.Row():
                        load_btn = gr.Button("⬇️ 加载模型", variant="primary", size="sm")
                        unload_btn = gr.Button("⏏️ 卸载模型", size="sm")
                    
                    # 图片目录
                    img_dir_input = gr.Textbox(
                        label="图片文件夹路径",
                        value=DEFAULT_IMAGE_DIR,
                        placeholder="/root/autodl-tmp/data/images",
                        info="包含图片的目录路径"
                    )
                    
                    # 索引操作按钮
                    build_btn = gr.Button("🔨 构建索引", variant="primary")
                    list_btn = gr.Button("📋 查看缓存索引")
                
                with gr.Column(scale=2):
                    # 详细日志输出 (移除了 show_copy_button=True)
                    log_output = gr.Textbox(
                        label="操作日志",
                        lines=20,
                        max_lines=30,
                        info="显示详细的操作日志和错误信息"
                    )
            
            # 绑定事件
            load_btn.click(
                load_selected_model,
                inputs=[model_dropdown, log_output],
                outputs=[log_output, gr.State(loaded_models_state)]
            )
            
            unload_btn.click(
                unload_model,
                inputs=[model_dropdown, log_output],
                outputs=[log_output, gr.State(loaded_models_state)]
            )
            
            build_btn.click(
                build_index,
                inputs=[img_dir_input, model_dropdown],
                outputs=[log_output]
            )
            
            list_btn.click(
                list_cached_indices,
                outputs=[log_output]
            )
        
        # ========== 页面2：文搜图 ==========
        with gr.TabItem("📝 文本搜图"):
            gr.Markdown("### 🔤 使用文本描述搜索图片（仅支持 CLIP 模型）")
            
            with gr.Row():
                with gr.Column(scale=1):
                    clip_model_radio = gr.Radio(
                        choices=[("CLIP 英文", "clip_en"), ("CLIP 中文", "clip_zh")],
                        value="clip_en",
                        label="选择 CLIP 模型",
                        info="文搜图仅支持 CLIP 模型"
                    )
                    
                    search_dir_text = gr.Textbox(
                        label="搜索的图片目录",
                        value=DEFAULT_IMAGE_DIR
                    )
                    
                    text_query = gr.Textbox(
                        label="搜索文本",
                        placeholder="例如：a dog on the beach / 海滩上的狗",
                        lines=2
                    )
                    
                    top_k_text = gr.Slider(1, 10, value=5, step=1, label="返回数量")
                    btn_text_search = gr.Button("🔍 搜索", variant="primary")
                
                with gr.Column(scale=2):
                    gallery_text = gr.Gallery(label="搜索结果", columns=3, height=400, object_fit="cover")
                    # 移除了 show_copy_button=True
                    info_text = gr.Textbox(label="详细信息", lines=10)
            
            btn_text_search.click(
                text_search,
                inputs=[text_query, search_dir_text, clip_model_radio, top_k_text],
                outputs=[gallery_text, info_text]
            )
        
        # ========== 页面3：图搜图 ==========
        with gr.TabItem("🖼️ 图搜图"):
            gr.Markdown("### 🔍 使用图片搜索相似图片（支持所有模型）")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_model_dropdown = gr.Dropdown(
                        choices=[(v["display_name"], k) for k, v in MODELS.items()],
                        label="选择模型",
                        info="选择一个已加载的模型"
                    )
                    
                    search_dir_image = gr.Textbox(
                        label="搜索的图片目录",
                        value=DEFAULT_IMAGE_DIR
                    )
                    
                    query_image = gr.Image(label="查询图片", type="pil")
                    top_k_image = gr.Slider(1, 10, value=5, step=1, label="返回数量")
                    btn_image_search = gr.Button("🔍 搜索", variant="primary")
                
                with gr.Column(scale=2):
                    gallery_image = gr.Gallery(label="相似图片", columns=3, height=400, object_fit="cover")
                    # 移除了 show_copy_button=True
                    info_image = gr.Textbox(label="详细信息", lines=10)
            
            btn_image_search.click(
                image_search,
                inputs=[query_image, search_dir_image, image_model_dropdown, top_k_image],
                outputs=[gallery_image, info_image]
            )

        # ========== 页面4：图片去重 ==========
        with gr.TabItem("🧹 图片去重"):
            gr.Markdown("### 🔍 基于特征相似度的数据集去重")
            gr.Markdown(
                "> **原理**：逐张提取图片特征，与已保留的图片进行 Top-1 相似度比对。\n"
                "> **安全**：重复的图片不会被删除，而是**移动**到您指定的文件夹中，防止误操作。\n"
                "> **建议阈值**：CLIP 模型建议设为 `0.95` ~ `0.98`；DINO 模型建议设为 `0.90` ~ `0.95`。"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    dedup_model_dropdown = gr.Dropdown(
                        choices=[(v["display_name"], k) for k, v in MODELS.items()],
                        label="选择去重模型",
                        info="建议使用 CLIP 或 DINOv3-ViT",
                        value="clip_en" # 默认值可根据您的 config 调整
                    )
                    
                    dedup_source_dir = gr.Textbox(
                        label="📂 源图片文件夹 (需要去重的目录)",
                        value=DEFAULT_IMAGE_DIR,
                        placeholder="/path/to/your/dataset"
                    )
                    
                    dedup_target_dir = gr.Textbox(
                        label="📁 重复图片存放目录",
                        value="/root/autodl-tmp/data/duplicates",
                        placeholder="/path/to/save_duplicates"
                    )
                    
                    dedup_threshold = gr.Slider(
                        minimum=0.80, maximum=0.99, value=0.95, step=0.01,
                        label="🎯 相似度阈值 (越高要求越严格)"
                    )
                    
                    dedup_btn = gr.Button("🚀 开始执行去重", variant="primary", elem_classes="danger-btn") # 用红色警示按钮
                
                with gr.Column(scale=2):
                    dedup_log = gr.Textbox(
                        label="📋 去重日志与统计",
                        lines=20,
                        max_lines=30,
                        info="实时显示处理进度和移动记录"
                    )
            
            # 绑定事件
            dedup_btn.click(
                fn=deduplicate_images,
                inputs=[dedup_model_dropdown, dedup_source_dir, dedup_target_dir, dedup_threshold],
                outputs=[dedup_log]
            )
    
    gr.Markdown("---")
    gr.Markdown("💡 **使用提示**：\n1. 先在【索引管理】中选择并加载模型\n2. 构建索引后，索引会自动缓存\n3. 文搜图仅支持 CLIP 模型\n4. 图搜图支持所有模型")

if __name__ == "__main__":
    # 注意：theme 参数移到了 launch() 方法中
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False, theme=gr.themes.Soft())