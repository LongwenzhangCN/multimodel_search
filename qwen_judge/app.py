import os
import sys
import time
import glob
import traceback
import torch
import gradio as gr
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_judge.encoder import QwenJudgeEncoder
from config import QWEN_MODELS, DEFAULT_QWEN_MODEL

print(" 正在初始化 Qwen 判别系统...")
system_ready = False
judge = None
current_model_key = None

def initialize_judge(model_key):
    """初始化或切换模型"""
    global judge, system_ready, current_model_key
    
    try:
        if judge is not None:
            # 卸载当前模型，释放显存
            del judge
            judge = None
            torch.cuda.empty_cache()
            print(f"🗑️ 已卸载旧模型: {current_model_key}")
        
        # 加载新模型
        judge = QwenJudgeEncoder(model_key=model_key)
        current_model_key = model_key
        system_ready = True
        model_name = QWEN_MODELS[model_key]["display_name"]
        print(f"✅ 系统就绪，当前模型: {model_name}")
        return True, f"✅ {model_name} 加载成功！显存已分配。"
    except Exception as e:
        system_ready = False
        error_msg = f"❌ 模型加载失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return False, error_msg

def unload_judge():
    """卸载模型，释放显存"""
    global judge, system_ready, current_model_key
    
    if judge is not None:
        del judge
        judge = None
        torch.cuda.empty_cache()
        system_ready = False
        msg = f"🗑️ 已卸载模型: {current_model_key}，显存已完全释放！"
        current_model_key = None
        print(msg)
        return msg
    return "️ 当前没有加载任何模型。"

# ⚠️ 关键修改：启动时绝对不自动加载模型！
# if DEFAULT_QWEN_MODEL in QWEN_MODELS:
#     success, msg = initialize_judge(DEFAULT_QWEN_MODEL)

def judge_with_description(image, description):
    """单图判别（用于调试 Prompt）"""
    if not system_ready: return "⚠️ 未就绪", "❌ 系统未就绪，请先选择并加载模型！"
    if image is None: return "⚠️ 无图片", "❌ 请先上传图片"
    
    try:
        result = judge.judge_single_image(image, description)
        match_emoji = "✅" if result["match"] else "❌"
        log_text = f"{match_emoji} **结果**: {'符合' if result['match'] else '不符合'}\n"
        log_text += f" **置信度**: {result['confidence']}\n"
        log_text += f"💡 **理由**: {result['reason']}\n\n"
        log_text += f"📄 **原始回答**:\n{result['raw_response']}"
        # 注意：这里必须返回字符串，不能返回布尔值，因为 UI 改成了 Textbox
        return "✅ 符合" if result["match"] else "❌ 不符合", log_text
    except Exception as e:
        return "❌ 失败", f"❌ 失败: {e}"

def batch_judge_folder(folder_path, description, output_txt_path, top_k=10, progress=gr.Progress()):
    """批量判别文件夹中的所有图片（仅文字），支持图片预览"""
    if not system_ready:
        return "❌ 系统未就绪，请先在上方选择并加载 Qwen 模型！", [], ""
    if not os.path.isdir(folder_path):
        return f"❌ 目录不存在: {folder_path}", [], ""
    if not description.strip():
        return "⚠️ 请输入判别描述", [], ""
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_paths:
        return f"⚠️ 目录下未找到图片", [], ""
    
    log_text = f"📂 **扫描完成**: 共找到 {len(image_paths)} 张图片\n"
    log_text += f"🔍 **筛选条件**: \"{description}\"\n"
    log_text += f"🔝 **显示 Top**: {top_k}\n"
    log_text += f"💾 **结果保存**: {output_txt_path}\n\n"
    log_text += "⏳ **开始批量判别...**\n\n"
    
    all_results = []
    failed_count = 0
    start_time = time.time()
    
    os.makedirs(os.path.dirname(output_txt_path) if os.path.dirname(output_txt_path) else '.', exist_ok=True)
    
    try:
        for i, img_path in enumerate(image_paths):
            try:
                progress((i + 1) / len(image_paths), desc=f"处理中: {i+1}/{len(image_paths)}")
                image = Image.open(img_path).convert('RGB')
                result = judge.judge_single_image(image, description)
                
                if result["match"]:
                    confidence_score = {"高": 3, "中": 2, "低": 1}.get(result["confidence"], 0)
                    all_results.append({
                        "path": img_path, "confidence": result["confidence"],
                        "confidence_score": confidence_score, "reason": result["reason"], "index": i
                    })
            except Exception as e:
                failed_count += 1
                log_text += f"⚠️ 跳过图片 {i+1}: {os.path.basename(img_path)} (错误: {str(e)[:50]})\n"
                continue
        
        all_results.sort(key=lambda x: (-x["confidence_score"], x["index"]))
        top_results = all_results[:top_k]
        
        with open(output_txt_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"# 判别条件: {description}\n# 总图片数: {len(image_paths)}\n# 匹配总数: {len(all_results)}\n\n")
            for idx, res in enumerate(all_results, 1):
                f_out.write(f"{idx}. {res['path']} | 置信度: {res['confidence']} | {res['reason']}\n")
        
        preview_images = []
        for res in top_results:
            try:
                img = Image.open(res["path"]).convert('RGB')
                caption = f"{os.path.basename(res['path'])}\n[{res['confidence']}] {res['reason'][:50]}"
                preview_images.append((img, caption))
            except: continue
        
        elapsed = time.time() - start_time
        log_text += f"\n✅ **批量处理完成！**\n📊 **统计信息**:\n   总图片数: {len(image_paths)}\n   符合条件: {len(all_results)}\n   显示 Top: {len(top_results)}\n   处理失败: {failed_count}\n   总耗时: {elapsed:.2f} 秒\n   平均速度: {len(image_paths)/elapsed:.2f} 张/秒\n\n💾 **全部结果已保存至**: `{output_txt_path}`\n"
        preview_text = "\n".join([f"{i+1}. {res['path']} ({res['confidence']})" for i, res in enumerate(top_results)])
        return log_text, preview_images, preview_text
    except Exception as e:
        return f"❌ 批量处理失败:\n{str(e)}\n\n{traceback.format_exc()}", [], ""

def batch_judge_with_reference(reference_img, description, folder_path, output_txt_path, top_k=10, progress=gr.Progress()):
    """批量判别：使用参考图+文字描述"""
    if not system_ready:
        return "❌ 系统未就绪，请先在上方选择并加载 Qwen 模型！", [], ""
    if reference_img is None: return "⚠️ 请上传参考图片", [], ""
    if not os.path.isdir(folder_path): return f"❌ 目录不存在: {folder_path}", [], ""
    if not description.strip(): return "️ 请输入文字描述", [], ""
    
    target_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        target_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        target_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    if not target_paths: return f"⚠️ 目录下未找到图片", [], ""
    
    log_text = f"📂 **扫描完成**: 共找到 {len(target_paths)} 张图片\n️ **参考图**: 已上传\n **描述**: \"{description}\"\n **显示 Top**: {top_k}\n💾 **结果保存**: {output_txt_path}\n\n **开始批量判别（参考图+文字）...**\n\n"
    all_results = []
    failed_count = 0
    start_time = time.time()
    os.makedirs(os.path.dirname(output_txt_path) if os.path.dirname(output_txt_path) else '.', exist_ok=True)
    
    try:
        for i, img_path in enumerate(target_paths):
            try:
                progress((i + 1) / len(target_paths), desc=f"处理中: {i+1}/{len(target_paths)}")
                target_image = Image.open(img_path).convert('RGB')
                result = judge.judge_image_match(reference_img, target_image, description)
                if result["match"]:
                    confidence_score = {"高": 3, "中": 2, "低": 1}.get(result["confidence"], 0)
                    all_results.append({"path": img_path, "confidence": result["confidence"], "confidence_score": confidence_score, "reason": result["reason"], "index": i})
            except Exception as e:
                failed_count += 1
                log_text += f"⚠️ 跳过图片 {i+1}: {os.path.basename(img_path)}\n"
                continue
        
        all_results.sort(key=lambda x: (-x["confidence_score"], x["index"]))
        top_results = all_results[:top_k]
        
        with open(output_txt_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"# 判别条件: {description}\n# 参考图: 已提供\n# 总图片数: {len(target_paths)}\n# 匹配总数: {len(all_results)}\n\n")
            for idx, res in enumerate(all_results, 1):
                f_out.write(f"{idx}. {res['path']} | 置信度: {res['confidence']} | {res['reason']}\n")
        
        preview_images = []
        for res in top_results:
            try:
                img = Image.open(res["path"]).convert('RGB')
                preview_images.append((img, f"{os.path.basename(res['path'])}\n{res['confidence']} | {res['reason'][:50]}"))
            except: continue
        
        elapsed = time.time() - start_time
        log_text += f"\n✅ **批量处理完成！**\n **统计信息**:\n   总图片数: {len(target_paths)}\n   符合条件: {len(all_results)}\n   显示 Top: {len(top_results)}\n   处理失败: {failed_count}\n   总耗时: {elapsed:.2f} 秒\n\n💾 **全部结果已保存至**: `{output_txt_path}`\n"
        preview_text = "\n".join([f"{i+1}. {res['path']} ({res['confidence']})" for i, res in enumerate(top_results)])
        return log_text, preview_images, preview_text
    except Exception as e:
        return f"❌ 批量处理失败:\n{str(e)}\n\n{traceback.format_exc()}", [], ""

def change_model(model_key, progress=gr.Progress()):
    """切换模型的回调函数"""
    progress(0, desc=f"正在加载 {QWEN_MODELS[model_key]['display_name']}...")
    success, msg = initialize_judge(model_key)
    progress(1.0, desc="完成")
    return msg

# ========== Gradio 界面 ==========
# 注意：Gradio 6.0 中 theme 参数移到了 launch() 方法中
with gr.Blocks(title="Qwen 智能判别系统") as demo:
    gr.Markdown("# 🔍 Qwen 智能判别系统")
    gr.Markdown("**功能**：支持 Qwen2-VL 和 Qwen3-VL 动态切换，单图调试、纯文字批量筛选、参考图+文字批量筛选")
    gr.Markdown("⚠️ **显存提示**：请在需要时加载模型，用完后点击卸载以释放显存。")
    
    # ========== 顶部控制区：始终显示 ==========
    with gr.Row():
        with gr.Column(scale=1):
            model_choices = [(v["display_name"], k) for k, v in QWEN_MODELS.items()]
            model_dropdown = gr.Dropdown(
                choices=model_choices,
                value=DEFAULT_QWEN_MODEL if DEFAULT_QWEN_MODEL in QWEN_MODELS else model_choices[0][1],
                label="🤖 选择 Qwen 模型",
                info="选择后要点击'加载模型'按钮"
            )
            
            with gr.Row():
                load_model_btn = gr.Button(" 加载/切换模型", variant="primary")
                unload_model_btn = gr.Button("🗑️ 卸载模型 (释放显存)", variant="stop")
                
            model_status = gr.Textbox(label="模型状态", lines=2, interactive=False, value="⚠️ 当前未加载任何模型，显存空闲。")
            
            load_model_btn.click(change_model, inputs=[model_dropdown], outputs=[model_status])
            unload_model_btn.click(unload_judge, outputs=[model_status])
    
    # ========== 功能 Tabs：始终显示，但在函数内部拦截未加载状态 ==========
    with gr.Tabs():
        # 标签页 1: 参考图+文字 批量判别
        with gr.TabItem("🖼️+📝 参考图+文字 批量判别"):
             gr.Markdown("### 自动化筛选：使用参考图片+文字描述，批量查找相似图片")
             with gr.Row():
                 with gr.Column(scale=1):
                     ref_img_input = gr.Image(label="🖼️ 上传参考图片", type="pil", height=250)
                     ref_desc_input = gr.Textbox(label="📝 文字描述（结合参考图）", placeholder="例如：图片中是否存在参考图右下角的这只狗？", lines=3)
                     folder_input = gr.Textbox(label=" 待搜索的图片文件夹路径", value="/root/autodl-tmp/multimodal_search/data/images")
                     output_txt_input = gr.Textbox(label="💾 结果保存路径 (.txt)", value="/root/autodl-tmp/multimodal_search/data/results/matched_with_ref.txt")
                     top_k_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="🔝 显示前 N 个匹配结果")
                     btn_batch_ref = gr.Button(" 开始批量筛选", variant="primary", size="lg")
                 with gr.Column(scale=2):
                     # 注意：移除了 show_copy_button=True
                     log_output = gr.Textbox(label="📋 处理日志", lines=10)
                     gallery_output = gr.Gallery(label="️ 匹配结果预览", columns=3, height="auto", object_fit="contain")
                     preview_output = gr.Textbox(label="📄 匹配路径列表", lines=5)
             btn_batch_ref.click(batch_judge_with_reference, inputs=[ref_img_input, ref_desc_input, folder_input, output_txt_input, top_k_slider], outputs=[log_output, gallery_output, preview_output])
                    
        # 标签页 2: 纯文字批量判别
        with gr.TabItem("📝 纯文字 批量判别"):
            gr.Markdown("### 自动化筛选：仅使用文字描述，批量查找符合要求的图片")
            with gr.Row():
                with gr.Column(scale=1):
                    folder_input_text = gr.Textbox(label=" 图片文件夹路径", value="/root/autodl-tmp/multimodal_search/data/images")
                    desc_input_text = gr.Textbox(label=" 筛选描述", placeholder="例如：图片中是否存在镜头起雾的情况", lines=3)
                    output_txt_input_text = gr.Textbox(label="💾 结果保存路径 (.txt)", value="/root/autodl-tmp/multimodal_search/data/results/matched_text_only.txt")
                    top_k_slider_text = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="🔝 显示前 N 个匹配结果")
                    btn_batch_text = gr.Button("🚀 开始批量筛选", variant="primary", size="lg")
                with gr.Column(scale=2):
                    # 注意：移除了 show_copy_button=True
                    log_output_text = gr.Textbox(label="📋 处理日志与进度", lines=10)
                    gallery_output_text = gr.Gallery(label="️ 匹配结果图片预览", columns=3, height="auto", object_fit="contain")
                    preview_output_text = gr.Textbox(label="📄 匹配路径列表", lines=5)
            btn_batch_text.click(batch_judge_folder, inputs=[folder_input_text, desc_input_text, output_txt_input_text, top_k_slider_text], outputs=[log_output_text, gallery_output_text, preview_output_text]) 
                     
        # 标签页 3: 单图调试
        with gr.TabItem("🔬 单图调试"):
            gr.Markdown("### 用于调试 Prompt 和测试模型反应")
            with gr.Row():
                with gr.Column(scale=1):
                    img1 = gr.Image(label="📷 上传图片", type="pil", height=300)
                    desc1 = gr.Textbox(label="🔍 描述要求", placeholder="例如：镜头起雾", lines=2)
                    btn1 = gr.Button("🔍 判别", variant="primary")
                with gr.Column(scale=2):
                    # 注意：改成了 Textbox，避免 Gradio 6 的 Label 报错
                    result1 = gr.Textbox(label="结果", interactive=False)
                    info1 = gr.Textbox(label="📄 详细回答", lines=10)
            btn1.click(judge_with_description, inputs=[img1, desc1], outputs=[result1, info1])

if __name__ == "__main__":
    # 注意：theme 参数移到了 launch() 方法中
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, inbrowser=False, theme=gr.themes.Soft())