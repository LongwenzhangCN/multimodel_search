import os
import sys
import time
import glob
import traceback
import gradio as gr
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_judge.encoder import QwenJudgeEncoder

print("🚀 正在初始化 Qwen 判别系统...")
system_ready = False
judge = None
try:
    judge = QwenJudgeEncoder()
    system_ready = True
except Exception as e:
    print(f"❌ Qwen 模型加载失败: {e}")
    traceback.print_exc()

def judge_with_description(image, description):
    """单图判别（用于调试 Prompt）"""
    if not system_ready: return None, "❌ 系统未就绪"
    if image is None: return None, "⚠️ 请上传图片"
    
    try:
        result = judge.judge_single_image(image, description)
        match_emoji = "✅" if result["match"] else "❌"
        log_text = f"{match_emoji} **结果**: {'符合' if result['match'] else '不符合'}\n"
        log_text += f" **置信度**: {result['confidence']}\n"
        log_text += f"💡 **理由**: {result['reason']}\n\n"
        log_text += f"📄 **原始回答**:\n{result['raw_response']}"
        return result["match"], log_text
    except Exception as e:
        return None, f"❌ 失败: {e}"

def batch_judge_folder(folder_path, description, output_txt_path, top_k=10, progress=gr.Progress()):
    """批量判别文件夹中的所有图片（仅文字），支持图片预览"""
    if not system_ready:
        return "❌ 系统未就绪", [], ""
    if not os.path.isdir(folder_path):
        return f"❌ 目录不存在: {folder_path}", [], ""
    if not description.strip():
        return "⚠️ 请输入判别描述", [], ""
    
    # 1. 扫描图片
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_paths:
        return f"️ 目录下未找到图片", [], ""
    
    log_text = f"📂 **扫描完成**: 共找到 {len(image_paths)} 张图片\n"
    log_text += f" **筛选条件**: \"{description}\"\n"
    log_text += f"🔝 **显示 Top**: {top_k}\n"
    log_text += f"💾 **结果保存**: {output_txt_path}\n\n"
    log_text += " **开始批量判别...**\n\n"
    
    all_results = []  # 存储所有匹配结果
    failed_count = 0
    start_time = time.time()
    
    # 2. 确保输出目录存在
    os.makedirs(os.path.dirname(output_txt_path) if os.path.dirname(output_txt_path) else '.', exist_ok=True)
    
    # 3. 循环处理
    try:
        for i, img_path in enumerate(image_paths):
            try:
                # 更新进度条
                progress((i + 1) / len(image_paths), desc=f"处理中: {i+1}/{len(image_paths)}")
                
                # 加载并推理
                image = Image.open(img_path).convert('RGB')
                result = judge.judge_single_image(image, description)
                
                # 如果匹配，记录结果
                if result["match"]:
                    # 将置信度转换为数值分数用于排序 (高=3, 中=2, 低=1)
                    confidence_score = {"高": 3, "中": 2, "低": 1}.get(result["confidence"], 0)
                    all_results.append({
                        "path": img_path,
                        "confidence": result["confidence"],
                        "confidence_score": confidence_score,
                        "reason": result["reason"],
                        "index": i
                    })
                        
            except Exception as e:
                failed_count += 1
                log_text += f"⚠️ 跳过图片 {i+1}: {os.path.basename(img_path)} (错误: {str(e)[:50]})\n"
                continue
        
        # 4. 排序：先按置信度分数降序，再按原始顺序
        all_results.sort(key=lambda x: (-x["confidence_score"], x["index"]))
        
        # 取 Top K
        top_results = all_results[:top_k]
        
        # 5. 保存所有结果到 TXT
        with open(output_txt_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"# 判别条件: {description}\n")
            f_out.write(f"# 总图片数: {len(image_paths)}\n")
            f_out.write(f"# 匹配总数: {len(all_results)}\n\n")
            
            for idx, res in enumerate(all_results, 1):
                f_out.write(f"{idx}. {res['path']} | 置信度: {res['confidence']} | {res['reason']}\n")
        
        # 6. 加载 Top K 图片用于 Gallery 预览
        preview_images = []
        for res in top_results:
            try:
                img = Image.open(res["path"]).convert('RGB')
                caption = f"{os.path.basename(res['path'])}\n[{res['confidence']}] {res['reason'][:50]}"
                preview_images.append((img, caption))
            except Exception as e:
                log_text += f"⚠️ 无法加载预览图: {os.path.basename(res['path'])}\n"
                continue
        
        # 7. 生成总结日志
        elapsed = time.time() - start_time
        log_text += f"\n✅ **批量处理完成！**\n"
        log_text += f"📊 **统计信息**:\n"
        log_text += f"   总图片数: {len(image_paths)}\n"
        log_text += f"   符合条件: {len(all_results)}\n"
        log_text += f"   显示 Top: {len(top_results)}\n"
        log_text += f"   处理失败: {failed_count}\n"
        log_text += f"   总耗时: {elapsed:.2f} 秒\n"
        log_text += f"   平均速度: {len(image_paths)/elapsed:.2f} 张/秒\n\n"
        log_text += f"💾 **全部结果已保存至**: `{output_txt_path}`\n"
        
        # 生成文本列表预览
        preview_text = "\n".join([f"{i+1}. {res['path']} ({res['confidence']})" for i, res in enumerate(top_results)])
        
        return log_text, preview_images, preview_text
        
    except Exception as e:
        return f"❌ 批量处理失败:\n{str(e)}\n\n{traceback.format_exc()}", [], ""
def batch_judge_with_reference(reference_img, description, folder_path, output_txt_path, top_k=10, progress=gr.Progress()):
    """批量判别：使用参考图+文字描述"""
    if not system_ready:
        return "❌ 系统未就绪", [], ""
    if reference_img is None:
        return "⚠️ 请上传参考图片", [], ""
    if not os.path.isdir(folder_path):
        return f"❌ 目录不存在: {folder_path}", [], ""
    if not description.strip():
        return "⚠️ 请输入文字描述", [], ""
    
    # 扫描目标图片
    target_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        target_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        target_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not target_paths:
        return f"⚠️ 目录下未找到图片", [], ""
    
    log_text = f"📂 **扫描完成**: 共找到 {len(target_paths)} 张图片\n"
    log_text += f"🖼️ **参考图**: 已上传\n"
    log_text += f"📝 **描述**: \"{description}\"\n"
    log_text += f"🔝 **显示 Top**: {top_k}\n"
    log_text += f" **结果保存**: {output_txt_path}\n\n"
    log_text += "⏳ **开始批量判别（参考图+文字）...**\n\n"
    
    all_results = []  # 存储所有匹配结果
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
                    # 将置信度转换为数值分数用于排序
                    confidence_score = {"高": 3, "中": 2, "低": 1}.get(result["confidence"], 0)
                    all_results.append({
                        "path": img_path,
                        "confidence": result["confidence"],
                        "confidence_score": confidence_score,
                        "reason": result["reason"],
                        "index": i  # 记录原始顺序
                    })
                    
            except Exception as e:
                failed_count += 1
                log_text += f"⚠️ 跳过图片 {i+1}: {os.path.basename(img_path)}\n"
                continue
        
        # 排序：先按置信度分数降序，再按原始顺序
        all_results.sort(key=lambda x: (-x["confidence_score"], x["index"]))
        
        # 取 Top K
        top_results = all_results[:top_k]
        
        # 保存所有结果到 TXT
        with open(output_txt_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"# 判别条件: {description}\n")
            f_out.write(f"# 参考图: 已提供\n")
            f_out.write(f"# 总图片数: {len(target_paths)}\n")
            f_out.write(f"# 匹配总数: {len(all_results)}\n\n")
            
            for idx, res in enumerate(all_results, 1):
                f_out.write(f"{idx}. {res['path']} | 置信度: {res['confidence']} | {res['reason']}\n")
        
        # 加载 Top K 图片用于预览
        preview_images = []
        for res in top_results:
            try:
                img = Image.open(res["path"]).convert('RGB')
                # 可以在图片上添加标注（可选）
                preview_images.append((img, f"{os.path.basename(res['path'])}\n{res['confidence']} | {res['reason'][:50]}"))
            except:
                continue
        
        elapsed = time.time() - start_time
        log_text += f"\n✅ **批量处理完成！**\n"
        log_text += f"📊 **统计信息**:\n"
        log_text += f"   总图片数: {len(target_paths)}\n"
        log_text += f"   符合条件: {len(all_results)}\n"
        log_text += f"   显示 Top: {len(top_results)}\n"
        log_text += f"   处理失败: {failed_count}\n"
        log_text += f"   总耗时: {elapsed:.2f} 秒\n"
        log_text += f"   平均速度: {len(target_paths)/elapsed:.2f} 张/秒\n\n"
        log_text += f"💾 **全部结果已保存至**: `{output_txt_path}`\n"
        
        # 生成预览文本列表
        preview_text = "\n".join([f"{i+1}. {res['path']} ({res['confidence']})" for i, res in enumerate(top_results)])
        
        return log_text, preview_images, preview_text
        
    except Exception as e:
        return f"❌ 批量处理失败:\n{str(e)}\n\n{traceback.format_exc()}", [], ""

# # 修改 UI 部分
# with gr.Blocks(title="Qwen2-VL 智能判别系统", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# 🔍 Qwen2-VL 智能判别系统")
    
#     if not system_ready:
#         gr.Markdown("### ❌ 系统初始化失败")
#     else:
#         with gr.Tabs():
 
#             # ... (其他标签页保持不变)
with gr.Blocks(title="Qwen2-VL 智能判别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Qwen2-VL 智能判别系统")
    gr.Markdown("**功能**：支持单图调试、纯文字批量筛选、**参考图+文字批量筛选**")
    
    if not system_ready:
        gr.Markdown("### ❌ 系统初始化失败，请检查终端日志。")
    else:
        with gr.Tabs():
            # ========== 标签页 1: 参考图+文字 批量判别 (核心功能) ==========
            with gr.TabItem("🖼️+📝 参考图+文字 批量判别"):
                 gr.Markdown("### 自动化筛选：使用参考图片+文字描述，批量查找相似图片")
                 
                 with gr.Row():
                     with gr.Column(scale=1):
                         ref_img_input = gr.Image(label="️ 上传参考图片", type="pil", height=250)
                         ref_desc_input = gr.Textbox(
                             label=" 文字描述（结合参考图）",
                             placeholder="例如：图片中是否存在参考图右下角的这只狗？",
                             lines=3
                         )
                         folder_input = gr.Textbox(
                             label="📂 待搜索的图片文件夹路径",
                             value="/root/autodl-tmp/multimodal_search/data/images"
                         )
                         output_txt_input = gr.Textbox(
                             label="💾 结果保存路径 (.txt)",
                             value="/root/autodl-tmp/multimodal_search/data/results/matched_with_ref.txt"
                         )
                         top_k_slider = gr.Slider(
                             minimum=1, maximum=50, value=10, step=1,
                             label="🔝 显示前 N 个匹配结果",
                             info="从所有匹配结果中选择前 N 个展示（按置信度排序）"
                         )
                         btn_batch_ref = gr.Button("🚀 开始批量筛选", variant="primary", size="lg")
                     
                     with gr.Column(scale=2):
                         log_output = gr.Textbox(label="📋 处理日志", lines=10, show_copy_button=True)
                         # 图片预览 Gallery
                         gallery_output = gr.Gallery(
                             label="️ 匹配结果预览",
                             show_label=True,
                             columns=3,
                             height="auto",
                             object_fit="contain"
                         )
                         # 文本列表
                         preview_output = gr.Textbox(label=" 匹配路径列表", lines=5)
                 
                 btn_batch_ref.click(
                     batch_judge_with_reference,
                     inputs=[ref_img_input, ref_desc_input, folder_input, output_txt_input, top_k_slider],
                     outputs=[log_output, gallery_output, preview_output]
                 )
                        
            # ========== 标签页 2: 纯文字批量判别 ==========
            with gr.TabItem("📝 纯文字 批量判别"):
                gr.Markdown("### 自动化筛选：仅使用文字描述，批量查找符合要求的图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        folder_input_text = gr.Textbox(
                            label="📂 图片文件夹路径",
                            value="/root/autodl-tmp/multimodal_search/data/images"
                        )
                        desc_input_text = gr.Textbox(
                            label="🔍 筛选描述",
                            placeholder="例如：图片中是否存在镜头起雾的情况",
                            lines=3
                        )
                        output_txt_input_text = gr.Textbox(
                            label=" 结果保存路径 (.txt)",
                            value="/root/autodl-tmp/multimodal_search/data/results/matched_text_only.txt"
                        )
                        # 新增：Top K 滑块
                        top_k_slider_text = gr.Slider(
                            minimum=1, maximum=50, value=10, step=1,
                            label=" 显示前 N 个匹配结果",
                            info="从所有匹配结果中选择前 N 个展示（按置信度排序）"
                        )
                        btn_batch_text = gr.Button("🚀 开始批量筛选", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        log_output_text = gr.Textbox(label=" 处理日志与进度", lines=10, show_copy_button=True)
                        # 新增：图片预览 Gallery
                        gallery_output_text = gr.Gallery(
                            label="🖼️ 匹配结果图片预览",
                            show_label=True,
                            columns=3,
                            height="auto",
                            object_fit="contain"
                        )
                        preview_output_text = gr.Textbox(label="📄 匹配路径列表", lines=5)
                
                # 更新 inputs 和 outputs
                btn_batch_text.click(
                    batch_judge_folder,
                    inputs=[folder_input_text, desc_input_text, output_txt_input_text, top_k_slider_text],
                    outputs=[log_output_text, gallery_output_text, preview_output_text]
                ) 
                         
            # ========== 标签页 3: 单图调试 ==========
            with gr.TabItem("🔬 单图调试"):
                gr.Markdown("### 用于调试 Prompt 和测试模型反应")
                with gr.Row():
                    with gr.Column(scale=1):
                        img1 = gr.Image(label="上传图片", type="pil", height=300)
                        desc1 = gr.Textbox(label="描述要求", placeholder="例如：镜头起雾", lines=2)
                        btn1 = gr.Button("🔍 判别", variant="primary")
                    with gr.Column(scale=2):
                        result1 = gr.Label(label="结果")
                        info1 = gr.Textbox(label="详细回答", lines=10)
                btn1.click(judge_with_description, inputs=[img1, desc1], outputs=[result1, info1])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)