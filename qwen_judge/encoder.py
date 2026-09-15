import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QWEN_MODEL_PATH, DEVICE

class QwenJudgeEncoder:
    def __init__(self):
        self.device = DEVICE
        print(f"🚀 正在加载 Qwen2-VL 判别模型: {QWEN_MODEL_PATH}")
        
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("✅ Qwen2-VL 判别模型加载完成！")

    @torch.no_grad()
    def judge_single_image(self, image, description):
        """判断单张图片是否符合描述"""
        prompt = (
            f"请仔细观察这张图片。\n"
            f"判断图片是否符合以下描述：\"{description}\"\n"
            f"请严格按照以下格式回答：\n"
            f"1. 结论：Yes 或 No\n"
            f"2. 置信度：高/中/低\n"
            f"3. 理由：一句话简述理由。"
        )
        return self._run_inference([image], prompt)

    @torch.no_grad()
    def judge_image_match(self, query_image, target_image, description=""):
        """判断两张图片是否匹配"""
        if description:
            prompt = (
                f"第一张图片是参考图，第二张图片是待判断的图片。\n"
                f"请判断第二张图片是否符合以下要求：\"{description}\"\n"
                f"请严格按照以下格式回答：\n"
                f"1. 结论：Yes 或 No\n"
                f"2. 置信度：高/中/低\n"
                f"3. 理由：一句话简述理由。"
            )
        else:
            prompt = (
                "第一张图片是参考图（可能包含特定标注），第二张图片是待判断的图片。\n"
                "请判断第二张图片中是否存在与第一张图片相同或相似的核心物体/场景。\n"
                "请严格按照以下格式回答：\n"
                "1. 结论：Yes 或 No\n"
                "2. 置信度：高/中/低\n"
                "3. 理由：一句话简述理由。"
            )
        return self._run_inference([query_image, target_image], prompt)

    @torch.no_grad()
    def batch_judge_with_reference(self, reference_image, description, target_images, progress_callback=None):
        """
        批量判别：使用参考图+描述，判断目标图片是否符合要求
        
        Args:
            reference_image: 参考图片 (PIL.Image)
            description: 文字描述
            target_images: 目标图片路径列表
            progress_callback: 进度回调函数
            
        Returns:
            List[Tuple[str, dict]]: [(图片路径, 判别结果), ...]
        """
        results = []
        
        for i, img_path in enumerate(target_images):
            if progress_callback:
                progress_callback(i + 1, len(target_images))
            
            try:
                target_image = Image.open(img_path).convert('RGB')
                result = self.judge_image_match(reference_image, target_image, description)
                results.append((img_path, result))
            except Exception as e:
                results.append((img_path, {"match": False, "error": str(e)}))
        
        return results
    def _run_inference(self, images, prompt):
        """执行推理并解析结果"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], 
            images=images, 
            return_tensors="pt"
        ).to(self.model.device)
        
        # 限制生成长度，使用 greedy decoding 保证输出稳定
        outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = self.processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return self._parse_judge_response(response)

    def _parse_judge_response(self, response):
        """解析 Qwen 的回答（增加正则匹配容错）"""
        result = {"match": False, "confidence": "中", "reason": "", "raw_response": response}
        
        # 1. 提取结论
        match = re.search(r"结论[：:]\s*(Yes|No|是|否)", response, re.IGNORECASE)
        if match:
            ans = match.group(1).lower()
            result["match"] = (ans in ["yes", "是"])
            
        # 2. 提取置信度
        match = re.search(r"置信度[：:]\s*(高|中|低)", response)
        if match:
            result["confidence"] = match.group(1)
            
        # 3. 提取理由
        match = re.search(r"理由[：:]\s*(.*)", response, re.DOTALL)
        if match:
            result["reason"] = match.group(1).strip().split('\n')[0] # 只取第一行
            
        return result