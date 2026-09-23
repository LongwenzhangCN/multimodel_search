import torch
from transformers import AutoProcessor
from PIL import Image
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QWEN_MODELS, DEFAULT_QWEN_MODEL, DEVICE

class QwenJudgeEncoder:
    def __init__(self, model_key=None):
        self.device = DEVICE
        self.model_key = model_key or DEFAULT_QWEN_MODEL
        
        if self.model_key not in QWEN_MODELS:
            raise ValueError(f"未知的模型密钥: {self.model_key}. 可用的模型: {list(QWEN_MODELS.keys())}")
        
        model_config = QWEN_MODELS[self.model_key]
        model_path = model_config["path"]
        model_class_name = model_config["model_class"]
        
        print(f"🚀 正在加载 {model_config['display_name']}: {model_path}")
        
        # 动态导入模型类
        if model_class_name == "Qwen2VLForConditionalGeneration":
            from transformers import Qwen2VLForConditionalGeneration
            ModelClass = Qwen2VLForConditionalGeneration
        elif model_class_name == "Qwen3VLForConditionalGeneration":
            from transformers import Qwen3VLForConditionalGeneration
            ModelClass = Qwen3VLForConditionalGeneration
        else:
            raise ValueError(f"不支持的模型类: {model_class_name}")
        
        # 加载处理器和模型
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # 根据模型大小选择加载策略
        if "8b" in self.model_key.lower():
            # 8B 模型使用 device_map="auto" 和 float16
            self.model = ModelClass.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # 7B 和 4B 模型直接加载
            self.model = ModelClass.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"✅ {model_config['display_name']} 加载完成！")

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
            result["reason"] = match.group(1).strip().split('\n')[0]
            
        return result