import torch
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor
from PIL import Image
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        print("🔧 模型管理器初始化完成 (纯本地加载模式)")
    
    def _check_path(self, model_key, model_config):
        """严格校验本地路径"""
        path = model_config["local_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f" 模型路径不存在: {path}\n请手动下载模型到该目录！")
        if not os.path.exists(os.path.join(path, "config.json")):
            raise FileNotFoundError(f"❌ 模型目录不完整，缺少 config.json: {path}")

    def load_model(self, model_key, model_config):
        """加载指定模型"""
        if model_key in self.loaded_models:
            print(f"✅ 模型 {model_config['display_name']} 已在显存中")
            return self.loaded_models[model_key]
        
        # 1. 校验路径
        self._check_path(model_key, model_config)
        print(f" 正在从本地加载 {model_config['display_name']}...")
        
        try:
            if model_config["type"] == "clip":
                processor = CLIPProcessor.from_pretrained(model_config["local_path"])
                model = CLIPModel.from_pretrained(model_config["local_path"]).to(DEVICE)
                model.eval()
                encoder = CLIPEncoder(model, processor)
            
            elif model_config["type"] == "dinov3":
                if model_config["backbone"] == "vit":
                    processor = AutoProcessor.from_pretrained(model_config["local_path"])
                    model = AutoModel.from_pretrained(model_config["local_path"]).to(DEVICE)
                    model.eval()
                    encoder = DINOv3ViTEncoder(model, processor)
                else:  # convnext
                    processor = None 
                    model = AutoModel.from_pretrained(model_config["local_path"], num_labels=0).to(DEVICE)
                    model.eval()
                    encoder = DINOv3ConvNeXtEncoder(model)
            
            # 保存到缓存字典
            self.loaded_models[model_key] = encoder
            print(f"✅ {model_config['display_name']} 本地加载完成！")
            return encoder
        
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return None

    def get_encoder(self, model_key):
        """获取已加载的编码器 (修复报错的关键方法)"""
        return self.loaded_models.get(model_key)
    
    def unload_model(self, model_key):
        """卸载模型以释放显存"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            torch.cuda.empty_cache()
            print(f"️ 模型 {model_key} 已卸载，显存已释放")


# ========== 编码器实现 ==========

class CLIPEncoder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    @torch.no_grad()
    def encode_image(self, image_path):
        inputs = self.processor(images=Image.open(image_path).convert('RGB'), return_tensors="pt", padding=True).to(DEVICE)
        feat = self.model.get_image_features(**inputs)
        return (feat / feat.norm(dim=-1, keepdim=True))[0].cpu().numpy()
    
    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(DEVICE)
        feat = self.model.get_text_features(**inputs)
        return (feat / feat.norm(dim=-1, keepdim=True))[0].cpu().numpy()


class DINOv3ViTEncoder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    @torch.no_grad()
    def encode_image(self, image_path):
        inputs = self.processor(images=Image.open(image_path).convert('RGB'), return_tensors="pt").to(DEVICE)
        feat = self.model(**inputs).last_hidden_state[:, 0, :]
        return (feat / feat.norm(dim=-1, keepdim=True))[0].cpu().numpy()


class DINOv3ConvNeXtEncoder:
    def __init__(self, model):
        self.model = model
        from torchvision import transforms
        self.transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def encode_image(self, image_path):
        inputs = self.transforms(Image.open(image_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        feat = self.model(inputs).last_hidden_state.mean(dim=1)
        return (feat / feat.norm(dim=-1, keepdim=True))[0].cpu().numpy()