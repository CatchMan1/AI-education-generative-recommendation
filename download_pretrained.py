"""
一次性下载 t5-small 权重到本地 pretrained/t5-small 目录。
运行后，三个模型将从本地加载，无需访问 HuggingFace。
"""
from transformers import T5EncoderModel, T5ForConditionalGeneration

save_dir = "./pretrained/t5-small"
print(f"Downloading t5-small to {save_dir} ...")

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.save_pretrained(save_dir)

print("Done. Weights saved to:", save_dir)
