
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import random
import os
import torch
os.environ["OPENAI_API_KEY"] = "sk-ACmwjXam65oCkcAQ66AfEcD894984f91BbEbE162765d502e"
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"

def generate_item_embedding(item_text_dic, tokenizer, model, word_drop_ratio=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order_texts = [[0]] * (len(item_text_dic))   # 列表，列表中的第i个元素代表ID为i的项目对应的文本。创建一个项目长度的列表

    # for item in item_text_dic:
    #     order_texts[item] = item_text_dic[item]# 变成嵌套列表的形式[[1,2,3],[4,5,6]],数字部分为内容

    # 生成embedding时
    max_item_id = max(item_text_dic.keys())
    order_texts = ["" if k == 0 else item_text_dic.get(k, "") for k in range(max_item_id + 1)]
    # for text in order_texts:
    #    assert text != [0]

    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]# 一次处理batch_size的内容
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(device)
        outputs = model(**encoded_sentences) #编译后的字段放入模型中进行推理
        # 计算平均池化嵌入
        masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1) # [batch_size, seq_len, 1]
        mean_output = masked_output[:,1:,:].sum(dim=1) / encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
        mean_output = mean_output.detach()
        embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    # 猜测embeddings的大小为：项目总数 * 嵌入大小

    print('Embeddings shape: ', embeddings.shape)
    return embeddings

def load_plm(model_name='bert-base-uncased'):
    """
    加载预训练语言模型，支持Windows环境
    在Windows环境下会自动降级到不使用量化配置
    """
    import platform
    import sys
    
    # 检查是否为Windows系统或者bitsandbytes不可用
    is_windows = platform.system() == "Windows"
    
    try:
        # 尝试导入和使用bitsandbytes（主要用于Linux/CUDA环境）
        if not is_windows:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("✅ 使用量化配置加载模型 (Linux/CUDA优化)")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config)
        else:
            raise ImportError("Windows环境，跳过量化配置")
            
    except (ImportError, Exception) as e:
        # 降级方案：不使用量化配置（适用于Windows或bitsandbytes不可用的情况）
        print(f"⚠️ 量化配置不可用 ({e.__class__.__name__})，使用标准配置")
        print("✅ 使用标准配置加载模型 (Windows兼容)")
        
        # 标准加载方式
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # 如果有GPU可用，将模型移到GPU
        if torch.cuda.is_available():
            print("检测到CUDA，将模型移至GPU")
            model = model.cuda()
        else:
            print("使用CPU运行模型")
    
    return tokenizer, model