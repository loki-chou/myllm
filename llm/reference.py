"""
Sample from a trained model
"""
import os
import tiktoken
import torch

from model import Model


# 设置随机种子以确保可复现性
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 如果有其他随机过程，也应该在这里设置种子


# 检查文件是否存在
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 '{file_path}' 不存在。")


# 安全加载模型
def safe_load_model(file_path, device):
    check_file_exists(file_path)
    # 使用 torch.load 的安全用法，并指定适当的 pickle 加载参数
    # 这里假设你不需要特别的pickle_load_args，根据实际情况调整
    model_dict = torch.load(file_path, map_location=device)
    model = Model()
    model.load_state_dict(model_dict)
    return model


# Hyperparameters
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # 使用mac gpu.
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if it's available.
TORCH_SEED = 1337
set_random_seed(TORCH_SEED)

encoding = tiktoken.get_encoding("cl100k_base")

# 从训练好的模型中初始化
model = safe_load_model('model/model-scifi-finetune.pt', device)
model.eval()
model.to(device)

# 定义开始的字符串
start = '帮我写一本小说，主角是黄强'
start_ids = encoding.encode(start)
# 修正了变量名的拼写错误
start_tensor = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# 运行生成
with torch.no_grad():
    y = model.generate(start_tensor)  # 假设generate函数接受start_tensor作为输入
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')

#%%

#%%
