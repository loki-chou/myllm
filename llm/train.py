"""
Train a model
"""
import os
import torch
import tiktoken
from aim import Run
from model import Model

# 训练模型类，定义训练各种参数

# Hyperparameters
batch_size = 4  # How many batches per training step
context_length = 128  # Length of the token chunk each batch
max_iters = 30000  # Total of training iterations <- Change this to smaller number for testing
learning_rate = 1e-3  # 0.001
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # 使用mac gpu.
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)
model_name = "myllm2"

# AIM Logs
run = Run()
run["hparams"] = {
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "batch_size": batch_size,
    "context_length": context_length,
    "model_name": model_name
}

# 准备训练数据
folder_path = '/Users/loki/Downloads/文本素材/'
text = ''
# 遍历文件夹
for filename in os.listdir(folder_path):
    # 检查文件是否为.txt文件
    if filename.endswith('.txt'):
        # 构建完整文件路径
        file_path = os.path.join(folder_path, filename)
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            text = text + file.read()

# Using TikToken (Same as GPT3) to tokenize the source text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
# max_token_value = max(tokenized_text)+1  # the maximum value of the tokenized numbers
tokenized_text = torch.tensor(tokenized_text)  # 将77,919个tokens 转换到Pytorch张量中

total_tokens = encoding.encode_ordinary(text)
print(f"数据集合计有 {len(total_tokens):,} tokens")

# 分割训练接
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# 初始化model
model = Model().to(device)


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(0, len(data) - context_length, (batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))
        run.track(round(losses['train'].item(), 3), 'Training Loss')
        run.track(round(losses['valid'].item(), 3), 'Validation Loss')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), 'model/' + model_name + '.pt')
print('训练结束.模型参数量为：', sum(p.numel() for p in model.parameters()))

#%%

#%%
