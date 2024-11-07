import torch
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import deepspeed
import torch.distributed as dist
import time

# 初始化分布式环境
deepspeed.init_distributed()
world_size = dist.get_world_size()
rank = dist.get_rank()

# 加载预训练的tokenizer和模型
tokenizer = BertTokenizer.from_pretrained("./tokenizer/")
model = BertForMaskedLM.from_pretrained("./model")

# 加载WikiText数据集
dataset = load_dataset("parquet", data_files="./data/train-00000-of-00001.parquet")

# 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_dataset = tokenized_dataset["train"]

# 使用分布式采样器
sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

# 设置数据加载器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, sampler=sampler)  

# 设置训练参数
epochs = 1

# 确保模型参数的连续性
for param in model.parameters():
    param.data = param.data.contiguous()

# 初始化deepspeed模型，将model转换成model_engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=None,
    model=model,
    model_parameters=model.parameters(),
    config='./ds_config.json'
)

model_engine.train()

start_time = time.time()

if rank == 0: 
    print("Training start...")

# 手动实现训练循环
for epoch in range(epochs):
    epoch_loss = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank != 0), mininterval=20)):  # 设置只有0号进程打印进度条且打印间隔为20s
        # 将数据移到GPU
        batch = {k: v.to(rank) for k, v in batch.items()}

        # 前向传播
        outputs = model_engine(**batch)
        loss = outputs.loss

        # 反向传播
        model_engine.backward(loss)
        
        # 更新参数
        model_engine.step()

        # 记录损失
        epoch_loss += loss.item()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nRank {rank}, Epoch {epoch+1}/{epochs}, Batch {step+1}/{len(train_loader)}, Loss: {epoch_loss/len(train_loader)}, total_time_used: {elapsed_time / 60:.2f} mins")

if rank == 0: 
    print("Training complete.")
