from src.Utils.layers import TransformerLM
from src.Utils.Training_loop_utils import load_checkpoint
from src.configs import CS_336_CONFIG, GPT_2_LARGE_CONFIG, GPT_2_SMALL_CONFIG
from src.BPE_Tokenizer.BPE import BPE
from src.Utils.utils import AdamW
import torch
checkpoint_path = "output/checkpoint/GPT_2_LARGE_checkpoint.pt"
vocab_path = "output/vocab_tiny_train.pkl"
merges_path = "output/merges_tiny_train.pkl"
print("正在加载BPE tokenizer...")
tokenizer = BPE.from_files(vocab_path, merges_path)
special_tokens = [tokenizer.encode(i)[0] for i in tokenizer.special_tokens]
print("BPE tokenizer加载完成")
# 创建模型
print("正在创建模型...")
config = GPT_2_LARGE_CONFIG
device = torch.device("cuda:1")
model = TransformerLM(
    config["vocab_size"],
    config["d_model"],
    config["num_layers"],
    config["num_heads"],
    config["d_ff"],
    config["rope_theta"],
    config["max_seq_len"],
    device=device
)

# 创建优化器（仅用于加载checkpoint，不会用于生成）
print("正在创建优化器...")
optimizer = AdamW(model.parameters(), lr=config["max_learning_rate"])

# 加载模型状态
print("正在加载模型状态...")
loaded_iteration = load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
print(f"成功加载checkpoint，迭代次数: {loaded_iteration}")

# 将模型设置为评估模式
model.eval()

# 从提示词生成文本

MAX_SEQ_LEN = 2048
while True:
    prompt = input("请输入提示词: ")
    tokens = tokenizer.encode(prompt)
    print(f"提示词编码为: {tokens}")
    tokens = torch.tensor(tokens, device=device)
    while True:
        next_token = model.generate(tokens, temperature=0.6, top_p=0.9)
        # print(f"下一个token为: {next_token.item()}, 对应文本为: {tokenizer.decode([next_token.item()])}")
        if next_token in special_tokens:
            break
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
        if tokens.shape[-1] > MAX_SEQ_LEN:
            break
    print(f"文本为: {tokenizer.decode(tokens.tolist())}")