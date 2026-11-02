# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
sys.path.append(".")
sys.path.append("./utils")
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Begain to debug...")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

replace_llama_attn_with_flash_attn()

# from train import _train
from train import _train

if __name__ == "__main__":
    # _train()
    _train()
