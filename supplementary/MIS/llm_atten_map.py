import os
import torch
import random
import numpy as np
import sys
import shutil
import json
import matplotlib.pyplot as plt
import seaborn as sns

GRAPH_TOKEN_INDEX = -200

# Get absolute path of current file
current_file = os.path.abspath(__file__)
# Get project root directory (Mol2Token directory)
project_root = os.path.dirname(os.path.dirname(current_file))
print(project_root)
# Add project root to Python path
sys.path.insert(0, project_root)

from torch.utils.data import DataLoader

import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from dataclasses import dataclass, field
from typing import Dict, Optional

from train.train import LazySupervisedGraphDataset, DataCollatorForSupervisedDataset
from utils.utils import tokenizer_graph_token, disable_torch_init, get_model_name_from_path
from model.builder import load_pretrained_model

# ------------------------------------------------------------
# constants & augmentations
# ------------------------------------------------------------
IGNORE_INDEX = -100
GRAPH_TOKEN_INDEX = -200
DEFAULT_GRAPH_TOKEN = "<graph>"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_CHUNK = 100
# ATTN_SAVE_DIR = "cls_clintox_viz" 
# os.makedirs("cls_clintox_viz", exist_ok=True)

ATTN_SAVE_DIR = "rg_atten_pad_dataset"

@dataclass
class DataArguments:
    model_path: Optional[str] = field(default=None, metadata={"help": "Path to the model"})
    model_base: Optional[str] = field(default=None, metadata={"help": "Base model path"})
    use_task: Optional[str] = field(default=None, metadata={"help": "model task"})
    use_dataset: Optional[str] = field(default=None, metadata={"help": "prossed dataset"})
    load_8bit: bool = False
    load_4bit: bool = False
    batch_size: int = field(default=4)
    temperature: int = field(default=0.0)
    cache_dir: Optional[str] = field(default=None, metadata={"help": "cache_dir"})


# Add this function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process attention maps")
    
    # Data arguments
    parser.add_argument("--model_path",type=str,default="./Projects/Mol2Token/checkpoints/qm9")
    parser.add_argument("--model_base", type=str, default="./Projects/Mol2Token/llm_ckpt")
    parser.add_argument("--use_task", type=str,default="cls")
    parser.add_argument("--use_dataset", type=str,default="clintox")      
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0) 
    parser.add_argument("--cache_dir", type=str,default="./cache")
  
    
    args = parser.parse_args()
    
    # Convert to data_args object
    data_args = DataArguments(
        model_path=args.model_path,
        model_base=args.model_base,
        use_task=args.use_task,
        use_dataset=args.use_dataset,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        batch_size=args.batch_size,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
    )
    
    return data_args

# Create dataset and collator for supervised fine-tuning
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    train_dataset = LazySupervisedGraphDataset(tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def get_dataloader(tokenizer):
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]

    dl = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        collate_fn=collator
    )
    return dl

def generate_avg_attn(model, dataloader, tokenizer, device):
    if isinstance(device, str):
        device = torch.device(device)
    with torch.no_grad():
        for batch in dataloader:
            # Expect batch: dict with input_ids, labels, attention_mask, graphs
            input_ids = batch["input_ids"].to(device)  # (B, T)
            attention_mask = batch.get("attention_mask")
            if attention_mask is None:
                attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
            else:
                attention_mask = attention_mask.to(device)
            smiles_list = batch["smiles"]

            # Use forward instead of generate
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                smiles=smiles_list,
                output_attentions=True,  # Enable attentions output
                return_dict=True  # Return dict output
            )

            # outputs.attentions: tuple length L, each (B, heads, T, T)
            # compute head mean per layer then stack -> (L, B, T, T)
            layer_mean_list = [outputs.attentions[i].mean(dim=1) for i in range(len(outputs.attentions))]
            stacked = torch.stack(layer_mean_list, dim=0)  # (L, B, T, T)
            all_attn = stacked.transpose(0, 1).to(torch.float16)  # (B, L, T, T)

            # mask out positions where input_ids == 0 (per your snippet)
            B, L, T, _ = all_attn.shape
            for i in range(B):
                cur_input_ids = input_ids[i]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
                # if no graph token, mask based on original input_ids
                if graph_token_indices.numel() == 0:
                    pad_mask = (cur_input_ids == 0)
                else:
                    # build extended pad_mask of length T
                    pad_mask = torch.zeros(T, dtype=torch.bool, device=device)
                    cur_pos = 0
                    start_idx = 0
                    while graph_token_indices.numel() > 0:
                        graph_token_start = graph_token_indices[0]
                        # front part: text tokens before graph, mask if ==0
                        front_ids = cur_input_ids[start_idx:graph_token_start]
                        front_len = len(front_ids)
                        pad_mask[cur_pos:cur_pos + front_len] = (front_ids == 0)
                        cur_pos += front_len

                        # graph part: motif tokens, same with fp_length in GraphConvNet
                        node_len = 32
                        cur_pos += node_len
                        # update for next graph token
                        start_idx = graph_token_start + 1
                        cur_input_ids = cur_input_ids[start_idx:]
                        graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
                    # back part: remaining text tokens after last graph, mask if ==0
                    back_ids = cur_input_ids
                    back_len = len(back_ids)
                    pad_mask[cur_pos:cur_pos + back_len] = (back_ids == 0)
                    # optional
                    pad_mask[cur_pos + back_len:] = True
                # apply mask: set attention to padding positions to 0
                all_attn[i, :, pad_mask] = 0
            yield all_attn.cpu()

# Save function (modified to save list of tensor)
def save_attentions_by_chunk(model, dataloader, tokenizer, save_dir=ATTN_SAVE_DIR, save_chunk=SAVE_CHUNK, device=DEVICE):
    os.makedirs(save_dir, exist_ok=True)
    buf = []
    batch_counter = 0
    chunk_counter = 0

    for all_attn in generate_avg_attn(model, dataloader, tokenizer, device):
        buf.append(all_attn)  # each all_attn is (B, L, T_i, T_i) on CPU
        batch_counter += 1

        if batch_counter % save_chunk == 0:
            chunk_counter += 1
            save_path = os.path.join(save_dir, f"atten_{chunk_counter}.pth")
            torch.save(buf, save_path)  # save as list of tensors (each is one batch's attn)
            print(f"[save] saved chunk {chunk_counter} -> {save_path}")
            buf.clear()

    if len(buf) > 0:
        save_path = os.path.join(save_dir, "atten_final.pth")
        torch.save(buf, save_path)
        print(f"[save] saved final partial chunk -> {save_path}")
        buf.clear()

# Load function
def iter_saved_attentions(saved_dir=ATTN_SAVE_DIR):
    files = [f for f in os.listdir(saved_dir) if f.endswith(".pth")]
    # sort by numeric suffix if present (atten_1, atten_2, ...), keep atten_final at the end
    def sort_key(x):
        if x == "atten_final.pth":
            return float("inf")
        nums = ''.join(ch for ch in x if ch.isdigit())
        return int(nums) if nums else 0
    files.sort(key=sort_key)
    for fname in files:
        path = os.path.join(saved_dir, fname)
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, list):
            for attn in loaded:  # each attn is (B, L, T_i, T_i)
                yield attn
        elif loaded.ndim == 4:
            yield loaded  # compatible with old single tensor saves, if any
        else:
            raise ValueError(f"Unsupported attention data in {path}")

# Get graph start attention index
def get_node_start_index(batch, sample_idx):
    """
    Determine (start_index, node_len)
    - start_index: position where GRAPH_TOKEN_INDEX occurs in the sample's input_ids (first occurrence)
    - node_len: fixed to 32
    """

    node_len = 32

    # get the sample's input_ids from batch
    input_ids = batch["input_ids"]  # tensor (B, T) or maybe list -> ensure tensor
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)

    # select sample
    cur_input_ids = input_ids[sample_idx]
    # find GRAPH_TOKEN_INDEX positions
    graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
    if graph_token_indices.numel() == 0:
        # fallback behavior: if not found, warn and assume start_index = 0
        print(f"[warning] GRAPH_TOKEN_INDEX not found in input_ids for sample {sample_idx}. Falling back to start_index=0")
        start_index = 0
    else:
        # take first occurrence as start (consistent with visualization logic)
        start_index = int(graph_token_indices[0].item())

    return start_index, node_len

# Randomly select 5 for visualization
def visualize_batch_random5(batch, attn_batch, out_dir=ATTN_SAVE_DIR, batch_id=None, test_data=None, sample_offset=0):
    os.makedirs(out_dir, exist_ok=True)
    smiles = batch.get("smiles", batch.get("smiles", None))
    if smiles is None:
        raise RuntimeError("batch must provide 'smiles'")

    B = attn_batch.shape[0]
    effective_B = min(B, len(smiles))

    indices = list(range(effective_B))
    random.shuffle(indices)
    chosen = indices[:min(5, effective_B)]

    for sample_idx in chosen:
        global_idx = sample_offset + sample_idx
        if test_data is None or global_idx >= len(test_data):
            print(f"[warning] No test_data or global_idx {global_idx} out of range. Skipping SMILES for sample {sample_idx}.")
            smiles = "Unknown"
        else:
            smiles = test_data[global_idx]["smiles"]

        start_idx, node_len = get_node_start_index(batch, sample_idx)
        if node_len <= 0:
            continue

        # bounds check: ensure start_idx+node_len <= attn sequence dimension
        L = attn_batch.shape[1]
        T = attn_batch.shape[2]
        if start_idx < 0 or start_idx + node_len > T:
            print(f"[warning] sample {sample_idx}: computed node span [{start_idx}, {start_idx+node_len}) is outside attn T={T}. Skipping.")
            continue

        # get sample attn: (L, T, T)
        attn_sample = attn_batch[sample_idx]
        node_indices = list(range(start_idx, start_idx + node_len))

        # slice node->node: (L, M, M)
        node_node = attn_sample[:, node_indices][:, :, node_indices]  # (L, M, M)
        avg = node_node.mean(dim=0).cpu().numpy()  # (M, M)
        M = node_len

        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(avg, cmap="viridis", cbar=True)
        title = f"Average attention of motif tokens"
        plt.title(title)

        plt.xlabel("motif index (local)")
        plt.ylabel("motif index (local)")
        plt.tight_layout()

        fname = f"attn_batch{batch_id}_sample{sample_idx}_smiles{smiles}.svg"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        plt.close()
        print(f"[visualize] saved {out_path}")

# Visualize specific index molecules
def visualize_batch_specific(batch, attn_batch, out_dir=ATTN_SAVE_DIR, batch_id=None, test_data=None, sample_offset=0):
    os.makedirs(out_dir, exist_ok=True)
    smiles_list = batch.get("smiles", None)
    if smiles_list is None:
        raise RuntimeError("batch must provide 'smiles'")

    B = attn_batch.shape[0]
    effective_B = min(B, len(smiles_list))

    # Fixed global index to visualize (for specific molecules, repeated twice)
    targets = [925, 926]  # index starting from 0, corresponding to lines 277 and 278

    for sample_idx in range(effective_B):
        global_idx = sample_offset + sample_idx
        if global_idx not in targets:
            continue  # Only visualize specific indexes

        if test_data is None or global_idx >= len(test_data):
            print(f"[warning] No test_data or global_idx {global_idx} out of range. Skipping SMILES for sample {sample_idx}.")
            cur_smiles = "Unknown"
        else:
            cur_smiles = test_data[global_idx]["smiles"]
            print(f"Now the smiles is {cur_smiles}")

        start_idx, node_len = get_node_start_index(batch, sample_idx)
        if node_len <= 0:
            continue

        # bounds check: ensure start_idx+node_len <= attn sequence dimension
        L = attn_batch.shape[1]
        T = attn_batch.shape[2]
        if start_idx < 0 or start_idx + node_len > T:
            print(f"[warning] sample {sample_idx} (global {global_idx}): computed node span [{start_idx}, {start_idx+node_len}) is outside attn T={T}. Skipping.")
            continue

        # get sample attn: (L, T, T)
        attn_sample = attn_batch[sample_idx]
        node_indices = list(range(start_idx, start_idx + node_len))

        # slice node->node: (L, M, M)
        node_node = attn_sample[:, node_indices][:, :, node_indices]  # (L, M, M)
        avg = node_node.mean(dim=0).cpu().numpy()  # (M, M)
        M = node_len

        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(avg, cmap="viridis", cbar=True)
        title = f"Average attention of motif tokens"
        plt.title(title)

        plt.xlabel("motif index (local)")
        plt.ylabel("motif index (local)")
        plt.tight_layout()

        # Filename adds global_idx and smiles fragment to distinguish
        fname = f"attn_batch{batch_id}_sample{sample_idx}_global{global_idx}_smiles{cur_smiles[:20]}.svg"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        plt.close()
        print(f"[visualize] saved {out_path}")

# Mark highest attention scores
def visualize_batch_specific_more(batch, attn_batch, out_dir=ATTN_SAVE_DIR, batch_id=None, test_data=None, sample_offset=0):
    os.makedirs(out_dir, exist_ok=True)
    smiles_list = batch.get("smiles", None)
    if smiles_list is None:
        raise RuntimeError("batch must provide 'smiles'")

    B = attn_batch.shape[0]
    effective_B = min(B, len(smiles_list))

    # Fixed global index to visualize (for specific molecules, repeated twice)
    targets = [925, 926]  # index starting from 0, corresponding to lines 277 and 278

    for sample_idx in range(effective_B):
        global_idx = sample_offset + sample_idx
        if global_idx not in targets:
            continue  # Only visualize specific indexes

        if test_data is None or global_idx >= len(test_data):
            print(f"[warning] No test_data or global_idx {global_idx} out of range. Skipping SMILES for sample {sample_idx}.")
            cur_smiles = "Unknown"
        else:
            cur_smiles = test_data[global_idx]["smiles"]

        start_idx, node_len = get_node_start_index(batch, sample_idx)
        if node_len <= 0 or node_len != 32:  # Additional check to ensure it's 32
            print(f"[warning] node_len {node_len} not 32, skipping annotation for global {global_idx}")
            continue

        # bounds check
        L = attn_batch.shape[1]
        T = attn_batch.shape[2]
        if start_idx < 0 or start_idx + node_len > T:
            print(f"[warning] sample {sample_idx} (global {global_idx}): node span out of range. Skipping.")
            continue

        # get sample attn: (L, T, T)
        attn_sample = attn_batch[sample_idx]
        node_indices = list(range(start_idx, start_idx + node_len))

        # slice node->node: (L, M, M)
        node_node = attn_sample[:, node_indices][:, :, node_indices]  # (L, M, M)
        avg = node_node.mean(dim=0).cpu().numpy()  # (M, M)
        M = node_len  # 32

        plt.figure(figsize=(9, 8))  # Slightly larger, suitable for annotations
        ax = sns.heatmap(avg, cmap="viridis", cbar=True, annot=False, fmt=".2f", linewidths=.5)

        # === New: Find and annotate top3 off-diagonal pairs ===
        off_diag = avg.copy()
        np.fill_diagonal(off_diag, -np.inf)  # Exclude diagonal

        # Find top 6 highest (considering symmetry, may repeat), then deduplicate to unique pairs
        flat_indices = np.argsort(off_diag.flatten())[::-1][:6]  # Take more to prevent less after deduplication
        seen_pairs = set()
        top_pairs = []  # Store (i, j, score), i < j

        for flat_idx in flat_indices:
            score = off_diag.flatten()[flat_idx]
            if score == -np.inf:
                continue
            row, col = divmod(flat_idx, M)
            i, j = min(row, col), max(row, col)
            print(f"row{i} and col{j} has score {score:.4f} correlation...")

            pair = (i, j)
            if pair not in seen_pairs and i != j:
                seen_pairs.add(pair)
                top_pairs.append((row, col, score))  # Original order, for annotation
                top_pairs.append((col, row, score))  # Symmetric position also annotated
                if len(seen_pairs) == 3:
                    break

        title = f"Average attention of motif tokens"
        plt.title(title)

        plt.xlabel("motif index (local)")
        plt.ylabel("motif index (local)")
        plt.tight_layout()

        # Filename adds global_idx
        fname = f"attn_batch{batch_id}_sample{sample_idx}_global{global_idx}_top3_annotated_smiles{cur_smiles[:20]}.svg"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        plt.close()
        print(f"[visualize] saved annotated heatmap: {out_path}")

# Enhanced visualization with indices
def visualize_batch_specific_with_indices(batch, attn_batch, out_dir=ATTN_SAVE_DIR, batch_id=None, test_data=None, sample_offset=0):
    os.makedirs(out_dir, exist_ok=True)
    smiles_list = batch.get("smiles", None)
    if smiles_list is None:
        raise RuntimeError("batch must provide 'smiles'")

    B = attn_batch.shape[0]
    effective_B = min(B, len(smiles_list))

    # Fixed global index to visualize (for specific molecules, repeated twice)
    targets = [925, 926]  # index starting from 0, corresponding to lines 277 and 278

    for sample_idx in range(effective_B):
        global_idx = sample_offset + sample_idx
        if global_idx not in targets:
            continue  # Only visualize specific indexes

        if test_data is None or global_idx >= len(test_data):
            print(f"[warning] No test_data or global_idx {global_idx} out of range. Skipping SMILES for sample {sample_idx}.")
            cur_smiles = "Unknown"
        else:
            cur_smiles = test_data[global_idx]["smiles"]

        start_idx, node_len = get_node_start_index(batch, sample_idx)
        if node_len <= 0:
            continue

        # bounds check: ensure start_idx+node_len <= attn sequence dimension
        L = attn_batch.shape[1]
        T = attn_batch.shape[2]
        if start_idx < 0 or start_idx + node_len > T:
            print(f"[warning] sample {sample_idx} (global {global_idx}): computed node span [{start_idx}, {start_idx+node_len}) is outside attn T={T}. Skipping.")
            continue

        # get sample attn: (L, T, T)
        attn_sample = attn_batch[sample_idx]
        node_indices = list(range(start_idx, start_idx + node_len))

        # slice node->node: (L, M, M)
        node_node = attn_sample[:, node_indices][:, :, node_indices]  # (L, M, M)
        avg = node_node.mean(dim=0).cpu().numpy()  # (M, M)
        M = node_len

        # Create heatmap, showing all motif indices but not grid lines
        plt.figure(figsize=(10, 8))  # Slightly larger to show indices
        
        # Use sns.heatmap to draw heatmap, set annot=True to show values, fmt='.2f' to keep two decimal places
        ax = sns.heatmap(
            avg, 
            cmap="viridis", 
            cbar=True,
            square=True,  # Square cells
            xticklabels=True,  # Show x-axis labels (motif index)
            yticklabels=True,  # Show y-axis labels (motif index)
            linewidths=0       # Don't show grid lines
        )
        
        title = f"Average attention of motif tokens"
        plt.title(title)

        plt.xlabel("motif index (local)")
        plt.ylabel("motif index (local)")
        plt.tight_layout()

        # Filename adds global_idx and smiles fragment to distinguish
        fname = f"attn_batch{batch_id}_sample{sample_idx}_global{global_idx}_with_indices_smiles{cur_smiles[:20]}.svg"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        plt.close()
        print(f"[visualize] saved {out_path}")

def main(save_only=False, visualize_only=False, max_save_batches=None, max_visual_chunks=None):
    global data_args, batch_counter, attention_cache
    data_args = parse_args()

    # device
    device = select_device(0) 
    
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _ = load_pretrained_model(
        model_path, data_args.model_base, model_name,
        load_8bit=data_args.load_8bit, load_4bit=data_args.load_4bit,
        cache_dir=data_args.cache_dir, mm_encoder_cfg=None, device_map="auto"
    )

    model = model.float().to(device)
    # model = model.to(torch.bfloat16)
    # Sampling 
    batch_size = data_args.batch_size
    
    dataloader = get_dataloader(tokenizer)
    print("DataLoader ready.")

    # optional limited dataloader for saving
    if max_save_batches is not None:
        import itertools
        dataloader_for_save = (b for i, b in enumerate(dataloader) if i < max_save_batches)
    else:
        dataloader_for_save = dataloader

    # load smiles *_test.jsonl file
    print("Loading test data from *_test.jsonl...")
    # with open("./dataset/clintox/clintox_test.jsonl", "r") as f:
    with open("./dataset/qm9/qm9_test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
    print(f"Loaded {len(test_data)} test samples.")

    if not visualize_only:
        print("STEP 1: Saving attentions by chunk ...") 
        save_attentions_by_chunk(model, dataloader_for_save, tokenizer, save_dir=ATTN_SAVE_DIR, save_chunk=SAVE_CHUNK, device=DEVICE)
        print("STEP 1 done.")

    if not save_only:
        print("STEP 2: Visualizing saved attentions (random 5 per batch) ...")
        saved_iter = iter_saved_attentions(ATTN_SAVE_DIR)
        # create a fresh data iterator to ensure same order
        data_iter = iter(get_dataloader(tokenizer))

        chunk_count = 0
        sample_counter = 0  # Global sample offset tracker
        for attn_batch in saved_iter:
            chunk_count += 1
            if max_visual_chunks is not None and chunk_count > max_visual_chunks:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                print("[main] data loader exhausted before attention files; stopping.")
                break

            # attn_batch is (B, L, T, T)
            B = attn_batch.shape[0]
            # visualize_batch_random5(batch, attn_batch, out_dir=ATTN_SAVE_DIR, batch_id=chunk_count, test_data=test_data, sample_offset=sample_counter)
            # visualize_batch_specific(batch, attn_batch, out_dir=ATTN_SAVE_DIR, batch_id=chunk_count, test_data=test_data, sample_offset=sample_counter)
            visualize_batch_specific_more(batch, attn_batch, out_dir=ATTN_SAVE_DIR, batch_id=chunk_count, test_data=test_data, sample_offset=sample_counter)
            sample_counter += B  # Update global offset for next batch
        print("STEP 2 done.")

def select_device(preferred_device_id=1):
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU")
        return torch.device("cpu")
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check if preferred device is available
    if preferred_device_id < gpu_count:
        selected_device = torch.device(f"cuda:{preferred_device_id}")
        print(f"Using preferred device: {selected_device}")
    else:
        # Fall back to first available GPU
        selected_device = torch.device("cuda:0")
        print(f"Preferred device cuda:{preferred_device_id} not available, falling back to: {selected_device}")
    
    return selected_device

if __name__ == "__main__":
    main(save_only=False, visualize_only=True, max_save_batches=None, max_visual_chunks=None)