#!/usr/bin/env python3
"""
Evaluate model on QM9 regression task.
Computes MAE (Mean Absolute Error) for HOMO, LUMO, and gap predictions.
"""

import os
import sys
import json
import argparse
import re
from tqdm import tqdm

import torch
import numpy as np

sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./model")

from model.builder import load_pretrained_model
from utils.utils import tokenizer_graph_token, disable_torch_init, get_model_name_from_path
from utils.conversation import conv_templates, SeparatorStyle


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_prompt_from_conversation(conversations, conv_mode="v1"):
    conv = conv_templates[conv_mode].copy()
    for s in conversations:
        if s["from"] == "human":
            value = s["value"]
            if isinstance(value, list):
                value = value[0] if value else ""
            conv.append_message(conv.roles[0], value)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def extract_number(text):
    """Extract first number from text."""
    # Handle cases like ".278900" -> "0.278900"
    text = str(text) # fix: float can't be strip
    text = text.strip()
    # Try to find a float pattern including leading decimal point
    match = re.search(r'-?\.?\d+\.?\d*', text)
    if match:
        try:
            num_str = match.group()
            # Fix leading decimal point without zero (e.g., ".278" -> "0.278")
            if num_str.startswith('.'):
                num_str = '0' + num_str
            elif num_str.startswith('-.'):
                num_str = '-0' + num_str[1:]
            return float(num_str)
        except:
            return None
    return None


def batch_iterable(iterable, batch_size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _ = load_pretrained_model(
        model_path, args.model_base, model_name,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
        cache_dir=args.cache_dir, mm_encoder_cfg=None, device_map="auto"
    )

    model = model.float().to(device)
    model.eval()

    test_examples = list(read_jsonl(os.path.expanduser(args.test_file)))
    if args.max_samples and args.max_samples > 0:
        test_examples = test_examples[:args.max_samples]
    print(f"Loaded {len(test_examples)} test examples from {args.test_file}")

    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)) or ".", exist_ok=True)
    answers_path = os.path.expanduser(args.answers_file)
    outfile = open(answers_path, "w", buffering=1, encoding="utf-8")

    # Group results by target
    results = {"lipo": []}

    for batch in tqdm(list(batch_iterable(test_examples, args.batch_size)), desc="Batches"):
        prompts = []
        smiles_list = []
        gts = []
        targets = []

        for sample in batch:
            if "conversations" in sample:
                conv = sample["conversations"]
                prompt_str, conv_obj = make_prompt_from_conversation(conv, conv_mode=args.conv_mode)
            else:
                raise ValueError("Expected 'conversations' in sample")

            prompts.append(prompt_str)
            smiles = sample.get("smiles", None)
            smiles_list.append(smiles)

            # Ground truth is in conversations[1]['value']
            gt = sample["conversations"][1].get("value", "")
            gts.append(gt)
            targets.append(sample.get("target", "unknown"))

        # Tokenize
        input_ids_list = []
        for p in prompts:
            try:
                toks = tokenizer_graph_token(p, tokenizer, return_tensors="pt")
                if isinstance(toks, torch.Tensor):
                    input_ids_list.append(toks.to(device))
                else:
                    input_ids_list.append(toks["input_ids"].squeeze(0).to(device))
            except:
                tok = tokenizer(p, return_tensors="pt", padding=False)
                input_ids_list.append(tok["input_ids"].squeeze(0).to(device))

        # Left padding for decoder-only models
        max_len = max(ids.shape[0] for ids in input_ids_list)
        input_ids_batch = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for i, ids in enumerate(input_ids_list):
            # Left pad: place tokens at the end
            input_ids_batch[i, max_len - ids.shape[0]:] = ids

        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids_batch,
                smiles=smiles_list,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=64,
                use_cache=True
            )

        # Decode outputs - only decode the generated part (skip input tokens)
        for i, (output, gt, target) in enumerate(zip(outputs, gts, targets)):
            # With left padding, generated tokens start after max_len (padded input length)
            generated_tokens = output[max_len:]
            # Filter out invalid token IDs that are out of vocabulary range
            vocab_size = len(tokenizer)
            valid_output = [t for t in generated_tokens.tolist() if 0 <= t < vocab_size]
            try:
                pred_text = tokenizer.decode(valid_output, skip_special_tokens=True)
            except Exception as e:
                pred_text = ""
            pred_text = pred_text.strip()

            pred_val = extract_number(pred_text)
            gt_val = extract_number(gt)


            # Filter out unreasonable predictions (QM9 values typically in -0.5 to 0.5 range)
            if pred_val is not None and pred_val < 5.0 and pred_val > 0:
                pred_val = None

            rec = {
                "smiles": smiles_list[i],
                "target": target,
                "gt": gt,
                "pred": pred_text,
                "gt_val": gt_val,
                "pred_val": pred_val
            }
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if pred_val is not None and gt_val is not None and target in results:
                squared_error = (pred_val - gt_val) ** 2
                results[target].append(squared_error)

    outfile.close()

    # Compute RMSE for each target
    print("=" * 50)
    print("Lipo Regression Results (RMSE)")
    print("=" * 50)

    for target in ["lipo"]:
        squared_errors = results[target]
        if squared_errors:
            rmse = np.sqrt(np.mean(squared_errors))
            print(f"{target.upper()}: RMSE = {rmse:.6f} (n={len(squared_errors)})")
        else:
            print(f"{target.upper()}: No valid predictions")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="pc_answers.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    main(args)
