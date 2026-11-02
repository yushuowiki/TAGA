#!/usr/bin/env python3
# eval_pretrain.py
"""
Evaluate pretrain model on SMILES -> molecular property classification (instruct-test jsonl).
Saves per-sample outputs to answers_file (jsonl), and prints ROC-AUC at the end.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score


def set_seed_deterministic(seed=42):
    """set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# add repo root if needed
sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./model")

# Try to import project utilities -- keep same names as in your repo
try:
    from model.builder import load_pretrained_model
except Exception as e:
    raise ImportError(f"Cannot import load_pretrained_model from model.builder: {e}")

try:
    from utils.utils import tokenizer_graph_token, disable_torch_init, get_model_name_from_path
    from utils.conversation import conv_templates, SeparatorStyle
except Exception as e:
    # fallback message but still allow script to run if user customizes imports
    raise ImportError(f"Cannot import utils modules. Make sure your repo has utils.utils and utils.conversation. Error: {e}")

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9527))
#     print("Begain to eval debug...")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# Helper to convert textual label to int (same as eval_hight.py)
def convert_label_to_int(label):
    label = str(label).strip()
    if label.lower() in ['active', "yes", "true", "1", "positive"]:
        return 1
    elif label.lower() in ['inactive', "no", "false", "0", "negative"]:
        return 0
    else:
        # try numeric cast
        try:
            v = float(label)
            return 1 if v > 0.5 else 0
        except:
            # unknown -> default 0
            # print("Unknown label:", label)
            return 0

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def make_prompt_from_conversation(conversations, conv_mode="v1"):
    """
    conversations: list of {"from": "human"/"gpt", "value": str or list}
    Use conv_templates[conv_mode] to build prompt (same style as your preprocess).
    For inference, only include human messages; exclude assistant (gpt) answers.
    """
    conv = conv_templates[conv_mode].copy()
    # Only add human messages for inference prompt
    for i, s in enumerate(conversations):
        if s["from"] == "human":
            value = s["value"]
            # Handle case where value is a list (take first element)
            if isinstance(value, list):
                value = value[0] if value else ""
            conv.append_message(conv.roles[0], value)
    # Add empty assistant message for generation
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv

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
    tokenizer, model, _= load_pretrained_model(
        model_path, args.model_base, model_name,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
        cache_dir=args.cache_dir, mm_encoder_cfg=None, device_map="auto"
    )

    # move model to device and dtype
    # load graph tower ckpt if provided (some multimodal models require it)
    if args.graph_checkpoint_path:
        # try:
        #     print(f"Graph tower has already been initialized:{model.get_graph_tower()}")
        #     model.get_graph_tower()._load_state_dict(args.graph_checkpoint_path, strict=False)
        #     model.get_graph_tower().requires_grad_(False)
        #     print(f"Loaded graph checkpoint from {args.graph_checkpoint_path}")
        # except Exception as e:
        #     print(f"Warning: failed to load graph checkpoint: {e}")
        gt = model.get_graph_tower()

        # load file to CPU first
        raw = torch.load(args.graph_checkpoint_path, map_location='cpu')
        # raw may be:
        # 1) a state_dict for graph_tower directly (keys like 'layer_output_weights.0', ...)
        # 2) a dict wrapper {'model': {...}} or keys like 'model.graph_tower.xxx'
        # 3) {'state_dict': {...}} or {'model_state_dict': {...}}
        cand = None

        # try common wrappers
        if isinstance(raw, dict):
            print("case 1...")
            # case 2: top-level keys with 'model.graph_tower.' prefix
            # build a sub-state-dict by extracting keys that contain 'graph_tower'
            extracted = {}
            for k, v in raw.items():
                if k.startswith('model.graph_tower.'):
                    newk = k[len('model.graph_tower.'):]
                    extracted[newk] = v
            if extracted:
                cand = extracted

            # case 'state_dict' or 'model_state_dict' wrappers
            if cand is None:
                print("case 2...")
                for wrapper_key in ('state_dict', 'model_state_dict', 'model'):
                    if wrapper_key in raw and isinstance(raw[wrapper_key], dict):
                        nested = raw[wrapper_key]
                        # try similar extraction inside nested dict
                        extracted = {}
                        for k, v in nested.items():
                            if k.startswith('model.graph_tower.'):
                                newk = k[len('model.graph_tower.'):]
                                extracted[newk] = v
                        if extracted:
                            cand = extracted
                            break
                        # maybe nested is already graph_tower state_dict
                        # check a few keys for expected graph keys
                        sample_keys = list(nested.keys())[:10]
                        # heuristic: if any key starts with 'layer_output_weights' we assume it's the direct state_dict
                        if any(sk.startswith('layer_output_weights') or sk.startswith('layer_output_biases') or sk.startswith('layer_degree_filters') for sk in sample_keys):
                            cand = nested
                            break

        # if still none, maybe raw itself is the graph_tower state_dict
        if cand is None:
            # if keys look like graph tower keys, accept raw
            if isinstance(raw, dict):
                print("case 3...")
                sample_keys = list(raw.keys())[:10]
                if any(sk.startswith('layer_output_weights') or sk.startswith('layer_output_biases') or sk.startswith('layer_degree_filters') for sk in sample_keys):
                    cand = raw

        if cand is None:
            print("case 4...")
            # give diagnostic message and return
            return False, f"Could not find graph_tower keys inside the file. Top-level keys: {list(raw.keys())[:30]}"

        # Now cand is the actual state_dict for graph_tower (keys aligned to gt.load_state_dict expectations)
        # load with strict=False (safe), but first move module to cpu then load
        gt_cpu = gt.to('cpu')
        try:
            # Use load_state_dict; if gt has custom _load_state_dict method, prefer it if plain load fails.
            try:
                gt_cpu.load_state_dict(cand, strict=False)
            except Exception as e:
                if hasattr(gt_cpu, '_load_state_dict'):
                    gt_cpu._load_state_dict(cand, strict=False)
                else:
                    raise

            # move gt to target device and dtype consistent with model
            # if model is on a device, prefer that
            # Use float() for projector compatibility; for fp16 model you may want gt.to(torch.float16)
            gt = gt_cpu.to(device).float()
            # rebind back to model - need to set model.model.graph_tower (not model.graph_tower)
            model.model.graph_tower = gt
            # freeze parameters
            gt.requires_grad_(False)

            if gt is not None:
                # 打印几项参数信息
                for i, p in enumerate(gt.parameters()):
                    if i >= 6: break
                    print(f"param {i}: device={p.device}, dtype={p.dtype}, shape={tuple(p.shape)}, norm={p.norm().item()}")
            else:
                print("graph_tower still None after load.")

            print("Loaded graph_tower state successfully.")
        
        except Exception as e:
            return False, f"Failed to load state_dict into graph_tower: {e}"
        

    # Use float32 to avoid NaN issues
    model = model.float().to(device)
    model.eval()

    # read test data (assume jsonl with fields like: 'conversations' and 'smiles' and 'label' or 'conversations' where conv[1]['value'] is gt)
    test_examples = list(read_jsonl(os.path.expanduser(args.test_file)))
    # Optional: limit test samples via args
    if args.max_samples and args.max_samples > 0:
        test_examples = test_examples[:args.max_samples]
    print(f"Loaded {len(test_examples)} test examples from {args.test_file}")
    

    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)) or ".", exist_ok=True)
    # if append mode and file exists, skip already computed lines
    answers_path = os.path.expanduser(args.answers_file)
    start_idx = 0
    if args.append and os.path.exists(answers_path):
        start_idx = len(open(answers_path, "r").readlines())
        print(f"Answers file exists. Will append. Skipping first {start_idx} samples.")

    outfile = open(answers_path, "a" if args.append else "w", buffering=1, encoding="utf-8")
    outs = []

    # batch processing
    for batch in tqdm(list(batch_iterable(test_examples[start_idx:], args.batch_size)), desc="Batches"):
        # for each sample in batch, construct prompt and collect SMILES and gts
        prompts = []
        conv_objs = []
        smiles_list = []
        gts = []
        for sample in batch:
            # support two formats:
            # 1) sample contains "conversations" (your graph-based formatting)
            # 2) sample contains "instruction" (a single instruction string) as in eval_hight
            if "conversations" in sample:
                conv = sample["conversations"]
                # build prompt similarly to preprocess_v1: expect conv be list of messages like {"from": "...", "value": "..."}
                # But to be robust, use make_prompt_from_conversation
                prompt_str, conv_obj = make_prompt_from_conversation(conv, conv_mode=args.conv_mode)
            elif "instruction" in sample:
                # plain instruction
                conv_obj = conv_templates[args.conv_mode].copy()
                conv_obj.append_message(conv_obj.roles[0], sample["instruction"])
                conv_obj.append_message(conv_obj.roles[1], None)
                prompt_str = conv_obj.get_prompt()
            else:
                raise ValueError("Unsupported sample format: expect 'conversations' or 'instruction' in jsonl line")

            prompts.append(prompt_str)
            conv_objs.append(conv_obj)

            # SMILES presence
            smiles = sample.get("smiles", None) or sample.get("SMILES", None)
            smiles_list.append(smiles)

            # ground truth label
            # try 'label', else second message value if conv structured as [human, assistant] etc.
            if "label" in sample:
                gt = sample["label"]
            else:
                # fallback: conversations[1]['value'] if present
                if "conversations" in sample and len(sample["conversations"]) > 1:
                    gt = sample["conversations"][1].get("value", "")
                else:
                    gt = sample.get("gt", "")
            gts.append(gt)

        # tokenize prompts. Use tokenizer_graph_token if multimodal token handling is required.
        # We'll call tokenizer(...) batch tokenization as fallback if tokenizer_graph_token not available for batch.
        input_ids_list = []
        for p in prompts:
            # prefer tokenizer_graph_token (handles graph tokens) if available
            try:
                toks = tokenizer_graph_token(p, tokenizer, return_tensors="pt")
                # tokenizer_graph_token returns 1D tensor; unsqueeze to maintain batch dims later
                if isinstance(toks, torch.Tensor):
                    input_ids_list.append(toks.to(device))
                else:
                    # if returns dict
                    input_ids_list.append(toks["input_ids"].squeeze(0).to(device))
            except Exception:
                # fallback
                tok = tokenizer(p, return_tensors="pt", padding=False)
                input_ids_list.append(tok["input_ids"].squeeze(0).to(device))

        # pad to max length in batch
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

        # build stopping criteria using conv_objs if needed
        stop_strs = []
        for conv in conv_objs:
            sep = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stop_strs.append(sep)

        # Use generate with custom kwargs; pass smiles list to let model process SMILES (multimodal forward expects 'smiles' argument)
        # convert smiles_list to python list (some may be None)
        # The model.generate wrapper should forward kwargs to the model; we pass `smiles=smiles_list`
        # If your model expects `smiles` as list of strings, pass that. If it expects list of list, adjust accordingly.
        gen_kwargs = dict(
            do_sample= True if args.do_sample else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True
        )
        # pass smiles
        gen_kwargs["smiles"] = smiles_list

        # Debug: print first batch info (uncomment for debugging)
        # if len(outs) == 0:
        #     print(f"[DEBUG] First batch smiles: {smiles_list[:2]}")
        #     print(f"[DEBUG] First prompt length: {len(prompts[0])}")
        #     print(f"[DEBUG] First prompt: {prompts[0]}")
        #     print(f"[DEBUG] input_ids_list[0] shape: {input_ids_list[0].shape}")

        # Use logits-based classification instead of generation
        # Get token ids for "Yes" and "No"
        yes_id = tokenizer.encode("Yes")[-1]
        no_id = tokenizer.encode("No")[-1]

        with torch.inference_mode():
            # Call model forward to get logits
            try:
                outputs = model(
                    input_ids=input_ids_batch,
                    smiles=smiles_list,
                )
                # Get logits for last token position
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            except Exception as e:
                print(f"[ERROR] Forward failed: {e}")
                # Fallback: all predictions are "No"
                outputs_texts = ["No"] * len(prompts)
                for prompt, gt, pred in zip(prompts, gts, outputs_texts):
                    rec = {"prompt": prompt, "gt": gt, "pred": pred}
                    outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    outs.append(rec)
                continue

            # Check for NaN
            if torch.isnan(logits).any():
                print(f"[WARNING] NaN detected in logits!")
                # Default to random prediction
                logits = torch.zeros_like(logits)

            # Get probabilities for Yes/No using softmax over [no, yes]
            yes_logits = logits[:, yes_id]
            no_logits = logits[:, no_id]

            # Compute probability of Yes using softmax
            stacked = torch.stack([no_logits, yes_logits], dim=-1)  # [batch, 2]
            probs = torch.softmax(stacked, dim=-1)
            yes_probs = probs[:, 1]  # probability of "Yes"

        # Determine predictions based on logits
        outputs_texts = []
        yes_prob_list = []
        for i in range(len(prompts)):
            if yes_logits[i] > no_logits[i]:
                pred = "Yes"
            else:
                pred = "No"
            outputs_texts.append(pred)
            yes_prob_list.append(yes_probs[i].item())

        # write per-sample and append to outs for metric calculation
        for prompt, gt, pred, prob in zip(prompts, gts, outputs_texts, yes_prob_list):
            rec = {"prompt": prompt, "gt": gt, "pred": pred, "yes_prob": prob}
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outs.append(rec)

    outfile.close()

    # compute ROC-AUC using probability scores for better ranking
    preds_bin = [convert_label_to_int(r["pred"]) for r in outs]
    gts_bin = [convert_label_to_int(r["gt"]) for r in outs]
    yes_probs = [r.get("yes_prob", 0.5) for r in outs]

    # If all gts are same label, roc_auc_score will error; handle gracefully
    try:
        # Use probability scores for ROC-AUC (better ranking metric)
        score = roc_auc_score(gts_bin, yes_probs)
    except Exception as e:
        print("Cannot compute ROC-AUC (maybe all labels are same).", e)
        score = None

    # Also compute accuracy
    correct = sum(1 for p, g in zip(preds_bin, gts_bin) if p == g)
    accuracy = correct / len(outs) if outs else 0

    # Count predictions
    yes_pred_count = sum(preds_bin)
    yes_gt_count = sum(gts_bin)

    print("=" * 50)
    print("Samples evaluated:", len(outs))
    print(f"Ground truth: {yes_gt_count} Yes, {len(outs) - yes_gt_count} No")
    print(f"Predictions:  {yes_pred_count} Yes, {len(outs) - yes_pred_count} No")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(outs)})")
    print(f"ROC-AUC (prob-based): {score}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="checkpoints/hiv/llaga-vicuna-7b-neuralfp-linear-projector_cls")
    parser.add_argument("--model-base", type=str, default=None, help="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--graph-checkpoint-path", type=str, default=None, help="checkpoints/hiv/llaga-vicuna-7b-neuralfp-linear-projector_cls/graph_tower.bin")
    parser.add_argument("--test-file", type=str, required=True, help="Test jsonl file with samples")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")

    parser.add_argument("--append", action="store_true", help="Append to answers_file if exists")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to evaluate (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # seet seed
    set_seed_deterministic(args.seed)

    main(args)
