#!/usr/bin/env python3
"""
Evaluate model on ChEBI-20 molecule captioning task.
Computes BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, and METEOR scores.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
import numpy as np

sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./model")

from model.builder import load_pretrained_model
from utils.utils import tokenizer_graph_token, disable_torch_init, get_model_name_from_path
from utils.conversation import conv_templates, SeparatorStyle

# Import metrics
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


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

    predictions = []
    references = []
    pred_texts = []
    ref_texts = []

    for batch in tqdm(list(batch_iterable(test_examples, args.batch_size)), desc="Batches"):
        prompts = []
        smiles_list = []
        gts = []

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
        input_ids_batch = torch.full(
            (len(input_ids_list), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device
        )
        for i, ids in enumerate(input_ids_list):
            # Left pad: place tokens at the end
            input_ids_batch[i, max_len - ids.shape[0]:] = ids

        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids_batch,
                smiles=smiles_list,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=512,
                use_cache=True
            )

        # Decode outputs
        for i, (output, gt) in enumerate(zip(outputs, gts)):
            # Filter out invalid token IDs that are out of vocabulary range
            vocab_size = len(tokenizer)
            valid_output = [t for t in output.tolist() if 0 <= t < vocab_size]
            try:
                pred_text = tokenizer.decode(valid_output, skip_special_tokens=True)
            except Exception as e:
                pred_text = ""
            # Extract the generated portion (after the prompt)
            if "ASSISTANT:" in pred_text:
                pred_text = pred_text.split("ASSISTANT:")[-1].strip()

            rec = {
                "smiles": smiles_list[i],
                "gt": gt,
                "pred": pred_text
            }
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Tokenize for BLEU
            pred_tokens = pred_text.lower().split()
            ref_tokens = gt.lower().split()

            predictions.append(pred_tokens)
            references.append([ref_tokens])
            pred_texts.append(pred_text)
            ref_texts.append(gt)

    outfile.close()

    # Compute metrics
    print("=" * 50)
    print("ChEBI-20 Captioning Results")
    print("=" * 50)

    # BLEU scores
    smooth = SmoothingFunction().method1
    bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

    # METEOR scores
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        try:
            score = meteor_score(ref, pred)
            meteor_scores.append(score)
        except:
            pass
    if meteor_scores:
        avg_meteor = np.mean(meteor_scores)
        print(f"METEOR: {avg_meteor:.4f}")

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(pred_texts, ref_texts):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
    print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
    print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
    print("=" * 50)

    # Summary in one line (for easy comparison)
    print(f"\nSummary: BLEU-2={bleu2:.4f}, BLEU-4={bleu4:.4f}, ROUGE-1={np.mean(rouge1_scores):.4f}, ROUGE-2={np.mean(rouge2_scores):.4f}, ROUGE-L={np.mean(rougeL_scores):.4f}, METEOR={avg_meteor:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="chebi20_answers.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    main(args)
