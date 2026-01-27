#!/usr/bin/env python3
"""
将 PEFT/LoRA 权重合并到基础模型并保存为常规 Hugging Face 格式的脚本。

用途示例：
  # 仅做检查（dry-run）
  python scripts/merge_lora_to_hf.py --lora-dir sft_without_paper --dry-run

  # 执行合并并保存到 output_dir
  python scripts/merge_lora_to_hf.py --lora-dir sft_without_paper --output-dir out_model --merge

注意：默认会从 `adapter_config.json` 中读取 `base_model_name_or_path`，若需要可通过 --base-model 覆盖。

依赖：transformers, peft, accelerate, safetensors, torch
pip install transformers peft accelerate safetensors torch
"""
import argparse
import json
import os
import sys

def read_adapter_config(lora_dir):
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {lora_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model and save as HF model")
    parser.add_argument("--lora-dir", required=True, help="Path to the folder containing LoRA weights and adapter_config.json (e.g. sft_without_paper)")
    parser.add_argument("--base-model", default=None, help="Optional override for base model name or path. If omitted, read from adapter_config.json")
    parser.add_argument("--output-dir", default="merged_model", help="Where to save the merged HF model")
    parser.add_argument("--dtype", choices=["float16","float32"], default="float16", help="Torch dtype to load model with (for speed/memory)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading model (useful for some community models)")
    parser.add_argument("--merge", action="store_true", help="Actually perform merge; default is dry-run that only prints planned actions")
    parser.add_argument("--device-map", default="auto", help="Device map to pass to from_pretrained (default 'auto')")
    args = parser.parse_args()

    lora_dir = args.lora_dir
    if not os.path.isdir(lora_dir):
        print(f"lora-dir {lora_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    cfg = read_adapter_config(lora_dir)
    base_model = args.base_model or cfg.get("base_model_name_or_path")
    if not base_model:
        print("base model not found in adapter_config.json; please pass --base-model", file=sys.stderr)
        sys.exit(2)

    print("LoRA dir:", lora_dir)
    print("Base model:", base_model)
    print("Output dir:", args.output_dir)
    print("Merge mode:", args.merge)

    # dry-run check
    if not args.merge:
        print("Dry run: no heavy model loading will occur. Re-run with --merge to perform actual merge.")
        print("Planned steps:")
        print(" 1) load base model and tokenizer from:", base_model)
        print(" 2) wrap base model with PeftModel.from_pretrained(..., lora_dir)")
        print(" 3) call merge_and_unload() to merge LoRA weights into base model")
        print(" 4) save merged model and tokenizer to:", args.output_dir)
        return

    # perform actual merge
    # delayed imports so dry-run is cheap
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftModel
    except Exception as e:
        print("Failed to import peft. Please install with: pip install peft", file=sys.stderr)
        raise

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    load_kwargs = dict(torch_dtype=dtype, device_map=args.device_map)
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    print("Loading base model (this may be large) ...")
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    # tokenizer: prefer tokens in lora dir (if present), otherwise load from base
    tokenizer = None
    for tok_name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(lora_dir, tok_name)):
            try:
                tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=args.trust_remote_code)
                print("Loaded tokenizer from lora dir")
                break
            except Exception:
                tokenizer = None
    if tokenizer is None:
        print("Loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)

    print("Applying LoRA weights from:", lora_dir)
    peft_model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=dtype)

    print("Merging and unloading LoRA to produce a single merged model (this will modify model weights)...")
    try:
        # merge_and_unload returns the merged base model in many peft versions
        merged = peft_model.merge_and_unload()
        if merged is None:
            # fallback: maybe peft_model itself is now merged
            merged = model
    except Exception as e:
        print("merge_and_unload() failed:", e, file=sys.stderr)
        print("Attempting alternative: save_pretrained on peft_model (may still be PEFT-wrapped)...")
        merged = peft_model

    print("Saving merged model to:", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # Save model and tokenizer
    try:
        merged.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print("Failed to save merged model:", e, file=sys.stderr)
        raise

    print("Done. Merged model saved at:", args.output_dir)


if __name__ == "__main__":
    main()
