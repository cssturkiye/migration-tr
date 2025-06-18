#!/usr/bin/env python
"""
Migration-TR perception-attitude classification inference script
--------------------------------------------------------------

Example
-------
$ python run_inference.py \
      --text "Mültecilere vatandaşlık verilmesin" \
      --adapter-path trained_models/perception_attitude_clf \
      --device cpu
"""

import argparse, sys, json
from pathlib import Path
from adapters import AutoAdapterModel
from transformers import RobertaTokenizer, RobertaConfig, TextClassificationPipeline

def load_pipeline(base_model: str,
                  adapter_path: Path,
                  adapter_name: str = "MPATurk",
                  device: str = "cpu",
                  batch_size: int = 32):
    """Initialise tokenizer, LoRA adapter and inference pipeline."""
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = 128
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation = True
    tokenizer.padding = "max_length"

    config = RobertaConfig.from_pretrained(base_model)
    model  = AutoAdapterModel.from_pretrained(base_model, config=config)
    model.load_adapter(adapter_path / adapter_name)
    model.load_head(adapter_path / "classification_head")
    model.set_active_adapters(adapter_name)

    device_id = -1 if device == "cpu" else 0
    pipe = TextClassificationPipeline(model=model,
                                      tokenizer=tokenizer,
                                      framework="pt",
                                      device=device_id,
                                      batch_size=batch_size)
    return pipe

def main():
    p = argparse.ArgumentParser(description="Run Migration-TR perception-attitude classification")
    p.add_argument("--text", nargs="+", help="one or more Turkish tweets", required=True)
    p.add_argument("--adapter-path", type=Path,
                   default=Path("trained_models/perception_attitude_clf"),
                   help="folder containing MPATurk adapter + classification head")
    p.add_argument("--base-model", default="VRLLab/TurkishBERTweet")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()

    pipe = load_pipeline(args.base_model, args.adapter_path, device=args.device)
    results = pipe(args.text, truncation=True, padding="max_length")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    sys.exit(main())
