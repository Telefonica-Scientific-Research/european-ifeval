#!/usr/bin/env python3
# © 2026 Telefónica Innovación Digital S.L.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate European IFEval response JSONL with a Hugging Face model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_TEMPLATE_PATH = (
    REPO_ROOT / "prompt_templates" / "alpaca_template_multilingual.py"
)
PROMPT_TEMPLATE_LANGUAGE_ALIASES = {
    "nn": "nno",
    "no": "nob",
    "oc": "oci",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a prompt/response JSONL file for the European IFEval "
            "evaluator using an off-the-shelf Hugging Face causal LM."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or path.")
    parser.add_argument("--language", required=True, help="Dataset language code.")
    parser.add_argument(
        "--model-type",
        choices=["it", "pretrained"],
        default="it",
        help="Use a chat template for 'it' or a raw prompt for 'pretrained'.",
    )
    parser.add_argument("--input", type=Path, help="Input JSONL; defaults to data/LANGUAGE.jsonl.")
    parser.add_argument("--output", type=Path, help="Output response JSONL path.")
    parser.add_argument("--tokenizer", help="Optional tokenizer ID or path.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate only the first N responses (useful for a smoke test).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code supplied by the model repository.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def load_prompt_templates() -> dict:
    spec = importlib.util.spec_from_file_location(
        "european_ifeval_prompt_templates", PROMPT_TEMPLATE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompt templates: {PROMPT_TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROMPT


def prompt_template_for(language: str, templates: dict) -> dict:
    template_language = PROMPT_TEMPLATE_LANGUAGE_ALIASES.get(language, language)
    try:
        return templates[template_language]
    except KeyError as exc:
        available = ", ".join(sorted(templates))
        raise ValueError(
            f"No prompt template for {language!r}. Available: {available}"
        ) from exc


def read_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"{path}:{line_number}: missing string 'prompt'")
            prompts.append(row["prompt"])
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def render_prompt(tokenizer, prompt: str, template: dict, model_type: str) -> str:
    prompt_text = template["prompt_no_input"].format_map({"instruction": prompt})
    if model_type == "pretrained":
        return prompt_text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Generation dependencies are missing. Install them with: "
            "python -m pip install -r requirements-generation.txt"
        ) from exc

    input_path = args.input or REPO_ROOT / "data" / f"{args.language}.jsonl"
    model_name = args.model.rstrip("/").split("/")[-1]
    output_path = args.output or (
        REPO_ROOT
        / "outputs"
        / "generated_responses"
        / args.language
        / model_name
        / "off-the-shelf.jsonl"
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    prompts = read_prompts(input_path, args.limit)
    templates = load_prompt_templates()
    template = prompt_template_for(args.language, templates)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    added_pad_token = False
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True

    dtype = getattr(torch, args.torch_dtype)
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }
    if torch.cuda.is_available():
        model_kwargs.update(device_map="auto", dtype=dtype)
    else:
        model_kwargs["dtype"] = torch.float32
        print("CUDA is unavailable; generation will run on CPU in float32.")

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for index, prompt in enumerate(prompts, 1):
            rendered = render_prompt(tokenizer, prompt, template, args.model_type)
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            inputs.pop("token_type_ids", None)
            input_length = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                generated[0][input_length:], skip_special_tokens=True
            ).strip()
            output_handle.write(
                json.dumps(
                    {"prompt": prompt, "response": response}, ensure_ascii=False
                )
                + "\n"
            )
            output_handle.flush()
            print(f"Generated {index}/{len(prompts)}")

    print(f"Wrote responses: {output_path}")


if __name__ == "__main__":
    main()
