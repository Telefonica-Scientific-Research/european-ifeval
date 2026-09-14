# European IFEval evaluator

Run the evaluator as a package from the repository root:

```bash
python -m evaluation.evaluation_main \
  --input_data=data/ca.jsonl \
  --input_response_data=/path/to/ca-responses.jsonl \
  --output_dir=outputs/results/ca/MODEL/EXPERIMENT \
  --glotlid_model_path=/path/to/glotlid/model.bin \
  --glotlid_min_confidence=0.5
```

The input response file must contain one JSON object per line with an exact
input prompt and its corresponding response:

```json
{"prompt": "...", "response": "..."}
```

If `--glotlid_model_path` is omitted, the evaluator uses
`GLOTLID_MODEL_PATH` or the Hugging Face cache/download mechanism for
`cis-lmu/glotlid`.
