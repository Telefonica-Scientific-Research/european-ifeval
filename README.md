# European IFEval

European IFEval is a 50-language benchmark for evaluating whether language
models follow verifiable instructions. This repository contains the normalized
benchmark data, the evaluator, and an optional self-contained Hugging Face
generation example.

The core workflow is independent of any model or inference framework:

```text
model or external inference system
              |
              v
{"prompt": "...", "response": "..."} JSONL
              |
              v
European IFEval evaluator
              |
              v
strict and loose scores
```

## Repository contents

- `data/`: 50 normalized JSONL test sets and `source.json` provenance metadata.
- `evaluation/`: instruction-following evaluator and unit tests.
- `examples/`: a minimal valid input/response pair for a smoke test.
- `scripts/1_generate_responses.py`: optional off-the-shelf Hugging Face example.
- `scripts/2_evaluate_generated_responses.sh`: convenience batch evaluator.
- `prompt_templates/`: multilingual prompt templates used by the example generator.

Generated responses, evaluation results, logs, and virtual environments are
excluded from Git.

## Install the evaluator

Python 3.10 is recommended. With pyenv:

```bash
git clone https://github.com/Telefonica-Scientific-Research/european-ifeval.git
cd european-ifeval

pyenv install -s 3.10.0
pyenv local 3.10.0

python -m venv venv-european-ifeval
source venv-european-ifeval/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

Check the installation:

```bash
python -m pip check
python -m unittest discover -s evaluation -p '*_test.py' -v
```

## Prediction file format

The evaluator accepts JSON Lines with one object per benchmark prompt:

```json
{"prompt": "The exact prompt from the input dataset", "response": "The model response"}
```

The `prompt` must exactly match its value in the input dataset. A complete
evaluation requires exactly one response for every input prompt. Response order
does not matter because the evaluator joins records by the prompt string.

Your inference system may use any framework, API, hosted model, or local model
as long as it produces this format.

## Evaluator smoke test

The included example does not contain a language-detection rule, so it does not
need the GlotLID model:

```bash
mkdir -p /tmp/european-ifeval-smoke

python -m evaluation.evaluation_main \
  --input_data=examples/input.example.jsonl \
  --input_response_data=examples/responses.example.jsonl \
  --output_dir=/tmp/european-ifeval-smoke
```

The strict and loose prompt-level scores should both be `1.000`.

## Evaluate model responses

For example, to evaluate Catalan responses:

```bash
mkdir -p outputs/results/ca/my-model/my-experiment

python -m evaluation.evaluation_main \
  --input_data=data/ca.jsonl \
  --input_response_data=/path/to/ca-responses.jsonl \
  --output_dir=outputs/results/ca/my-model/my-experiment \
  --glotlid_min_confidence=0.5
```

The output directory contains:

- `eval_results_strict.jsonl`
- `eval_results_loose.jsonl`
- `language_detection_failures_strict.jsonl`
- `language_detection_failures_loose.jsonl`

The command also prints strict and loose aggregate scores.

### GlotLID

Language constraints are evaluated with
[GlotLID](https://huggingface.co/cis-lmu/glotlid). When a language rule is
encountered, the evaluator loads `cis-lmu/glotlid` through the Hugging Face
cache/download mechanism.

To use a local copy instead:

```bash
export GLOTLID_MODEL_PATH=/path/to/glotlid/model.bin
```

## Optional Hugging Face generation example

Generation is optional and has a separate, much larger dependency set. The
provided example supports ordinary off-the-shelf causal language models; it
does not depend on a training repository or private checkpoint layout.

Install generation dependencies:

```bash
python -m pip install -r requirements-generation.txt
```

Check the command without loading a model:

```bash
python scripts/1_generate_responses.py --help
```

Generate the full Catalan response file:

```bash
python scripts/1_generate_responses.py \
  --model CohereLabs/tiny-aya-global \
  --language ca \
  --model-type it
```

The default output is:

```text
outputs/generated_responses/ca/tiny-aya-global/off-the-shelf.jsonl
```

For a quick generation smoke test, use `--limit`; a partial response file is
useful for checking model loading but cannot be evaluated against the complete
benchmark dataset:

```bash
python scripts/1_generate_responses.py \
  --model CohereLabs/tiny-aya-global \
  --language ca \
  --model-type it \
  --limit 2 \
  --output /tmp/ca-responses-smoke.jsonl
```

Generation uses deterministic greedy decoding. `CohereLabs/tiny-aya-global`
requires a Hugging Face account, acceptance of its access conditions, and
adherence to its non-commercial license and acceptable-use policy. Pass
`--trust-remote-code` only for model repositories you trust.

## Evaluate generated files by convention

If responses use the default generator directory structure, the convenience
script can evaluate one or more languages:

```bash
LANGS="ca de es" \
MODEL_NAME="tiny-aya-global" \
EXP_NAME="off-the-shelf" \
scripts/2_evaluate_generated_responses.sh
```

Results are written under:

```text
outputs/results/{language}/{model}/{experiment}/
```

## Languages

```text
ast bg bs ca cs cy da de el es et eu fi fo fr fur ga gd gl hr hu is it ka lb lij lmo lt ltg lv mk mt nl nn no oc pl pt ro ru sc scn sk sl sq sr sv szl uk vec
```

The repository uses `no` for Norwegian Bokmål and `nn` for Norwegian Nynorsk.
Datasets contain between 522 and 541 prompts because some localized datasets
have missing samples recorded in `data/source.json`. See that file for source
and normalization metadata for each test set.
