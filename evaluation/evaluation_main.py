# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Modifications Copyright © 2026 Telefónica Innovación Digital S.L.
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

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
from typing import Dict, Optional, Sequence, Union

from absl import app
from absl import flags
from absl import logging

from . import instructions
from . import instructions_registry

_INPUT_DATA = flags.DEFINE_string(
    "input_data", "./data/input_data.jsonl",
    "path to input data", required=False
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", "./data/input_response_data.jsonl",
    "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "./output/",
    "Output directory for inference and eval results.",
    required=False,
)

_GLOTLID_MODEL_PATH = flags.DEFINE_string(
    "glotlid_model_path",
    None,
    "Optional local path to the GlotLID FastText model. If omitted, the "
    "evaluator uses GLOTLID_MODEL_PATH or Hugging Face cache/download.",
    required=False,
)

_GLOTLID_MIN_CONFIDENCE = flags.DEFINE_float(
    "glotlid_min_confidence",
    0.5,
    "Minimum GlotLID prediction confidence accepted for language rules.",
    required=False,
)


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def _normalize_instruction_kwargs(raw_kwargs):
    """Drops nulls and converts parquet-style integer floats to ints."""
    clean_kwargs = {}
    for key, value in raw_kwargs.items():
        if value is None:
            continue
        if isinstance(value, float) and value.is_integer():
            clean_kwargs[key] = int(value)
        else:
            clean_kwargs[key] = value
    return clean_kwargs


def read_prompt_list(input_jsonl_filename):
    """Read inputs from jsonl."""
    inputs = []
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            example = json.loads(l)
            inputs.append(
                InputExample(key=example["key"],
                             instruction_id_list=example["instruction_id_list"],
                             prompt=example["prompt"],
                             kwargs=example["kwargs"]))
    return inputs


def write_outputs(output_jsonl_filename, outputs):
    """Writes outputs to jsonl."""
    assert outputs
    with open(output_jsonl_filename, "w") as f:
        for o in outputs:
            f.write(
                json.dumps(
                    {
                        attr_name: o.__getattribute__(attr_name)
                        for attr_name in [
                        name for name in dir(o) if not name.startswith("_")
                    ]
                    }
                )
            )
            f.write("\n")


def write_language_detection_failures(output_jsonl_filename, failures):
    """Writes structured language-detection failure diagnostics to jsonl."""
    with open(output_jsonl_filename, "w") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False))
            f.write("\n")


def test_instruction_following_strict(
        inp,
        prompt_to_response,
):
    """Tests response to see if instrutions are followed."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        raw_kwargs = inp.kwargs[index] if index < len(inp.kwargs) else {}
        clean_kwargs = _normalize_instruction_kwargs(raw_kwargs)
        instruction.build_description(**clean_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        instructions.set_language_detection_context(
            evaluation_mode="strict",
            key=inp.key,
            prompt=inp.prompt,
            instruction_id=instruction_id,
            instruction_index=index,
        )
        try:
            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)
        finally:
            instructions.clear_language_detection_context()

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
        inp,
        prompt_to_response,
):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.prompt]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    variant_names = [
        "original",
        "without_asterisks",
        "without_first_line",
        "without_last_line",
        "without_first_and_last_line",
        "without_first_line_and_asterisks",
        "without_last_line_and_asterisks",
        "without_first_last_line_and_asterisks",
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        raw_kwargs = inp.kwargs[index] if index < len(inp.kwargs) else {}
        clean_kwargs = _normalize_instruction_kwargs(raw_kwargs)
        instruction.build_description(**clean_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for variant_index, r in enumerate(all_responses):
            instructions.set_language_detection_context(
                evaluation_mode="loose",
                key=inp.key,
                prompt=inp.prompt,
                instruction_id=instruction_id,
                instruction_index=index,
                variant_index=variant_index,
                variant_name=variant_names[variant_index],
            )
            try:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break
            finally:
                instructions.clear_language_detection_context()

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def read_prompt_to_response_dict(input_jsonl_filename):
    """Creates dictionary matching prompt and response."""
    return_dict = {}
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            example = json.loads(l)
            return_dict[example["prompt"]] = example["response"]
    return return_dict


def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
                instruction_id_list, follow_instruction_list
        ):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    print(
        "Prompt-level accuracy:      "
        f"{prompt_correct / prompt_total:.3f} "
        f"({prompt_correct}/{prompt_total})"
    )
    print(
        "Instruction-level accuracy: "
        f"{instruction_correct / instruction_total:.3f} "
        f"({instruction_correct}/{instruction_total})"
    )
    print("Per-instruction accuracy:")
    for instruction_id in sorted(tier1_total.keys()):
        accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
        print(
            f"  {instruction_id}: {accuracy:.3f} "
            f"({tier1_correct[instruction_id]}/{tier1_total[instruction_id]})"
        )

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    instructions.configure_language_detector(
        model_path=_GLOTLID_MODEL_PATH.value,
        min_confidence=_GLOTLID_MIN_CONFIDENCE.value,
    )

    inputs = read_prompt_list(_INPUT_DATA.value)
    prompt_to_response = read_prompt_to_response_dict(
        _INPUT_RESPONSE_DATA.value)
    language_detection_failures_by_run = {}

    # get instruction following results
    for func, output_file_name in [
        (test_instruction_following_strict, "eval_results_strict"),
        (test_instruction_following_loose, "eval_results_loose"),
    ]:
        logging.info("Generating %s...", output_file_name)
        instructions.reset_language_detection_failures()
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, prompt_to_response))
        output_jsonl_name = os.path.join(
            _OUTPUT_DIR.value, output_file_name + ".jsonl"
        )
        write_outputs(output_jsonl_name, outputs)
        logging.info("Generated: %s", output_jsonl_name)

        language_detection_failures = (
            instructions.get_language_detection_failure_details()
        )
        language_detection_failures_name = os.path.join(
            _OUTPUT_DIR.value,
            output_file_name.replace(
                "eval_results",
                "language_detection_failures",
            ) + ".jsonl",
        )
        write_language_detection_failures(
            language_detection_failures_name,
            language_detection_failures,
        )
        logging.info("Generated: %s", language_detection_failures_name)

        # Prints instruction following accuracy report.
        print("=" * 64)
        evaluation_mode = output_file_name.removeprefix("eval_results_")
        print(f"{evaluation_mode.capitalize()} evaluation")
        print(f"Results: {output_jsonl_name}")
        print_report(outputs)
        language_detection_failure_count = (
            instructions.get_language_detection_failures()
        )
        print(
            "Language-detection failures: "
            f"{language_detection_failure_count}"
        )
        if language_detection_failure_count:
            print(
                "Note: a language-detection failure means the detector could "
                "not provide a usable decision. Confident wrong-language "
                "predictions count as incorrect but are not included here."
            )
        language_detection_failures_by_run[os.path.basename(output_jsonl_name)] = (
            language_detection_failure_count
        )

    if any(language_detection_failures_by_run.values()):
        summary = ", ".join(
            f"{name}={count}"
            for name, count in language_detection_failures_by_run.items()
        )
        print(f"Language-detection failure summary: {summary}")


if __name__ == "__main__":
    app.run(main)
