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

"""Tests for the evaluator command-line report."""

import contextlib
import io

from absl.testing import absltest

from evaluation import evaluation_main


class EvaluationMainTest(absltest.TestCase):

  def test_print_report_labels_scores_and_counts(self):
    outputs = [
        evaluation_main.OutputExample(
            instruction_id_list=["format:first", "content:second"],
            prompt="First prompt",
            response="First response",
            follow_all_instructions=False,
            follow_instruction_list=[True, False],
        ),
        evaluation_main.OutputExample(
            instruction_id_list=["format:first"],
            prompt="Second prompt",
            response="Second response",
            follow_all_instructions=True,
            follow_instruction_list=[True],
        ),
    ]

    report = io.StringIO()
    with contextlib.redirect_stdout(report):
      evaluation_main.print_report(outputs)

    self.assertEqual(
        report.getvalue(),
        "Prompt-level accuracy:      0.500 (1/2)\n"
        "Instruction-level accuracy: 0.667 (2/3)\n"
        "Per-instruction accuracy:\n"
        "  content:second: 0.000 (0/1)\n"
        "  format:first: 1.000 (2/2)\n",
    )


if __name__ == "__main__":
  absltest.main()
