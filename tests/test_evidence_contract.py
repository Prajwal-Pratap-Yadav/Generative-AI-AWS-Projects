import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from awsgenai_audit import (
    NOTEBOOKS,
    bounded_read,
    compare_prompts,
    main,
    notebook_summary,
)


class EvidenceContract(unittest.TestCase):
    def book(self, root, source, outputs=None):
        path = root / "sample.ipynb"
        path.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "source": source,
                            "outputs": outputs or [],
                        }
                    ]
                }
            )
        )
        return path

    def test_empty_comparison_never_reports_equal_generation_parameters(self):
        result = compare_prompts({"literal_prompts": []}, {"literal_prompts": []})
        self.assertEqual(result["pairing_status"], "no_pairs")
        self.assertIsNone(result["all_matched_generation_parameters_equal"])
        self.assertIsNone(result["all_matched_prompt_bytes_equal"])

    def test_missing_explicit_parameters_are_distinct_from_a_verified_match(self):
        prompt = {"literal_prompts": [{"prompt": "A", "parameters": {}}]}
        self.assertEqual(
            compare_prompts(prompt, prompt)["pairing_status"], "implicit_parameters"
        )

    def test_comments_and_strings_are_not_cloud_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.book(
                Path(directory),
                '# estimator.fit({})\nexample="model.deploy()"\nestimator.fit({})\npredictor.delete_endpoint ()',
            )
            result = notebook_summary(path)
            self.assertEqual(result["training_calls"], 1)
            self.assertEqual(result["deployment_calls"], 0)
            self.assertTrue(result["contains_endpoint_cleanup"])

    def test_nested_payload_does_not_become_a_saved_evaluation_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.book(
                Path(directory),
                "def unused():\n    payload={'inputs':'unasked','parameters':{'temperature':0.6}}",
            )
            result = notebook_summary(path)
            self.assertEqual(result["literal_prompts"], [])
            self.assertEqual(len(result["skipped_payloads"]), 1)

    def test_multiple_payloads_in_one_cell_count_one_response_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            source = "payload={'inputs':'a'}\npayload={'inputs':'b'}"
            path = self.book(
                Path(directory),
                source,
                [{"data": {"text/plain": ["> saved response"]}}],
            )
            result = notebook_summary(path)
            self.assertEqual(len(result["literal_prompts"]), 2)
            self.assertEqual(result["saved_response_cells"], 1)

    def test_non_json_parameters_are_rejected_before_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            for parameters in ["None", "[]", "{'x': {1,2}}", "{'x': 1j}"]:
                path = self.book(
                    Path(directory),
                    "payload={'inputs':'A','parameters':" + parameters + "}",
                )
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    notebook_summary(path)

    def test_magic_cell_omission_is_visible_and_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.book(Path(directory), "!pip install forbidden\n")
            result = notebook_summary(path)
            self.assertEqual(result["unparsed_code_cells"], [0])
            self.assertEqual(result["literal_prompts"], [])

    def test_a_payload_hidden_in_an_unparseable_cell_prevents_a_complete_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "payload={'inputs':'A','parameters':{'temperature':0.6}}"
            left = notebook_summary(self.book(root, source))
            book = {
                "cells": [
                    {"cell_type": "code", "source": source, "outputs": []},
                    {
                        "cell_type": "code",
                        "source": "!not-a-python-command\npayload=build_dynamic()",
                        "outputs": [],
                    },
                ]
            }
            path = root / "other.ipynb"
            path.write_text(json.dumps(book))
            right = notebook_summary(path)
            self.assertEqual(
                compare_prompts(left, right)["pairing_status"], "unsupported_payloads"
            )

    def test_notebook_size_limit_is_enforced_during_the_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.ipynb"
            path.write_bytes(b" " * 33)
            with patch("awsgenai_audit.MAX_BYTES", 32), self.assertRaises(ValueError):
                bounded_read(path)

    def test_incomplete_pairing_emits_report_and_fails_the_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in NOTEBOOKS:
                (root / name).write_text(json.dumps({"cells": []}))
            (root / "ITDataset.txt").write_text("fixture")
            output = root / "audit.json"
            args = ["audit", "--root", str(root), "--output", str(output)]
            with (
                patch("sys.argv", args),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(
                json.loads(output.read_text())["comparison"]["pairing_status"],
                "no_pairs",
            )


if __name__ == "__main__":
    unittest.main()
