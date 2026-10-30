import json
from pathlib import Path
import tempfile
import unittest

from awsgenai_audit import audit, compare_prompts, normalize_prompt, notebook_summary


class NotebookAuditTests(unittest.TestCase):
    def write_book(self, root, source):
        p = root / "book.ipynb"
        p.write_text(
            json.dumps(
                {"cells": [{"cell_type": "code", "source": source, "outputs": []}]}
            )
        )
        return p

    def test_prompt_normalization_pairs_case_and_whitespace_only(self):
        self.assertEqual(normalize_prompt("  Outline\nDATA "), "outline data")

    def test_literal_payload_is_parsed_without_execution(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write_book(
                Path(d),
                "payload={'inputs':'Question', 'parameters':{'temperature':0.6}}\nraise RuntimeError('must not run')",
            )
            result = notebook_summary(path)
            self.assertEqual(len(result["literal_prompts"]), 1)

    def test_dynamic_payload_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write_book(
                Path(d), "payload=__import__('os').system('must-not-run')"
            )
            self.assertEqual(notebook_summary(path)["literal_prompts"], [])

    def test_duplicate_prompts_fail_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write_book(
                Path(d), "payload={'inputs':'A'}\npayload={'inputs':'a'}"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                notebook_summary(path)

    def test_different_generation_settings_are_reported(self):
        a = {"literal_prompts": [{"prompt": "a", "parameters": {"temperature": 0.1}}]}
        b = {"literal_prompts": [{"prompt": "A", "parameters": {"temperature": 0.6}}]}
        self.assertFalse(
            compare_prompts(a, b)["all_matched_generation_parameters_equal"]
        )

    def test_missing_prompt_is_not_hidden(self):
        a = {"literal_prompts": [{"prompt": "a", "parameters": {}}]}
        result = compare_prompts(a, {"literal_prompts": []})
        self.assertEqual(result["matched_prompts"], 0)
        self.assertEqual(result["unpaired_base"], ["a"])

    def test_repository_notebooks_have_three_comparable_prompt_pairs(self):
        result = audit(Path(__file__).resolve().parents[1])
        self.assertEqual(result["comparison"]["matched_prompts"], 3)
        self.assertTrue(result["comparison"]["all_matched_generation_parameters_equal"])
        self.assertFalse(result["corpus"]["redistribution_rights_verified"])


if __name__ == "__main__":
    unittest.main()
