import unittest

from label_fields import (
    BASE_RECORD_FIELDS,
    QUALITY_LABEL_FIELDS,
    QUALITY_REVIEW_CSV_FIELDS,
    REVIEW_CSV_FIELDS,
    TOP_LEVEL_LABEL_FIELDS,
)
from prompts import prompt_configs


class LabelFieldsAndPromptTests(unittest.TestCase):
    def test_review_csv_fields_match_shared_field_order(self) -> None:
        expected = BASE_RECORD_FIELDS + TOP_LEVEL_LABEL_FIELDS + QUALITY_REVIEW_CSV_FIELDS + ["notes"]
        self.assertEqual(REVIEW_CSV_FIELDS, expected)

    def test_prompt_types_are_unique_lowercase_and_non_empty(self) -> None:
        prompt_types = [config["type"] for config in prompt_configs]

        self.assertEqual(len(prompt_types), len(set(prompt_types)))
        for prompt_type in prompt_types:
            self.assertEqual(prompt_type, prompt_type.strip())
            self.assertEqual(prompt_type, prompt_type.lower())
            self.assertTrue(prompt_type)

    def test_prompt_texts_are_non_empty_and_do_not_contain_known_typos(self) -> None:
        for config in prompt_configs:
            prompt = config["prompt"]
            self.assertEqual(prompt, prompt.strip())
            self.assertGreater(len(prompt), 40)
            self.assertNotIn("onlly", prompt.lower())
            self.assertNotIn("diyy", prompt.lower())

    def test_quality_field_count_matches_labeling_guide_expectation(self) -> None:
        self.assertEqual(len(QUALITY_LABEL_FIELDS), 8)


if __name__ == "__main__":
    unittest.main()