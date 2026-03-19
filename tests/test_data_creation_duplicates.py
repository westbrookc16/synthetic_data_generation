import importlib.util
import unittest
from pathlib import Path


DATA_CREATION_PATH = Path(__file__).resolve().parent.parent / "data-creation.py"
spec = importlib.util.spec_from_file_location("data_creation", DATA_CREATION_PATH)
data_creation = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(data_creation)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class FakeCompletions:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("No more fake payloads available")
        return FakeResponse(self.payloads.pop(0))


class FakeChat:
    def __init__(self, payloads):
        self.completions = FakeCompletions(payloads)


class FakeClient:
    def __init__(self, payloads):
        self.chat = FakeChat(payloads)


class DataCreationDuplicateTests(unittest.TestCase):
    def test_normalize_question_text_strips_case_and_punctuation(self) -> None:
        normalized = data_creation.normalize_question_text("How do I fix a sink leak?!  ")
        self.assertEqual(normalized, "how do i fix a sink leak")

    def test_classify_question_similarity_checks_global_exact_and_category_near(self) -> None:
        result_exact = data_creation.classify_question_similarity(
            candidate_question="How do I fix a sink leak?",
            global_normalized_questions={"how do i fix a sink leak"},
            category_normalized_questions=[],
            similarity_threshold=0.9,
        )
        result_near = data_creation.classify_question_similarity(
            candidate_question="How do I stop my sink from leaking overnight?",
            global_normalized_questions=set(),
            category_normalized_questions=["how do i stop a sink from leaking overnight"],
            similarity_threshold=0.9,
        )

        self.assertEqual(result_exact, "duplicate")
        self.assertEqual(result_near, "near-duplicate")

    def test_generate_records_uses_same_category_history_only(self) -> None:
        payloads = [
            self._payload("Why is my refrigerator making a buzzing noise and not cooling?"),
            self._payload("How do I stop a kitchen sink faucet from dripping constantly?"),
            self._payload("Why does one outlet in my bedroom stop working intermittently?"),
            self._payload("Why is my thermostat on but the house still feels warm?"),
            self._payload("How do I patch a small hole in drywall near a door frame?"),
            self._payload("How do I stop my fridge from buzzing loudly while running?"),
        ]
        client = FakeClient(payloads)

        records = data_creation.generate_records(
            client=client,
            model="fake-model",
            count=6,
            temperature=0.9,
            top_p=1.0,
            question_similarity_threshold=0.98,
            request_delay_seconds=0,
            retry_max=0,
        )

        self.assertEqual(len(records), 6)
        sixth_user_prompt = client.chat.completions.calls[5]["messages"][1]["content"]
        self.assertIn("refrigerator making a buzzing noise", sixth_user_prompt.lower())
        self.assertNotIn("kitchen sink faucet", sixth_user_prompt.lower())
        self.assertNotIn("bedroom stop working intermittently", sixth_user_prompt.lower())

    @staticmethod
    def _payload(question: str):
        return {
            "question": question,
            "answer": "Turn off power or water as needed, inspect the component, replace the worn part, and test the repair safely.",
            "equipment_problem": "Common household repair issue",
            "tools_required": ["Screwdriver", "Flashlight", "Replacement part"],
            "steps": [
                "1. Shut off the relevant power or water supply.",
                "2. Inspect the part causing the issue and remove it carefully.",
                "3. Install the replacement part and test the repair.",
            ],
            "safety_info": "Disconnect power or water before starting, and wear appropriate protective gear for the task.",
            "tips": ["Take a photo before disassembly to make reassembly easier."],
        }


if __name__ == "__main__":
    unittest.main()

