import unittest
import pandas as pd
from core.execution_engine import run_code_in_sandbox
from stats.stat_engine import analyze_quantization_impact
from core.recommender import recommend_model
from main import generate_synthetic_evaluation_matrix


class TestLocalLLMFramework(unittest.TestCase):
    
    def test_execution_engine_success(self):
        code = "def square(x):\n    return x * x\n"
        assertions = ["assert square(2) == 4", "assert square(5) == 25"]
        res = run_code_in_sandbox(code, assertions)
        self.assertEqual(res["passed_tests"], 2)
        self.assertEqual(res["total_tests"], 2)
        self.assertAlmostEqual(res["unit_test_pass_rate"], 1.0)

    def test_execution_engine_timeout(self):
        code = "while True: pass\n"
        assertions = ["assert True"]
        res = run_code_in_sandbox(code, assertions, timeout_sec=0.5)
        self.assertEqual(res["passed_tests"], 0)
        self.assertEqual(res["unit_test_pass_rate"], 0.0)
        self.assertIn("TimeoutExpired", res["results"][0]["error"])

    def test_execution_engine_syntax_error(self):
        code = "def broken_code(:\n    pass\n"
        assertions = ["assert True"]
        res = run_code_in_sandbox(code, assertions)
        self.assertEqual(res["passed_tests"], 0)
        self.assertIsNotNone(res["results"][0]["error"])

    def test_statistical_engine(self):
        df = generate_synthetic_evaluation_matrix(n_prompts=20, seed=42)
        res = analyze_quantization_impact(df)
        self.assertIn("friedman_chi2", res)
        self.assertIn("p_value", res)
        self.assertIn("kendall_w", res)
        self.assertIn("elbow_point", res)
        self.assertIn("posthoc_dunn_matrix", res)
        self.assertLessEqual(res["kendall_w"], 1.0)
        self.assertGreaterEqual(res["kendall_w"], 0.0)

    def test_recommendation_engine_optimal(self):
        stat_results = {
            "elbow_point": "Q5_K_M"
        }
        rec = recommend_model(stat_results, user_vram_gb=8.0)
        self.assertEqual(rec["status"], "OPTIMAL")
        self.assertEqual(rec["recommended_quantization"], "Q5_K_M")
        self.assertEqual(rec["required_vram_gb"], 5.0)

    def test_recommendation_engine_insufficient_vram(self):
        stat_results = {
            "elbow_point": "Q5_K_M"
        }
        rec = recommend_model(stat_results, user_vram_gb=4.0)
        self.assertEqual(rec["status"], "INSUFFICIENT_VRAM")
        self.assertIsNone(rec["recommended_quantization"])


    def test_humaneval_loader(self):
        from data.dataset_loader import load_humaneval_subset
        subset = load_humaneval_subset(limit=3)
        self.assertEqual(len(subset), 3)
        self.assertIn("task_id", subset[0])
        self.assertIn("prompt", subset[0])
        self.assertIn("entry_point", subset[0])
        self.assertIn("check(", subset[0]["test_code"])
        self.assertIn("test_assertions", subset[0])
        self.assertGreater(len(subset[0]["test_assertions"]), 1)

    def test_humaneval_plus_loader(self):
        from data.dataset_loader import load_humaneval_plus_subset
        subset = load_humaneval_plus_subset(limit=3)
        self.assertEqual(len(subset), 3)
        self.assertIn("task_id", subset[0])
        self.assertIn("prompt", subset[0])
        self.assertIn("entry_point", subset[0])
        self.assertIn("test_assertions", subset[0])

    def test_granular_scoring_partial(self):
        # Function that works for positive numbers but fails for negative numbers
        code = "def check_positive(x):\n    return x > 0\n"
        assertions = [
            "assert check_positive(5) == True",
            "assert check_positive(10) == True",
            "assert check_positive(1) == True",
            "assert check_positive(-2) == True"  # Intentional fail
        ]
        res = run_code_in_sandbox(code, assertions)
        self.assertEqual(res["total_tests"], 4)
        self.assertEqual(res["passed_tests"], 3)
        self.assertAlmostEqual(res["unit_test_pass_rate"], 0.75)

    def test_extract_assertions_context(self):
        from data.dataset_loader import extract_assertions_from_test_code
        sample_test = """
def check(candidate):
    base = 10
    assert candidate(2) == 12
    multiplier = 3
    assert candidate(5) == 25
"""
        extracted = extract_assertions_from_test_code(sample_test, "my_func")
        self.assertEqual(len(extracted), 2)
        self.assertIn("candidate = my_func", extracted[0])
        self.assertIn("base = 10", extracted[0])
        self.assertIn("assert candidate(2) == 12", extracted[0])
        self.assertIn("multiplier = 3", extracted[1])


if __name__ == "__main__":
    unittest.main()
