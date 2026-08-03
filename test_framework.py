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


if __name__ == "__main__":
    unittest.main()
