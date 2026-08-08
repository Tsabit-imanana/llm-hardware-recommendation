import os
# pyrefly: ignore [missing-import]
from human_eval.data import read_problems

def load_humaneval_subset(limit: int = 20) -> list[dict]:
    """
    Loads HumanEval dataset problems.
    Each item contains task_id, prompt, entry_point, canonical_solution, and test_code assertions.
    """
    dataset_path = "./data/HumanEval.jsonl.gz"
    if os.path.exists(dataset_path):
        problems = read_problems(evalset_file=dataset_path)
    else:
        problems = read_problems()
    dataset = []
    
    for task_id, data in list(problems.items())[:limit]:
        dataset.append({
            "task_id": task_id,
            "prompt": data["prompt"],
            "entry_point": data["entry_point"],
            "test_code": data["test"],
            "canonical_solution": data["canonical_solution"],
            "base_input": [],
            "plus_input": []
        })
        
    return dataset

def load_humaneval_plus_subset(limit: int = 20) -> list[dict]:
    """
    Loads HumanEvalPlus dataset problems via evalplus.
    Each item contains task_id, prompt, entry_point, canonical_solution, test_code, base_input, plus_input.
    """
    try:
        # pyrefly: ignore [missing-import]
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    except Exception as e:
        print(f"Warning: Could not load evalplus ({e}), falling back to HumanEval dataset.")
        return load_humaneval_subset(limit=limit)

    dataset = []
    for task_id, data in list(problems.items())[:limit]:
        entry_point = data["entry_point"]
        test_code = data.get("test", "")
        
        base_inputs = data.get("base_input", [])
        plus_inputs = data.get("plus_input", [])
        canonical = data.get("canonical_solution", "")
        
        assertions_code = test_code
        if (base_inputs or plus_inputs) and canonical:
            eval_harness = f"\n\n# --- HumanEvalPlus Verification Harness ---\n"
            eval_harness += f"# Canonical reference implementation\n"
            eval_harness += f"def __ref_{entry_point}(*args):\n"
            # Indent canonical lines under __ref_ entry_point
            indented_canonical = "\n".join("    " + line for line in canonical.splitlines())
            eval_harness += indented_canonical + f"\n    return {entry_point}(*args)\n\n"
            
            all_inputs = base_inputs + plus_inputs
            eval_harness += f"eval_inputs = {all_inputs!r}\n"
            eval_harness += f"for args in eval_inputs:\n"
            eval_harness += f"    expected = __ref_{entry_point}(*args)\n"
            eval_harness += f"    actual = {entry_point}(*args)\n"
            eval_harness += f"    assert actual == expected, f'EvalPlus failed on input {{args}}: expected {{expected}}, got {{actual}}'\n"
            
            assertions_code += eval_harness

        dataset.append({
            "task_id": task_id,
            "prompt": data["prompt"],
            "entry_point": entry_point,
            "test_code": assertions_code,
            "canonical_solution": canonical,
            "base_input": base_inputs,
            "plus_input": plus_inputs
        })
        
    return dataset

if __name__ == "__main__":
    data = load_humaneval_plus_subset(5)
    print(f"✓ Successfully loaded {len(data)} sample problems from HumanEvalPlus.")
    print(f"Sample Task ID: {data[0]['task_id']}")
    print(f"Base inputs: {len(data[0]['base_input'])}, Plus inputs: {len(data[0]['plus_input'])}")
