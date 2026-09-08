import os
import ast
# pyrefly: ignore [missing-import]
from human_eval.data import read_problems

def extract_assertions_from_test_code(test_code: str, entry_point: str) -> list[str]:
    """
    Extracts individual assertions from HumanEval test code so each assertion can be
    evaluated independently in the sandbox, enabling granular per-assertion scoring.
    """
    try:
        tree = ast.parse(test_code)
    except Exception:
        fallback = test_code
        if f"check({entry_point})" not in fallback:
            fallback = f"{fallback}\ncheck({entry_point})\n"
        return [fallback]

    outside_stmts = []
    check_func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_func = node
        else:
            outside_stmts.append(ast.unparse(node))

    if not check_func:
        fallback = test_code
        if f"check({entry_point})" not in fallback:
            fallback = f"{fallback}\ncheck({entry_point})\n"
        return [fallback]

    cand_name = check_func.args.args[0].arg if check_func.args.args else "candidate"
    setup_header = "\n".join(outside_stmts).strip()
    if setup_header:
        setup_header += "\n"
    setup_header += f"{cand_name} = {entry_point}\n"

    assertions = []
    current_context = []
    valid_context_types = (
        ast.Assign,
        ast.AugAssign,
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.ClassDef,
        ast.AnnAssign,
    )

    for stmt in check_func.body:
        if isinstance(stmt, (ast.Assert, ast.For)):
            stmt_code = ast.unparse(stmt)
            if current_context:
                full_assert = setup_header + "\n".join(current_context) + "\n" + stmt_code
            else:
                full_assert = setup_header + stmt_code
            assertions.append(full_assert)
        elif isinstance(stmt, valid_context_types):
            current_context.append(ast.unparse(stmt))

    if not assertions:
        fallback = test_code
        if f"check({entry_point})" not in fallback:
            fallback = f"{fallback}\ncheck({entry_point})\n"
        return [fallback]

    return assertions

def load_humaneval_subset(limit: int = 20) -> list[dict]:
    """
    Loads HumanEval dataset problems.
    Each item contains task_id, prompt, entry_point, canonical_solution, test_code,
    and individual test_assertions for granular scoring.
    """
    dataset_path = "./data/HumanEval.jsonl.gz"
    if os.path.exists(dataset_path):
        problems = read_problems(evalset_file=dataset_path)
    else:
        problems = read_problems()
    dataset = []
    
    for task_id, data in list(problems.items())[:limit]:
        entry_point = data["entry_point"]
        test_code = data["test"]
        if f"check({entry_point})" not in test_code:
            test_code = f"{test_code}\ncheck({entry_point})\n"
        test_assertions = extract_assertions_from_test_code(data["test"], entry_point)
        dataset.append({
            "task_id": task_id,
            "prompt": data["prompt"],
            "entry_point": entry_point,
            "test_code": test_code,
            "test_assertions": test_assertions,
            "canonical_solution": data["canonical_solution"],
            "base_input": [],
            "plus_input": []
        })
        
    return dataset

def load_humaneval_plus_subset(limit: int = 20) -> list[dict]:
    """
    Loads HumanEvalPlus dataset problems via evalplus.
    Each item contains task_id, prompt, entry_point, canonical_solution, test_code,
    test_assertions, base_input, plus_input.
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

        test_assertions = extract_assertions_from_test_code(assertions_code, entry_point)

        dataset.append({
            "task_id": task_id,
            "prompt": data["prompt"],
            "entry_point": entry_point,
            "test_code": assertions_code,
            "test_assertions": test_assertions,
            "canonical_solution": canonical,
            "base_input": base_inputs,
            "plus_input": plus_inputs
        })
        
    return dataset

if __name__ == "__main__":
    data = load_humaneval_subset(5)
    print(f"✓ Successfully loaded {len(data)} sample problems from standard HumanEval.")
    print(f"Sample Task ID: {data[0]['task_id']}")
    print(f"Entry point: {data[0]['entry_point']}")
