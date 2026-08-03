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
            "canonical_solution": data["canonical_solution"]
        })
        
    return dataset

if __name__ == "__main__":
    data = load_humaneval_subset(5)
    print(f"✓ Successfully loaded {len(data)} sample problems from HumanEval.")
    print(f"Sample Task ID: {data[0]['task_id']}")
