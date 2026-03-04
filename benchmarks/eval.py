"""
Benchmark evaluation for NovaMind-3B on coding and math tasks.

Supports:
  - HumanEval (code generation, pass@k)
  - MBPP (code generation, pass@k)
  - GSM8K (grade school math, accuracy)
  - MATH (competition math, accuracy)
"""
import os
import sys
import json
import re
import argparse
import signal
from io import StringIO
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import NovaMind3BConfig
from model.transformer import NovaMind3B
from tokenizer.tokenizer import get_tokenizer


# ============================================================
# Utility
# ============================================================
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    config = NovaMind3BConfig()
    config.mtp_depth = 0
    model = NovaMind3B(config)
    
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mtp_module")}
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device).eval()
    return model


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=512, temperature=0.0, top_p=0.95, device="cuda"):
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    generated = output_ids[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(generated)


# ============================================================
# HumanEval
# ============================================================
def eval_humaneval(model, tokenizer, data_dir, device="cuda", n_samples=1, temperature=0.2):
    """Evaluate on HumanEval benchmark."""
    humaneval_path = os.path.join(data_dir, "humaneval.jsonl")
    if not os.path.exists(humaneval_path):
        print(f"HumanEval data not found at {humaneval_path}")
        print("Run: python data/download.py --stage benchmark")
        return {}
    
    problems = []
    with open(humaneval_path) as f:
        for line in f:
            problems.append(json.loads(line))
    
    print(f"Evaluating HumanEval ({len(problems)} problems, {n_samples} samples each)...")
    
    results = []
    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]
        
        completions = []
        for s in range(n_samples):
            temp = temperature if n_samples > 1 else 0.0
            completion = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=512, temperature=temp,
                device=device,
            )
            # Stop at end markers
            for stop in ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint"]:
                if stop in completion:
                    completion = completion[:completion.index(stop)]
            completions.append(completion)
        
        # Test each completion
        passed = 0
        for comp in completions:
            full_code = prompt + comp + "\n" + test + f"\ncheck({entry_point})"
            try:
                with time_limit(10):
                    exec_globals = {}
                    exec(full_code, exec_globals)
                passed += 1
            except Exception:
                pass
        
        results.append({
            "task_id": task_id,
            "passed": passed,
            "total": n_samples,
        })
        
        if (i + 1) % 10 == 0:
            cur_pass = sum(r["passed"] > 0 for r in results) / len(results) * 100
            print(f"  {i+1}/{len(problems)}: pass@1 = {cur_pass:.1f}%")
    
    pass_at_1 = sum(r["passed"] > 0 for r in results) / len(results) * 100
    print(f"\nHumanEval Results: pass@1 = {pass_at_1:.1f}%")
    return {"humaneval_pass@1": pass_at_1}


# ============================================================
# MBPP
# ============================================================
def eval_mbpp(model, tokenizer, data_dir, device="cuda", n_samples=1, temperature=0.2):
    """Evaluate on MBPP benchmark."""
    mbpp_path = os.path.join(data_dir, "mbpp.jsonl")
    if not os.path.exists(mbpp_path):
        print(f"MBPP data not found at {mbpp_path}")
        return {}
    
    problems = []
    with open(mbpp_path) as f:
        for line in f:
            problems.append(json.loads(line))
    
    # Use test split (task_id 11-510)
    test_problems = [p for p in problems if 11 <= p.get("task_id", 0) <= 510]
    if not test_problems:
        test_problems = problems[:500]
    
    print(f"Evaluating MBPP ({len(test_problems)} problems)...")
    
    results = []
    for i, problem in enumerate(test_problems):
        prompt_text = problem["text"]
        test_cases = problem.get("test_list", [])
        
        # Format as few-shot prompt
        prompt = f"# {prompt_text}\ndef solution"
        
        completion = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=256, temperature=0.0,
            device=device,
        )
        
        for stop in ["\nclass ", "\n#", "\nif __name__"]:
            if stop in completion:
                completion = completion[:completion.index(stop)]
        
        full_code = prompt + completion
        
        passed = True
        for test in test_cases:
            try:
                with time_limit(5):
                    exec_globals = {}
                    exec(full_code + "\n" + test, exec_globals)
            except Exception:
                passed = False
                break
        
        results.append(passed)
        
        if (i + 1) % 50 == 0:
            acc = sum(results) / len(results) * 100
            print(f"  {i+1}/{len(test_problems)}: acc = {acc:.1f}%")
    
    accuracy = sum(results) / len(results) * 100
    print(f"\nMBPP Results: accuracy = {accuracy:.1f}%")
    return {"mbpp_accuracy": accuracy}


# ============================================================
# GSM8K
# ============================================================
def extract_number(text):
    """Extract the final number from a math solution."""
    # Look for #### pattern first
    match = re.search(r"####\s*([\-\d,.]+)", text)
    if match:
        num = match.group(1).replace(",", "")
        try:
            return float(num)
        except ValueError:
            pass
    
    # Look for "The answer is" pattern
    match = re.search(r"[Tt]he answer is\s*([\-\d,.]+)", text)
    if match:
        num = match.group(1).replace(",", "")
        try:
            return float(num)
        except ValueError:
            pass
    
    # Extract last number
    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def eval_gsm8k(model, tokenizer, data_dir, device="cuda"):
    """Evaluate on GSM8K benchmark."""
    gsm8k_path = os.path.join(data_dir, "gsm8k_test.jsonl")
    if not os.path.exists(gsm8k_path):
        print(f"GSM8K data not found at {gsm8k_path}")
        return {}
    
    problems = []
    with open(gsm8k_path) as f:
        for line in f:
            problems.append(json.loads(line))
    
    print(f"Evaluating GSM8K ({len(problems)} problems)...")
    
    correct = 0
    total = 0
    
    for i, problem in enumerate(problems):
        question = problem["question"]
        answer_text = problem["answer"]
        
        # Extract gold answer
        gold = extract_number(answer_text)
        if gold is None:
            continue
        
        # Generate solution using chain-of-thought
        prompt = (
            f"Question: {question}\n"
            f"Let's solve this step by step.\n"
        )
        
        response = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=512, temperature=0.0,
            device=device,
        )
        
        pred = extract_number(response)
        
        if pred is not None and abs(pred - gold) < 1e-3:
            correct += 1
        total += 1
        
        if (i + 1) % 100 == 0:
            acc = correct / total * 100
            print(f"  {i+1}/{len(problems)}: accuracy = {acc:.1f}% ({correct}/{total})")
    
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\nGSM8K Results: accuracy = {accuracy:.1f}% ({correct}/{total})")
    return {"gsm8k_accuracy": accuracy}


# ============================================================
# MATH
# ============================================================
def normalize_math_answer(answer):
    """Normalize math answers for comparison."""
    answer = str(answer).strip()
    # Remove LaTeX formatting
    answer = answer.replace("\\$", "")
    answer = re.sub(r"\\text\{.*?\}", "", answer)
    answer = re.sub(r"\\(?:frac|dfrac)\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\", "")
    answer = answer.strip()
    return answer


def eval_math(model, tokenizer, data_dir, device="cuda"):
    """Evaluate on MATH benchmark."""
    math_path = os.path.join(data_dir, "math_test.jsonl")
    if not os.path.exists(math_path):
        print(f"MATH data not found at {math_path}")
        return {}
    
    problems = []
    with open(math_path) as f:
        for line in f:
            problems.append(json.loads(line))
    
    print(f"Evaluating MATH ({len(problems)} problems)...")
    
    correct = 0
    total = 0
    
    for i, problem in enumerate(problems):
        question = problem["problem"]
        answer = problem["solution"]
        
        # Extract boxed answer
        boxed = re.search(r"\\boxed\{(.*?)\}", answer)
        gold = boxed.group(1) if boxed else answer.split("=")[-1].strip()
        gold_norm = normalize_math_answer(gold)
        
        prompt = (
            f"Problem: {question}\n"
            f"Solution: Let me solve this step by step.\n"
        )
        
        response = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=512, temperature=0.0,
            device=device,
        )
        
        pred_norm = normalize_math_answer(response.split("=")[-1].strip() if "=" in response else response.strip())
        
        if gold_norm and pred_norm and gold_norm == pred_norm:
            correct += 1
        total += 1
        
        if (i + 1) % 100 == 0:
            acc = correct / total * 100
            print(f"  {i+1}/{len(problems)}: accuracy = {acc:.1f}% ({correct}/{total})")
    
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\nMATH Results: accuracy = {accuracy:.1f}% ({correct}/{total})")
    return {"math_accuracy": accuracy}


# ============================================================
# Main
# ============================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("NovaMind-3B Benchmark Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    tokenizer = get_tokenizer()
    
    data_dir = args.data_dir or "/mnt/zone/A/datasets/benchmarks"
    benchmarks = args.benchmarks.split(",") if args.benchmarks else ["humaneval", "mbpp", "gsm8k", "math"]
    
    all_results = {}
    
    for bench in benchmarks:
        bench = bench.strip().lower()
        print(f"\n{'='*40}")
        print(f"Running: {bench}")
        print(f"{'='*40}\n")
        
        if bench == "humaneval":
            results = eval_humaneval(model, tokenizer, data_dir, device=device)
        elif bench == "mbpp":
            results = eval_mbpp(model, tokenizer, data_dir, device=device)
        elif bench == "gsm8k":
            results = eval_gsm8k(model, tokenizer, data_dir, device=device)
        elif bench == "math":
            results = eval_math(model, tokenizer, data_dir, device=device)
        else:
            print(f"Unknown benchmark: {bench}")
            continue
        
        all_results.update(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for k, v in all_results.items():
        print(f"  {k}: {v:.1f}%")
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NovaMind-3B Benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--benchmarks", type=str, default="humaneval,mbpp,gsm8k,math")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/benchmark_results.json")
    args = parser.parse_args()
    
    main(args)
