"""Evaluate MedRAX (+ optional playbook) on ChestAgentBench samples.

Writes one JSON line per question in the format read by ``experiments/compare_runs.py``
(``case_id``, ``question_id``, ``model``, ``model_answer``, ``correct_answer`` and
``input.question_data.metadata.categories``), so baseline and ACE runs can be compared with::

    python experiments/compare_runs.py results/baseline/predictions.jsonl results/ace/predictions.jsonl
"""

import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

from experiments.ace.utils import extract_answer

from .data import ChestAgentBenchProcessor, extract_letter


def _load_done_ids(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def summarize(predictions_path: str) -> Dict[str, Any]:
    """Overall and per-category accuracy from a predictions JSONL file."""
    total = correct = errors = no_answer = 0
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            correct += int(rec["is_correct"])
            errors += int(bool(rec.get("error")))
            no_answer += int(not rec["model_answer"])
            for cat in rec["input"]["question_data"]["metadata"]["categories"]:
                per_cat[cat]["total"] += 1
                per_cat[cat]["correct"] += int(rec["is_correct"])
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "agent_errors": errors,
        "no_answer": no_answer,
        "per_category": {
            c: {**v, "accuracy": v["correct"] / v["total"]} for c, v in sorted(per_cat.items())
        },
    }


def evaluate(
    generator,
    processor: ChestAgentBenchProcessor,
    samples: List[Dict[str, Any]],
    playbook: str,
    out_dir: str,
    run_label: str,
    workers: int = 4,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run ``generator`` on ``samples`` and write predictions.jsonl + summary.json to out_dir.

    With ``resume=True``, questions already present in predictions.jsonl are skipped, so an
    interrupted run can be restarted with the same command.
    """
    os.makedirs(out_dir, exist_ok=True)
    predictions_path = os.path.join(out_dir, "predictions.jsonl")
    with open(os.path.join(out_dir, "playbook_used.txt"), "w") as f:
        f.write(playbook or "")

    done = _load_done_ids(predictions_path) if resume else set()
    if not resume and os.path.exists(predictions_path):
        os.remove(predictions_path)
    todo = [(i, s) for i, s in enumerate(samples) if s["others"]["question_id"] not in done]
    print(
        f"Evaluating {len(todo)} questions ({len(done)} already done) with {workers} workers "
        f"-> {predictions_path}"
    )

    write_lock = threading.Lock()

    def run_one(i: int, sample: Dict[str, Any]) -> bool:
        response, bullet_ids, call_info = generator.generate(
            question=sample["question"],
            playbook=playbook,
            context=sample["context"],
            reflection="(empty)",
            call_id=f"{run_label}_{i}",
        )
        raw_answer = extract_answer(response)
        letter = extract_letter(raw_answer)
        is_correct = processor.answer_is_correct(letter, sample["target"])
        others = sample["others"]
        record = {
            "case_id": others["case_id"],
            "question_id": others["question_id"],
            "timestamp": datetime.now().isoformat(),
            "model": run_label,
            "agent_model": generator.model,
            "duration": round(call_info.get("total_time", 0.0), 2),
            "model_answer": letter,
            "raw_answer": raw_answer,
            "correct_answer": sample["target"],
            "is_correct": is_correct,
            "bullet_ids": bullet_ids,
            "num_tool_calls": call_info.get("num_tool_calls", 0),
            "error": call_info.get("error"),
            "option_permutation": others.get("option_permutation"),
            "input": {
                "question_data": {
                    "question": sample["question"],
                    "explanation": others.get("explanation", ""),
                    "metadata": {"categories": others.get("categories", [])},
                    "figures": json.loads(sample["context"]).get("image_paths", []),
                }
            },
        }
        with write_lock, open(predictions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return is_correct

    n_done = n_correct = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run_one, i, s) for i, s in todo]
        for fut in as_completed(futures):
            n_done += 1
            n_correct += int(fut.result())
            if n_done % 10 == 0 or n_done == len(todo):
                print(f"  {n_done}/{len(todo)} done, running accuracy {n_correct / n_done:.3f}")

    if not os.path.exists(predictions_path):
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    summary = summarize(predictions_path)
    summary.update({"run_label": run_label, "agent_model": generator.model})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Accuracy: {summary['accuracy']:.3f} ({summary['correct']}/{summary['total']})")
    for cat, stats in summary["per_category"].items():
        print(f"  {cat:<18} {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    return summary
