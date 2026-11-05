"""Run ACE (Agentic Context Engineering) experiments with the MedRAX agent on ChestAgentBench.

Run from the MedRAX repository root:

    # 1. Baseline: MedRAX with no playbook on the held-out test split
    python -m experiments.ace_medrax.run --mode eval --label medrax-baseline

    # 2. Train a playbook (offline ACE on the train split, validated on the val split),
    #    then evaluate the best playbook on the test split
    python -m experiments.ace_medrax.run --mode offline --eval-test

    # 3. Evaluate any playbook on the test split
    python -m experiments.ace_medrax.run --mode eval --playbook <run>/ace_run_*/best_playbook.txt

    # Smoke test: 2 questions per split
    python -m experiments.ace_medrax.run --mode offline --max-samples 2 --eval-test

See experiments/ace_medrax/README.md for details.
"""

import argparse
import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List

from .data import (
    ChestAgentBenchProcessor,
    load_metadata,
    load_or_create_splits,
    rows_for_cases,
)

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default.json")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str) -> Dict[str, Any]:
    """Load default.json, then overlay ``path`` if it is a different file.

    Experiment configs therefore only need the keys they change.
    """
    with open(DEFAULT_CONFIG, "r") as f:
        config = json.load(f)
    if path and os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG):
        with open(path, "r") as f:
            deep_update(config, json.load(f))
    return config


def parse_set_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Turn ``--set ace.max_num_rounds=0 data.shuffle_answer_options=true`` into a nested dict."""
    overrides: Dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


# ---------------------------------------------------------------------------
# Building the MedRAX generator
# ---------------------------------------------------------------------------


def resolve_model_dir(medrax_cfg: Dict[str, Any]) -> str:
    return (
        medrax_cfg.get("model_dir")
        or os.getenv("MEDRAX_MODEL_DIR")
        or os.path.expanduser("~/.cache/huggingface/hub")
    )


def build_generator(config: Dict[str, Any], run_log_path: str):
    # Imported lazily so `--help` and config errors don't pay for loading torch/gradio.
    from langchain_openai import ChatOpenAI

    from main import build_tools
    from medrax.utils import load_prompts_from_file

    from .generator import MedRAXGenerator, SerializedToolAgent

    m = config["medrax"]
    model_dir = resolve_model_dir(m)
    print(f"Loading MedRAX tools {m['tools']} (weights in {model_dir}, device {m['device']})")
    tools = build_tools(m["tools"], model_dir=model_dir, temp_dir=m["temp_dir"], device=m["device"])

    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url
    llm = ChatOpenAI(
        model=m["model"], temperature=m["temperature"], top_p=m["top_p"], **openai_kwargs
    )

    # The system prompt (base + playbook) is passed per call by MedRAXGenerator, so the agent
    # itself gets an empty one and no checkpointer (each question starts from a clean state).
    agent = SerializedToolAgent(
        llm,
        tools=list(tools.values()),
        system_prompt="",
        log_tools=False,
        checkpointer=None,
        tool_lock=threading.Lock(),
    )
    base_prompt = load_prompts_from_file("medrax/docs/system_prompts.txt")["MEDICAL_ASSISTANT"]
    return MedRAXGenerator(
        agent,
        base_system_prompt=base_prompt,
        model_name=m["model"],
        max_agent_steps=m["max_agent_steps"],
        max_tool_output_chars=m["max_tool_output_chars"],
        send_images_to_llm=m["send_images_to_llm"],
        run_log_path=run_log_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


def main():
    parser = argparse.ArgumentParser(
        description="ACE playbook learning for MedRAX on ChestAgentBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["eval", "offline", "online"], required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Experiment config (JSON)")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set ace.max_num_rounds=0 medrax.device=cuda:1",
    )
    parser.add_argument("--playbook", help="Playbook file (eval: playbook to test; offline/online: "
                        "starting playbook, overrides ace.initial_playbook)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Split to evaluate in --mode eval (default: test)")
    parser.add_argument("--eval-test", action="store_true",
                        help="offline mode: evaluate the best playbook on the test split afterwards")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Use only the first N questions of each split (smoke tests)")
    parser.add_argument("--label", help="Run name (used for the output folder and as the "
                        "'model' field read by compare_runs.py)")
    parser.add_argument("--out-dir", help="Output folder; reuse an existing one to resume an "
                        "interrupted --mode eval run")
    args = parser.parse_args()

    config = deep_update(load_config(args.config), parse_set_overrides(args.set))

    # -- data -----------------------------------------------------------------
    rows = load_metadata(config["data_dir"])
    s = config["split"]
    splits = load_or_create_splits(
        config["split_file"], rows, s["n_train_cases"], s["n_val_cases"], s["seed"]
    )
    processor = ChestAgentBenchProcessor(config["data_dir"], **config["data"])

    def samples_for(split_name: str):
        samples = processor.process_task_data(rows_for_cases(rows, splits[split_name]))
        return samples[: args.max_samples] if args.max_samples else samples

    # -- output folder ----------------------------------------------------------
    if args.playbook and args.mode == "eval" and not args.label:
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.playbook)))
        label = f"medrax+playbook-{parent}"
    else:
        label = args.label or ("medrax" if args.mode == "eval" else f"ace-{args.mode}")
    label = slugify(label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(config["save_dir"], f"{timestamp}_{label}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "label": label, "config": config}, f, indent=2)
    print(f"Output folder: {out_dir}")

    generator = build_generator(config, run_log_path=os.path.join(out_dir, "agent_runs.jsonl"))

    from .evaluate import evaluate

    # -- eval ---------------------------------------------------------------------
    if args.mode == "eval":
        playbook = ""
        if args.playbook:
            with open(args.playbook, "r") as f:
                playbook = f.read()
        evaluate(
            generator,
            processor,
            samples_for(args.split),
            playbook,
            out_dir=out_dir,
            run_label=label,
            workers=config["ace"]["test_workers"],
        )
        return

    # -- ACE training ---------------------------------------------------------------
    from experiments.ace import ACE

    a = config["ace"]
    initial_playbook_path = args.playbook or a.get("initial_playbook")
    initial_playbook = None
    if initial_playbook_path:
        with open(initial_playbook_path, "r") as f:
            initial_playbook = f.read()

    ace_system = ACE(
        api_provider=a["api_provider"],
        generator_model=config["medrax"]["model"],
        reflector_model=a["reflector_model"],
        curator_model=a["curator_model"],
        max_tokens=a["max_tokens"],
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=a["use_bulletpoint_analyzer"],
        bulletpoint_analyzer_threshold=a["bulletpoint_analyzer_threshold"],
    )
    ace_system.generator = generator  # swap ACE's text-only generator for the MedRAX agent

    ace_config = {
        key: a[key]
        for key in [
            "num_epochs", "max_num_rounds", "curator_frequency", "eval_steps", "save_steps",
            "online_eval_frequency", "playbook_token_budget", "json_mode", "no_ground_truth", "test_workers",
            "use_bulletpoint_analyzer", "bulletpoint_analyzer_threshold",
            "skip_post_curate_generation", "api_provider",
        ]
    }
    ace_config.update({"task_name": "chestagentbench", "save_dir": out_dir})

    if args.mode == "offline":
        train, val = samples_for("train"), samples_for("val")
        if ace_config["eval_steps"] > len(train):
            print(f"eval_steps ({ace_config['eval_steps']}) > train size ({len(train)}); "
                  f"validating once at the end instead")
            ace_config["eval_steps"] = len(train)
        elif len(train) % ace_config["eval_steps"]:
            print(f"Warning: the last {len(train) % ace_config['eval_steps']} training steps "
                  f"come after the final validation, so their updates can only appear in "
                  f"final_playbook.txt, not best_playbook.txt")
        ace_system.run(
            mode="offline",
            train_samples=train,
            val_samples=val,
            test_samples=None,  # test is evaluated below with per-question logs
            data_processor=processor,
            config=ace_config,
        )
        best_path = os.path.join(out_dir, "best_playbook.txt")
        with open(best_path, "w") as f:
            f.write(ace_system.best_playbook)
        print(f"Best playbook: {best_path}")

        if args.eval_test:
            evaluate(
                generator,
                processor,
                samples_for("test"),
                ace_system.best_playbook,
                out_dir=os.path.join(out_dir, "test_eval"),
                run_label=f"{label}-best-playbook",
                workers=a["test_workers"],
            )

    else:  # online: ACE evaluates the test split before adapting, then adapts while testing
        ace_system.run(
            mode="online",
            test_samples=samples_for("test"),
            data_processor=processor,
            config=ace_config,
        )
        with open(os.path.join(out_dir, "final_playbook.txt"), "w") as f:
            f.write(ace_system.playbook)


if __name__ == "__main__":
    main()
