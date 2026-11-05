"""Unit tests for the ACE <-> MedRAX integration (experiments/ace, experiments/ace_medrax).

No GPU, model weights or API calls are needed: the MedRAX agent is replaced by fakes.
Note: importing experiments.ace loads tiktoken's cl100k_base encoding, which tiktoken
downloads once if it is not cached yet.

Run from the repository root:  pytest tests/test_ace_medrax.py
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from experiments.ace_medrax.data import (
    ChestAgentBenchProcessor,
    extract_letter,
    load_or_create_splits,
    make_case_splits,
    rows_for_cases,
    shuffle_options,
)
from experiments.ace_medrax.generator import (
    MedRAXGenerator,
    format_trace,
    parse_final_json,
)
from medrax.utils import PLAYBOOK_HEADER, add_playbook_to_prompt

QUESTION = (
    "Which finding is present?\n"
    "A) Pneumothorax\n"
    "B) Pleural effusion\n"
    "C) Cardiomegaly\n"
    "D) Normal"
)

PLAYBOOK = """## TOOL SELECTION AND TRUST
[tsat-00001] helpful=2 harmful=0 :: Cross-check classifier effusion scores with segmentation.

## OTHERS"""


def make_rows(n_cases=10, per_case=3):
    rows = []
    for c in range(n_cases):
        for q in range(per_case):
            rows.append(
                {
                    "case_id": str(1000 + c),
                    "question_id": f"{1000 + c}_{q}",
                    "question": QUESTION,
                    "answer": "B",
                    "explanation": "Blunted costophrenic angle indicates effusion.",
                    "images": [f"figures/{1000 + c}/figure_1.jpg"],
                    "image_source_urls": [],
                    "categories": "detection,reasoning",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B", "B"),
        ("b", "B"),
        ("B)", "B"),
        ("(C) Cardiomegaly", "C"),
        ("D. Normal", "D"),
        ("The answer is E.", "E"),
        ("Answer: F", "F"),
        ("I pick (A) because", "A"),
        ("A patient with effusion", ""),  # prose, not a choice
        ("the answer is a lesion", ""),
        ("", ""),
        (None, ""),
        ("No final answer found", ""),
    ],
)
def test_extract_letter(text, expected):
    assert extract_letter(text) == expected


def test_case_splits_are_disjoint_and_deterministic():
    rows = make_rows(n_cases=20)
    s1 = make_case_splits(rows, n_train_cases=8, n_val_cases=4, seed=1)
    s2 = make_case_splits(rows, n_train_cases=8, n_val_cases=4, seed=1)
    assert s1 == s2
    train, val, test = set(s1["train"]), set(s1["val"]), set(s1["test"])
    assert len(train) == 8 and len(val) == 4 and len(test) == 8
    assert not (train & val) and not (train & test) and not (val & test)
    # Every question of a case lands in the same split
    assert {r["case_id"] for r in rows_for_cases(rows, s1["train"])} == train


def test_case_splits_reject_oversized_request():
    with pytest.raises(ValueError):
        make_case_splits(make_rows(n_cases=5), n_train_cases=4, n_val_cases=1)


def test_split_file_is_frozen(tmp_path):
    rows = make_rows(n_cases=20)
    path = tmp_path / "splits.json"
    first = load_or_create_splits(str(path), rows, 8, 4, seed=1)
    # Different arguments must not change an existing split file
    second = load_or_create_splits(str(path), rows, 2, 2, seed=99)
    assert first == second


def test_shuffle_options_keeps_answer_text():
    new_q, new_answer, perm = shuffle_options(QUESTION, "B", seed_key="q1")
    assert perm is not None and sorted(perm) == [0, 1, 2, 3]
    option = {l[0]: l[3:] for l in new_q.split("\n") if len(l) > 2 and l[1] == ")"}
    assert option[new_answer] == "Pleural effusion"
    assert shuffle_options(QUESTION, "B", seed_key="q1") == (new_q, new_answer, perm)
    assert new_q.split("\n")[0] == "Which finding is present?"


def test_shuffle_options_leaves_unparseable_questions_alone():
    q = "Free text question without options"
    assert shuffle_options(q, "B", seed_key="x") == (q, "B", None)


def test_processor_samples_and_feedback(tmp_path):
    rows = make_rows(n_cases=1, per_case=1) + [
        {**make_rows(1, 1)[0], "question_id": "noimg", "images": []}
    ]
    proc = ChestAgentBenchProcessor(str(tmp_path))
    samples = proc.process_task_data(rows)
    assert len(samples) == 1  # the question without images is skipped
    sample = samples[0]
    ctx = json.loads(sample["context"])
    assert ctx["image_paths"] == [str(tmp_path / "figures/1000/figure_1.jpg")]
    assert "answer" not in ctx  # the answer never leaks into what the agent/curator sees
    assert sample["target"] == "B"
    assert sample["others"]["categories"] == ["detection", "reasoning"]

    assert proc.answer_is_correct("B", "B") and not proc.answer_is_correct("C", "B")
    assert proc.evaluate_accuracy(["B", "C"], ["B", "B"]) == 0.5

    fb = proc.environment_feedback(sample, False, "Predicted answer does not match ground truth")
    assert "Blunted costophrenic angle" in fb
    proc_no_expl = ChestAgentBenchProcessor(str(tmp_path), use_explanation_feedback=False)
    assert proc_no_expl.environment_feedback(sample, False, "default") == "default"


# ---------------------------------------------------------------------------
# playbook prompt (shared by main.py and training)
# ---------------------------------------------------------------------------


def test_add_playbook_to_prompt():
    assert add_playbook_to_prompt("base", None) == "base"
    assert add_playbook_to_prompt("base", "   \n") == "base"
    assert add_playbook_to_prompt("base", PLAYBOOK) == "base" + PLAYBOOK_HEADER + PLAYBOOK


# ---------------------------------------------------------------------------
# generator.py
# ---------------------------------------------------------------------------


class FakeAgent:
    """Mimics agent.workflow.invoke: appends scripted AI/tool messages to the input."""

    def __init__(self, scripted_turns, raise_exc=None):
        self.turns = list(scripted_turns)
        self.raise_exc = raise_exc
        self.inputs = []
        self.workflow = SimpleNamespace(invoke=self._invoke)

    def _invoke(self, state, config):
        self.inputs.append(state["messages"])
        if self.raise_exc:
            raise self.raise_exc
        return {"messages": state["messages"] + self.turns.pop(0)}


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "xray.jpg"
    path.write_bytes(b"\xff\xd8\xff fake jpeg")
    return str(path)


def context_for(image_path):
    return json.dumps({"case_id": "1", "question_id": "1_1", "question": QUESTION,
                       "image_paths": [image_path]})


def tool_turn():
    return [
        AIMessage(content="", tool_calls=[{"name": "classifier", "args": {"image_path": "x"},
                                           "id": "t1"}]),
        ToolMessage(content="{'Effusion': 0.91}", tool_call_id="t1", name="classifier"),
    ]


def test_generator_happy_path(image, tmp_path):
    final = AIMessage(content='Effusion is visible.\n{"final_answer": "B", '
                              '"bullet_ids": ["tsat-00001", "made-up-00009"]}')
    agent = FakeAgent([tool_turn() + [final]])
    log = tmp_path / "runs.jsonl"
    gen = MedRAXGenerator(agent, "BASE PROMPT", "gpt-4o", run_log_path=str(log))

    response, bullet_ids, info = gen.generate(QUESTION, PLAYBOOK, context_for(image),
                                              call_id="t")

    # The playbook goes into the per-call system message; images go to the LLM as data URLs
    sent = agent.inputs[0]
    assert isinstance(sent[0], SystemMessage) and "tsat-00001" in sent[0].content
    assert isinstance(sent[1], HumanMessage)
    assert sent[1].content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image in sent[1].content[0]["text"]

    assert bullet_ids == ["tsat-00001"]  # hallucinated IDs are dropped
    assert info["num_tool_calls"] == 1 and info["error"] is None
    assert "TOOL CALL classifier" in response and "TOOL RESULT classifier" in response

    # ACE's own answer extraction must find the final answer in the response
    from experiments.ace.utils import extract_answer
    assert extract_answer(response) == "B"

    record = json.loads(log.read_text().strip())
    assert record["question_id"] == "1_1" and record["final_answer"] == "B"
    assert record["tools_used"] == ["classifier"]


def test_generator_reflection_is_included(image):
    agent = FakeAgent([[AIMessage(content='{"final_answer": "C", "bullet_ids": []}')]])
    gen = MedRAXGenerator(agent, "BASE", "gpt-4o")
    gen.generate(QUESTION, PLAYBOOK, context_for(image), reflection="You ignored the angle.")
    assert "You ignored the angle." in agent.inputs[0][1].content[0]["text"]


def test_generator_followup_when_no_json(image):
    agent = FakeAgent([
        [AIMessage(content="I think it is probably an effusion.")],
        [AIMessage(content='{"final_answer": "B", "bullet_ids": []}')],
    ])
    gen = MedRAXGenerator(agent, "BASE", "gpt-4o")
    response, _, _ = gen.generate(QUESTION, "", context_for(image))
    assert len(agent.inputs) == 2  # one follow-up turn
    from experiments.ace.utils import extract_answer
    assert extract_answer(response) == "B"


def test_generator_records_errors(image):
    agent = FakeAgent([], raise_exc=RuntimeError("CUDA out of memory"))
    gen = MedRAXGenerator(agent, "BASE", "gpt-4o")
    response, bullet_ids, info = gen.generate(QUESTION, PLAYBOOK, context_for(image))
    assert "CUDA out of memory" in info["error"]
    assert "AGENT ERROR" in response and bullet_ids == []
    from experiments.ace.utils import extract_answer
    assert extract_letter(extract_answer(response)) == ""


def test_parse_final_json_takes_last_object():
    text = 'tool said {"final_answer": "A"} but\n{"final_answer": "D", "bullet_ids": []}'
    assert parse_final_json(text)["final_answer"] == "D"
    assert parse_final_json("no json here") is None


def test_format_trace_truncates_tool_output():
    msgs = [ToolMessage(content="x" * 100, tool_call_id="1", name="seg")]
    assert "truncated 90 chars" in format_trace(msgs, max_tool_output_chars=10)


# ---------------------------------------------------------------------------
# evaluate.py
# ---------------------------------------------------------------------------


class ScriptedGenerator:
    model = "fake"

    def __init__(self, answers):
        self.answers = answers

    def generate(self, question, playbook, context, reflection="(empty)", call_id="", **_):
        qid = json.loads(context)["question_id"]
        answer = self.answers[qid]
        return (f'trace\n\nFINAL: {{"final_answer": "{answer}", "bullet_ids": []}}', [],
                {"total_time": 0.1, "num_tool_calls": 2, "error": None})


def test_evaluate_writes_compare_runs_format_and_resumes(tmp_path):
    from experiments.ace_medrax.evaluate import evaluate

    proc = ChestAgentBenchProcessor(str(tmp_path))
    samples = proc.process_task_data(make_rows(n_cases=2, per_case=2))
    answers = {s["others"]["question_id"]: ("B" if i % 2 == 0 else "C")
               for i, s in enumerate(samples)}
    out = tmp_path / "eval"
    summary = evaluate(ScriptedGenerator(answers), proc, samples, PLAYBOOK, str(out),
                       run_label="test-run", workers=2)
    assert summary["total"] == 4 and summary["correct"] == 2
    assert summary["per_category"]["detection"]["accuracy"] == 0.5

    lines = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    for rec in lines:  # keys experiments/compare_runs.py relies on
        for key in ["case_id", "question_id", "model", "model_answer", "correct_answer"]:
            assert key in rec
        assert rec["input"]["question_data"]["metadata"]["categories"]
    assert (out / "playbook_used.txt").read_text() == PLAYBOOK

    # Resuming skips finished questions
    class Exploding(ScriptedGenerator):
        def generate(self, *a, **k):
            raise AssertionError("should not be called")

    summary2 = evaluate(Exploding(answers), proc, samples, PLAYBOOK, str(out),
                        run_label="test-run", workers=2)
    assert summary2["total"] == 4


# ---------------------------------------------------------------------------
# vendored ACE: modifications documented in experiments/ace/NOTICE.md
# ---------------------------------------------------------------------------


def test_ace_training_step_uses_feedback_hook_and_skips_post_curate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # client construction only; no calls made
    from experiments.ace import ACE

    ace = ACE(api_provider="openai", generator_model="fake", reflector_model="fake",
              curator_model="fake", initial_playbook=PLAYBOOK)

    gen_calls = []

    class WrongGenerator:
        model = "fake"

        def generate(self, **kwargs):
            gen_calls.append(kwargs)
            return 'trace\n\nFINAL: {"final_answer": "C", "bullet_ids": ["tsat-00001"]}', \
                ["tsat-00001"], {}

    feedback_seen = []

    class FakeReflector:
        model = "fake"

        def reflect(self, **kwargs):
            feedback_seen.append(kwargs["environment_feedback"])
            return "reflection text", [{"id": "tsat-00001", "tag": "harmful"}], {}

    class FakeCurator:
        model = "fake"

        def curate(self, current_playbook, next_global_id, **kwargs):
            return current_playbook, next_global_id, [], {}

    ace.generator, ace.reflector, ace.curator = WrongGenerator(), FakeReflector(), FakeCurator()

    proc = ChestAgentBenchProcessor(str(tmp_path))
    sample = proc.process_task_data(make_rows(n_cases=1, per_case=1))[0]
    params = ace._extract_config_params({"max_num_rounds": 1, "skip_post_curate_generation": True})

    pre, post, tracking = ace._train_single_sample(
        task_dict=sample, data_processor=proc, step_id="train_e_1_s_1", epoch=1, step=1,
        usage_log_path=str(tmp_path / "usage.jsonl"), log_dir=str(tmp_path),
        config_params=params, total_samples=1,
    )

    assert len(gen_calls) == 2  # initial + 1 reflection round, no post-curate run
    assert tracking["post_train_result"]["post_curate_generation_skipped"] is True
    assert "Blunted costophrenic angle" in feedback_seen[0]  # explanation reached the Reflector
    assert "harmful=1" in ace.playbook  # bullet counters were updated
