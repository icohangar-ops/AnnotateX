"""CPU-only behavioral tests for the AnnotateX ICL engine.

No model download, no GPU: every external interaction (tokenizer, model,
quantization config) is stubbed or recorded. These tests are the evidence-matrix
targets referenced by ``evidence/matrix.yaml``; they pin the pipeline claims the
README states — few-shot selection, answer-only prompting, optional
self-consistency voting, truncation, extraction fallbacks, and the
cubiczan_resilience retry/atomic-sidecar wiring.
"""

import inspect
import json
import random
from pathlib import Path

import torch

import icl_annotation_solver as mod


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakeBatch(dict):
    """Mimics a HF BatchEncoding: dict-unpackable, attribute access, ``.to()``."""

    def __init__(self, n_tokens: int):
        super().__init__(input_ids=torch.ones(1, n_tokens, dtype=torch.long))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def to(self, device):
        return self


class FakeTokenizer:
    """Records the kwargs the engine passes when tokenizing a prompt."""

    def __init__(self, n_tokens: int = 7):
        self.n_tokens = n_tokens
        self.pad_token = "pad"
        self.eos_token = "eos"
        self.eos_token_id = 151643
        self.calls = []

    def __call__(self, prompt, return_tensors=None, **kwargs):
        self.calls.append(kwargs)
        return FakeBatch(self.n_tokens)

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded answer"


class ChatTokenizer(FakeTokenizer):
    """FakeTokenizer plus a working chat template (same shape as the engine's
    manual fallback template)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class FakeModel:
    """Returns prompt_len + n_new tokens so generate() slices the new span."""

    def __init__(self, n_new: int = 3):
        self.device = torch.device("cpu")
        self.n_new = n_new

    def eval(self):
        return self

    def generate(self, input_ids=None, **kwargs):
        new = torch.full((1, self.n_new), 2, dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


class RecordingBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class RecordingAutoTokenizer:
    last_init_kwargs = None

    def __init__(self):
        self.pad_token = "pad"
        self.eos_token = "eos"

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.last_init_kwargs = kwargs
        return cls()


class RecordingAutoModel:
    last_kwargs = None

    def __init__(self):
        self.device = torch.device("cpu")

    def eval(self):
        return self

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.last_kwargs = kwargs
        return cls()


class FakeLoader:
    """Duck-typed stand-in for TaskLoader (run_all reads these two attrs)."""

    def __init__(self, samples, tasks):
        self.all_test_samples = samples
        self.tasks = tasks


def make_task(examples=None, is_classification=False, is_binary=False,
              definition="Label the input."):
    return {
        "task_id": "task1",
        "task_name": "Task One",
        "definition": definition,
        "examples": examples if examples is not None else [],
        "is_classification": is_classification,
        "is_binary": is_binary,
        "output_types": [],
    }


def mk_engine(**config_overrides):
    config = mod.Config()
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return mod.ICLEngine(config)


# ---------------------------------------------------------------------------
# C001 — engine config pins Qwen3-4B generation settings
# ---------------------------------------------------------------------------

def test_config_pins_qwen3_4b_and_generation_settings():
    cfg = mod.Config
    assert cfg.MODEL_NAME == "Qwen/Qwen3-4B"
    assert cfg.TEMPERATURE == 0.1
    assert cfg.TOP_P == 0.85
    assert cfg.NUM_SHOT == 5
    assert cfg.MAX_INPUT_TOKENS == 30000


# ---------------------------------------------------------------------------
# C002 / C010 — model loading: NF4 quantization, device_map auto, fp16
# ---------------------------------------------------------------------------

def test_load_model_configures_nf4_quantization(monkeypatch):
    monkeypatch.setattr(mod, "AutoTokenizer", RecordingAutoTokenizer)
    monkeypatch.setattr(mod, "AutoModelForCausalLM", RecordingAutoModel)
    monkeypatch.setattr(mod, "BitsAndBytesConfig", RecordingBitsAndBytesConfig)
    engine = mk_engine()  # Config.USE_4BIT is True by default
    engine.load_model()

    kwargs = RecordingAutoModel.last_kwargs
    qcfg = kwargs["quantization_config"]
    assert qcfg.kwargs["load_in_4bit"] is True
    assert qcfg.kwargs["bnb_4bit_quant_type"] == "nf4"
    assert qcfg.kwargs["bnb_4bit_use_double_quant"] is True
    assert qcfg.kwargs["bnb_4bit_compute_dtype"] is torch.float16


def test_load_model_uses_device_map_auto_and_fp16(monkeypatch):
    monkeypatch.setattr(mod, "AutoTokenizer", RecordingAutoTokenizer)
    monkeypatch.setattr(mod, "AutoModelForCausalLM", RecordingAutoModel)
    monkeypatch.setattr(mod, "BitsAndBytesConfig", RecordingBitsAndBytesConfig)
    engine = mk_engine()
    engine.load_model()

    kwargs = RecordingAutoModel.last_kwargs
    assert kwargs["device_map"] == "auto"
    assert kwargs["torch_dtype"] is torch.float16
    assert "quantization_config" in kwargs


# ---------------------------------------------------------------------------
# C003 — few-shot selection: balanced for binary, uniform otherwise
# ---------------------------------------------------------------------------

def test_prompt_builder_balances_binary_classification_shots():
    random.seed(7)
    examples = [{"input": f"pos input {i}", "output": "positive"} for i in range(4)]
    examples += [{"input": f"neg input {i}", "output": "negative"} for i in range(4)]
    task = make_task(examples=examples, is_classification=True, is_binary=True)

    engine = mk_engine()
    prompt = engine.prompt_builder.build_prompt(task, "test input", ChatTokenizer())

    assert prompt.count("Output: positive") == 2
    assert prompt.count("Output: negative") == 2


def test_prompt_builder_uniform_sampling_for_non_binary_tasks():
    random.seed(7)
    examples = [{"input": f"input {i}", "output": "x"} for i in range(6)]
    task = make_task(examples=examples, is_classification=True, is_binary=False)

    engine = mk_engine()
    prompt = engine.prompt_builder.build_prompt(task, "test input", ChatTokenizer())

    assert "Example 5:" in prompt  # NUM_SHOT=5 examples embedded
    assert "Example 6:" not in prompt


# ---------------------------------------------------------------------------
# C004 — answer-only ICL prompting (no CoT scaffold committed)
# ---------------------------------------------------------------------------

def test_system_prompt_requests_answer_only_output():
    task = make_task(examples=[{"input": "ex in", "output": "ex out"}],
                     definition="Label the sentiment.")
    engine = mk_engine()
    prompt = engine.prompt_builder.build_prompt(task, "test input", ChatTokenizer())

    assert "Provide only the answer, nothing else" in prompt
    assert "Label the sentiment." in prompt
    assert "Input: ex in" in prompt
    assert "Output: ex out" in prompt


# ---------------------------------------------------------------------------
# C005 — self-consistency: voting available, single run is the default
# ---------------------------------------------------------------------------

def test_self_consistency_majority_vote_resolves():
    engine = mk_engine()
    engine.config.SELF_CONSISTENCY_RUNS = 3

    temps, answers = [], ["positive", "positive", "negative"]

    def fake_generate(prompt, temperature=None):
        temps.append(temperature)
        return answers[len(temps) - 1]

    engine.generate = fake_generate
    task = make_task(is_classification=True)
    task["output_types"] = ["positive", "negative"]

    assert engine.predict(task, "input text") == "positive"
    assert len(temps) == 3
    assert [round(t, 1) for t in temps] == [0.1, 0.2, 0.3]


def test_self_consistency_defaults_to_single_run():
    assert mod.Config.SELF_CONSISTENCY_RUNS == 1
    assert mod.Config.MAX_PREDICT_ATTEMPTS == 3

    engine = mk_engine()
    calls = []

    def fake_generate(prompt, temperature=None):
        calls.append(temperature)
        return "answer single"

    engine.generate = fake_generate
    task = make_task()  # not classification: first-line fallback applies

    assert engine.predict(task, "input text") == "answer single"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# C006 — tokenizer truncation at MAX_INPUT_TOKENS, no chunking
# ---------------------------------------------------------------------------

def test_generate_truncates_long_inputs_at_max_input_tokens():
    engine = mk_engine()
    tokenizer = FakeTokenizer(n_tokens=9)
    engine.tokenizer = tokenizer
    engine.model = FakeModel(n_new=3)

    assert engine.generate("a long prompt") == "decoded answer"
    assert tokenizer.calls, "tokenizer was never invoked"
    kwargs = tokenizer.calls[0]
    assert kwargs.get("truncation") is True
    assert kwargs.get("max_length") == 30000


# ---------------------------------------------------------------------------
# C007 — AnswerExtractor normalization strategies
# ---------------------------------------------------------------------------

def test_extractor_matches_known_label_first():
    task = make_task(is_classification=True)
    task["output_types"] = ["positive", "negative"]
    assert mod.AnswerExtractor.extract("I think this is negative overall", task) == "negative"


def test_extractor_output_pattern():
    task = make_task()
    assert mod.AnswerExtractor.extract("reasoning here\nOutput: 42\ntrailing", task) == "42"


def test_extractor_think_tag_and_first_line_fallback():
    task = make_task()
    assert mod.AnswerExtractor.extract(
        "<think>chain of thought</think>Final: blue", task
    ) == "Final: blue"
    assert mod.AnswerExtractor.extract("first line\nsecond line", task) == "first line"


# ---------------------------------------------------------------------------
# C008 — resilient decorator wiring on generate
# ---------------------------------------------------------------------------

def test_generate_carries_resilient_decorator():
    source = inspect.getsource(mod.ICLEngine)
    assert "@resilient(timeout=300, max_attempts=3)" in source
    assert getattr(mod.ICLEngine.generate, "__wrapped__", None) is not None


def test_generate_retries_transient_failures_under_resilient():
    engine = mk_engine()

    class FlakyTokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__(n_tokens=7)
            self.failures_left = 2
            self.n_calls = 0

        def __call__(self, prompt, return_tensors=None, **kwargs):
            self.n_calls += 1
            if self.failures_left > 0:
                self.failures_left -= 1
                raise RuntimeError("transient inference failure")
            return FakeBatch(self.n_tokens)

    engine.tokenizer = FlakyTokenizer()
    engine.model = FakeModel(n_new=3)

    assert engine.generate("prompt") == "decoded answer"
    assert engine.tokenizer.n_calls == 3  # two transient failures, then success


# ---------------------------------------------------------------------------
# C009 — per-sample retry and auditable atomic error sidecar
# ---------------------------------------------------------------------------

def test_run_all_retries_each_sample_up_to_max_attempts():
    engine = mk_engine()
    attempts = {"n": 0}

    def flaky_predict(task, test_input):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return "recovered"

    engine.predict = flaky_predict
    loader = FakeLoader(
        samples=[{"id": "id1", "input": "x", "task_id": "task1"}],
        tasks=[make_task()],
    )

    df = engine.run_all(loader)
    assert attempts["n"] == 3
    assert df.iloc[0]["Predicted"] == "recovered"


def test_run_all_records_exhausted_failures_in_error_sidecar(tmp_path):
    engine = mk_engine()

    def always_fail(task, test_input):
        raise ValueError("nope")

    engine.predict = always_fail
    loader = FakeLoader(
        samples=[{"id": "id9", "input": "x", "task_id": "task1"}],
        tasks=[make_task()],
    )

    df = engine.run_all(loader)
    assert df.iloc[0]["Predicted"] == ""  # empty-answer fallback

    sidecar = engine.flush_errors(str(tmp_path / "submission.csv"))
    assert sidecar == str(tmp_path / "submission.errors.json")
    payload = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    assert payload[0]["id"] == "id9"
    assert payload[0]["task_id"] == "task1"
    assert payload[0]["exc_type"] == "ValueError"

    # A fresh engine with no recorded errors writes nothing.
    assert mk_engine().flush_errors(str(tmp_path / "clean.csv")) is None
