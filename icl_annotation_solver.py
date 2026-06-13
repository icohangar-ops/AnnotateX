"""
AnnotateX: LLM Automatic Data Annotation in Long-Context Scenarios
===================================================================
FlagOS Open Computing Hackathon - Track 3
ICL (In-Context Learning) based solution using Qwen3-4B.

Handles 8 diverse OpenSeek tasks:
  1. Closest Integers (math)
  2. Count Nouns & Verbs (NLP)
  3. Collatz Conjecture (math)
  4. Concat Strings (code)
  5. Tweet Sadness Detection (binary classification)
  6. MNLI Genre Classification (binary classification)
  7. Jeopardy Answer Generation (QA)
  8. Kernel Generation (code generation)
"""

import json
import os
import glob
import random
import re
import time
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cubiczan_resilience import resilient, atomic_write


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    MODEL_NAME = "Qwen/Qwen3-4B"
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.1
    TOP_P = 0.85
    NUM_SHOT = 5
    SELF_CONSISTENCY_RUNS = 1
    USE_4BIT = True
    MAX_INPUT_TOKENS = 30000
    MAX_PREDICT_ATTEMPTS = 3

    INPUT_DIR = "/kaggle/input/track-3-llm-automatic-data-annotation-in-long-context-scenarios"
    OUTPUT_PATH = "/kaggle/working/submission.csv"


# ============================================================
# DATA LOADER
# ============================================================
class TaskLoader:
    """Loads and manages multi-task OpenSeek data."""

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.tasks = []
        self.all_test_samples = []

    def load(self) -> List[Dict]:
        """Load all JSON task files."""
        json_files = sorted(glob.glob(os.path.join(self.input_dir, "*.json")))
        print(f"Found {len(json_files)} task files")

        for fpath in json_files:
            with open(fpath, 'r') as f:
                data = json.load(f)

            definition = data['Definition'][0] if isinstance(data['Definition'], list) else data['Definition']
            examples = data['examples']
            test_samples = data['test_samples']

            # Determine output types
            sample_outputs = [
                str(ex['output'][0]) if isinstance(ex['output'], list) else str(ex['output'])
                for ex in examples[:50]
            ]
            unique_outputs = list(set(sample_outputs))
            is_binary = len(unique_outputs) == 2
            is_classification = len(unique_outputs) <= 10

            task = {
                'task_id': data['task_id'],
                'task_name': data['task_name'],
                'definition': definition,
                'examples': examples,
                'test_samples': test_samples,
                'is_binary': is_binary,
                'is_classification': is_classification,
                'output_types': unique_outputs[:5],
            }
            self.tasks.append(task)

            # Collect test samples
            for ts in test_samples:
                self.all_test_samples.append({
                    'id': ts['id'],
                    'input': ts['input'],
                    'task_id': data['task_id'],
                    'task_name': data['task_name'],
                })

            print(f"  {data['task_id']:20s} | ex={len(examples):5d} | test={len(test_samples):4d} | binary={is_binary}")

        print(f"\nTotal: {len(self.all_test_samples)} test samples across {len(self.tasks)} tasks")
        return self.tasks


# ============================================================
# PROMPT BUILDER
# ============================================================
class PromptBuilder:
    """Builds task-specific ICL prompts using Qwen3 chat template."""

    def __init__(self, config: Config):
        self.config = config

    def build_prompt(self, task: Dict, test_input, tokenizer) -> str:
        """Build ICL prompt for a specific task and test input."""
        definition = task['definition']
        examples = task['examples']
        num_shot = self.config.NUM_SHOT

        # Select diverse few-shot examples
        if task['is_classification'] and task['is_binary']:
            labels = [
                str(ex['output'][0]) if isinstance(ex['output'], list) else str(ex['output'])
                for ex in examples
            ]
            unique_labels = list(set(labels))
            selected = []
            per_label = max(1, num_shot // len(unique_labels))
            for lbl in unique_labels:
                matching = [
                    ex for ex in examples
                    if str(ex['output'][0] if isinstance(ex['output'], list) else ex['output']) == lbl
                ]
                selected.extend(random.sample(matching, min(per_label, len(matching))))
            random.shuffle(selected)
            selected = selected[:num_shot]
        else:
            selected = random.sample(examples, min(num_shot, len(examples)))

        # Build messages for chat template
        messages = []

        system_msg = (
            f"You are an expert assistant. Your task is: {definition}\n\n"
            "Follow the format of the examples below. Provide only the answer, nothing else."
        )
        messages.append({"role": "system", "content": system_msg})

        # Few-shot examples
        examples_text = ""
        for i, ex in enumerate(selected):
            inp = str(ex['input'])
            out = str(ex['output'][0]) if isinstance(ex['output'], list) else str(ex['output'])
            examples_text += f"\nExample {i+1}:\nInput: {inp}\nOutput: {out}\n"

        user_msg = f"Here are some examples:\n{examples_text}\n\nNow solve this:\nInput: {str(test_input)}\nOutput:"
        messages.append({"role": "user", "content": user_msg})

        # Apply chat template
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback manual template
            prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg['role'] == 'user':
                    prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

        return prompt


# ============================================================
# ANSWER EXTRACTOR
# ============================================================
class AnswerExtractor:
    """Extracts clean answers from model responses."""

    @staticmethod
    def extract(response: str, task: Dict) -> str:
        """Extract answer based on task type."""
        resp = response.strip()

        # For classification tasks, find matching label
        if task['is_classification']:
            for label in task['output_types']:
                if label.lower() in resp.lower():
                    return label

        # Try "Output:" pattern
        match = re.search(r'Output:\s*(.+?)(?:\n|$)', resp, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Handle thinking models
        think_match = re.search(r'</think[^>]*>\s*(.+)', resp, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()

        # Fallback: first line
        first_line = resp.split('\n')[0].strip()
        return first_line if first_line else resp


# ============================================================
# ICL ENGINE
# ============================================================
class ICLEngine:
    """Main inference engine."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.prompt_builder = PromptBuilder(config)
        self.extractor = AnswerExtractor()

    def load_model(self):
        """Load Qwen3-4B with optional 4-bit quantization."""
        print(f"Loading {self.config.MODEL_NAME}...")
        t0 = time.time()

        bnb_config = None
        if self.config.USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": "auto"}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(self.config.MODEL_NAME, **kwargs)
        self.model.eval()

        print(f"Model loaded in {time.time()-t0:.1f}s")

    @resilient(timeout=300, max_attempts=3)
    def generate(self, prompt: str, temperature: float = None) -> str:
        """Generate model response."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.config.MAX_INPUT_TOKENS
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=temperature or self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                do_sample=(temperature or self.config.TEMPERATURE) > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def predict(self, task: Dict, test_input) -> str:
        """Generate prediction for a single test sample."""
        prompt = self.prompt_builder.build_prompt(task, test_input, self.tokenizer)

        if self.config.SELF_CONSISTENCY_RUNS > 1:
            votes = Counter()
            for run in range(self.config.SELF_CONSISTENCY_RUNS):
                temp = self.config.TEMPERATURE + run * 0.1
                resp = self.generate(prompt, temperature=temp)
                ans = self.extractor.extract(resp, task)
                votes[ans] += 1
            return votes.most_common(1)[0][0]
        else:
            resp = self.generate(prompt)
            return self.extractor.extract(resp, task)

    def run_all(self, loader: TaskLoader) -> pd.DataFrame:
        """Process all test samples across all tasks."""
        task_lookup = {t['task_id']: t for t in loader.tasks}
        predictions = []
        errors = []
        total = len(loader.all_test_samples)
        start = time.time()

        for i, sample in enumerate(loader.all_test_samples):
            task = task_lookup[sample['task_id']]

            if i % 10 == 0:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                print(f"[{i+1}/{total}] {task['task_id'][:25]:25s} | ETA: {eta/60:.1f}m")

            answer = ""
            last_exc = None
            for attempt in range(self.config.MAX_PREDICT_ATTEMPTS):
                try:
                    answer = self.predict(task, sample['input'])
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    print(f"  ERROR {sample['id']} (attempt {attempt+1}"
                          f"/{self.config.MAX_PREDICT_ATTEMPTS}): "
                          f"{type(e).__name__}: {e}")
                    # Free GPU memory between attempts before retrying.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if last_exc is not None:
                # All attempts exhausted: fall back to empty answer but record
                # the failure so it is auditable in the sidecar file.
                errors.append({
                    'id': sample['id'],
                    'task_id': sample['task_id'],
                    'exc_type': type(last_exc).__name__,
                    'message': str(last_exc),
                })

            predictions.append({'ID': sample['id'], 'Predicted': answer})

            if (i + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - start
        print(f"\nDone! {total} samples in {elapsed/60:.1f}m ({elapsed/total:.1f}s/sample)")
        if errors:
            print(f"WARNING: {len(errors)} sample(s) failed all "
                  f"{self.config.MAX_PREDICT_ATTEMPTS} attempts; see error sidecar.")

        self._last_errors = errors
        return pd.DataFrame(predictions)

    def flush_errors(self, submission_path: str) -> Optional[str]:
        """Write any collected prediction errors to a sidecar file alongside
        the submission so failures are auditable. Returns the sidecar path,
        or None if there were no errors."""
        errors = getattr(self, '_last_errors', None)
        if not errors:
            return None
        sidecar = os.path.splitext(submission_path)[0] + ".errors.json"
        atomic_write(sidecar, json.dumps(errors, indent=2, ensure_ascii=False))
        print(f"Error sidecar written: {sidecar} ({len(errors)} failures)")
        return sidecar


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("AnnotateX: FlagOS Track 3 - ICL Data Annotation")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config = Config()

    # Load data
    loader = TaskLoader(config.INPUT_DIR)
    loader.load()

    # Load model & generate
    engine = ICLEngine(config)
    engine.load_model()

    sub_df = engine.run_all(loader)

    # Save
    os.makedirs(os.path.dirname(config.OUTPUT_PATH), exist_ok=True)
    sub_df.to_csv(config.OUTPUT_PATH, index=False)

    # Persist any prediction failures alongside the submission for auditing.
    engine.flush_errors(config.OUTPUT_PATH)

    print(f"\nSubmission saved: {config.OUTPUT_PATH}")
    print(f"Rows: {len(sub_df)}, Columns: {list(sub_df.columns)}")
    print(f"\nPreview:")
    print(sub_df.head(10).to_string(index=False))

    return sub_df


if __name__ == "__main__":
    main()
