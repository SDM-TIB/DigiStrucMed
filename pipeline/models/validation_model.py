from typing import Optional, List, Dict
import json
import re
import ast
class ValidationModel:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.model_name = model_name
        self.llm_inference = None
        self._load_model()
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def extract_statements_batch(
        self,
        texts: List[str],
        entities_list: List[List[Dict]],
        max_new_tokens: int = 400,
        batch_size: int = 4
    ) -> List[List[Dict]]:
        import torch
        all_results: List[List[Dict]] = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_entities = entities_list[i:i + batch_size]
            prompts = [
                self._build_prompt(text, ents)
                for text, ents in zip(batch_texts, batch_entities)
            ]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            if self.device == "cuda":
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            input_lengths = inputs["input_ids"].shape[1]
            for j, output in enumerate(outputs):
                generated_ids = output[input_lengths:]
                raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                statements = self._parse_extraction_output(raw)
                all_results.append(statements)
        return all_results
    def validate_statements_batch(
        self,
        texts: List[str],
        entities_list: List[List[Dict]],
        max_new_tokens: int = 400,
        batch_size: int = 4
    ) -> List[List[Dict]]:
        return self.extract_statements_batch(texts, entities_list, max_new_tokens, batch_size)
    def _build_prompt(self, text: str, entities: Optional[List[Dict]] = None) -> str:
        return f"""{self._get_system_prompt()}
TEXT:
{text.strip()}
"""
    def _parse_extraction_output(self, raw: str) -> List[Dict]:
        extracted = self._extract_json_array_or_null(raw)
        if extracted is None:
            return []
        try:
            if isinstance(extracted, list):
                items = extracted
            else:
                items = [extracted] if isinstance(extracted, dict) else []
            result = []
            for data in items:
                if not isinstance(data, dict):
                    continue
                data = self._normalize_nullish_values(data)
                cleaned = self._validate_and_clean_single_statement(data)
                if cleaned:
                    result.append(cleaned)
            return result
        except Exception:
            return []
    def _parse_single_output(self, raw: str) -> Optional[Dict]:
        statements = self._parse_extraction_output(raw)
        return statements[0] if statements else None
    def run_inference(self, prompt: str, max_new_tokens: int = 400) -> str:
        import torch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        )
        if self.device == "cuda":
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    def validate_statement(
        self,
        text: str,
        entities: Optional[List[Dict]] = None,
        max_new_tokens: int = 150
    ) -> Optional[Dict]:
        prompt = self._build_prompt(text, entities)
        raw_output = self.run_inference(prompt, max_new_tokens=max_new_tokens).strip()
        return self._parse_single_output(raw_output)
    def _get_system_prompt(self) -> str:
        return """You are a medical guideline assistant. Your task is to EXTRACT every explicit factual statement from the given TEXT. You do NOT validate whether statements are true—experts will do that later.
Think step-by-step to identify all statements, but DO NOT include your reasoning in the output.
Goal: maximize recall. If a statement is explicitly stated, extract it.
If there is ANY declarative clause with a subject and a verb, output at least one statement (do NOT return [] just because it seems weak).
If subject or object is unclear, still output the statement and use null for the unclear field.
Split compound or multi-sentence text into separate statements when they express distinct facts.
Extract any explicit factual statement (recommendations, definitions, criteria, risks, effects, associations, or descriptive facts).
Do NOT add medical knowledge or infer beyond what is explicitly stated.
Only output [] if the text is purely a header/title/figure caption/metadata with no declarative clause.
If the text is a list or multi-line block, extract statements from each line that contains a declarative clause.
Ignore citation markers and numbering (e.g., "1.", "2.", "3.", "1–5", "1A") and remove them from the statement text.
Output JSON ONLY (no reasoning or extra text). Use double quotes and valid JSON.
Schema for each statement in the JSON array:
- subject: who or what the fact is about. null if unclear.
- predicate: the verb or relation.
- object: what the predicate applies to. null if not applicable.
- exception: ONLY if the text states an explicit exception. null otherwise.
- duration: ONLY if the text states an explicit timeframe. NOT study periods.
Example output:
[{"subject": "ACEi", "predicate": "is", "object": "angiotensin-converting enzyme inhibitors", "exception": null, "duration": null}, {"subject": "patients with HFrEF", "predicate": "should receive", "object": "ACE inhibitors", "exception": null, "duration": null}]
If no factual statement, output: []""".strip()
    def _extract_json_or_null(self, raw: str) -> Optional[str]:
        if raw is None:
            return None
        raw = re.sub(r"^\s*```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
        raw = raw.strip()
        if re.fullmatch(r"null", raw, flags=re.IGNORECASE):
            return "null"
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            return m.group(0).strip()
        if "{" not in raw and "}" not in raw and re.search(r"\bnull\b", raw, flags=re.IGNORECASE):
            return "null"
        return None
    def _extract_json_array_or_null(self, raw: str) -> Optional[List]:
        if raw is None:
            return None
        raw_clean = re.sub(r"^\s*```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw_clean = re.sub(r"\s*```\s*$", "", raw_clean).strip()
        try:
            parsed = json.loads(raw_clean)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            i = raw_clean.find(start_char)
            if i == -1:
                continue
            depth = 0
            for j in range(i, len(raw_clean)):
                if raw_clean[j] == start_char:
                    depth += 1
                elif raw_clean[j] == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(raw_clean[i : j + 1])
                            return [parsed] if isinstance(parsed, dict) else parsed
                        except Exception:
                            pass
                        break
        return None
    def _normalize_nullish_values(self, d: Dict) -> Dict:
        nullish_strings = {"null", "none", "n/a", "na", ""}
        out = dict(d)
        for k, v in list(out.items()):
            if isinstance(v, str) and v.strip().lower() in nullish_strings:
                out[k] = None
        return out
    def _validate_and_clean_single_statement(self, data: Dict) -> Optional[Dict]:
        subject = data.get("subject") or data.get("population")
        predicate = data.get("predicate")
        obj = data.get("object")
        exception = data.get("exception")
        duration = data.get("duration")
        action = data.get("action")
        if action and isinstance(action, str) and (not predicate or not obj):
            parts = action.strip().split(None, 1)
            if not predicate and parts:
                predicate = parts[0]
            if not obj and len(parts) > 1:
                obj = parts[1]
            elif not obj and len(parts) == 1:
                obj = None
        if not subject and not predicate:
            return None
        subject = subject.strip() if isinstance(subject, str) and subject else None
        predicate = predicate.strip() if isinstance(predicate, str) and predicate else None
        obj = obj.strip() if isinstance(obj, str) and obj else None
        if not predicate or len((predicate or "").strip()) < 2:
            return None
        if exception and isinstance(exception, str):
            exception_lower = exception.lower().strip()
            if exception_lower in {"unless", "except", "exception", "null", "none", "n/a"} or len(exception.split()) < 2:
                exception = None
            else:
                exception = exception.strip()
        else:
            exception = None
        if duration and isinstance(duration, str):
            duration_lower = duration.lower().strip()
            vague = {"short-term", "short term", "long-term", "long term", "temporary", "permanent", "brief", "extended"}
            if duration_lower in vague:
                duration = None
            else:
                temporal = ["day", "week", "month", "year", "hour", "minute", "until", "before", "after", "within", "during"]
                if not any(kw in duration_lower for kw in temporal):
                    duration = None
                else:
                    duration = duration.strip()
        else:
            duration = None
        return {
            "subject": subject or "unspecified",
            "predicate": predicate,
            "object": obj,
            "exception": exception,
            "duration": duration,
        }
