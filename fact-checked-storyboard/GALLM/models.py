"""Helper-model wrappers. GALLM sends a system message and samples at temperature 1.0,
while GAAPO's optimizer wrapper sends one user message at temperature 0."""
import re
import time


class ChatLLM:
    window_retries, window_seconds = 3, 60

    def __init__(self, client, model, max_tokens=None):
        self.client, self.model, self.max_tokens = client, model, max_tokens

    @classmethod
    def from_settings(cls, settings):
        from GAAPO.models import configured_llm
        settings = dict(settings)
        # Free tiers (e.g. Groq) answer 429 when over their per-minute token limit; the OpenAI client
        # waits as the provider's retry-after header says and tries again.
        max_retries = settings.pop('max_retries', 2)
        # Some providers count the reply budget against the per-minute limit before answering.
        max_tokens = settings.pop('max_tokens', None)
        # Reuse GAAPO's key handling (api_key_env, model_env, base_url); keep only its client.
        llm = configured_llm(settings)
        return cls(llm.client.with_options(max_retries=max_retries), llm.model, max_tokens)

    def generate(self, prompt, system=None, temperature=1.0, top_p=0.9):
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]

        def complete(max_tokens):
            limit = {'max_tokens': max_tokens} if max_tokens else {}
            return self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature,
                                                       top_p=top_p, **limit).choices[0].message.content
        for attempt in range(self.window_retries + 1):
            try:
                content = complete(self.max_tokens)
                break
            except Exception as error:
                # Groq answers 413 "Limit 8000, Requested 8321" when this request plus the tokens already
                # used in the current minute exceed its per-minute limit. Unlike its 429 answers it sends
                # no retry-after, so wait for the minute to pass. A request that alone exceeds the limit
                # keeps failing and is raised after the last attempt.
                if (getattr(error, 'status_code', None) != 413 or not re.search(r'Limit \d+, Requested \d+', str(error))
                        or attempt == self.window_retries):
                    raise
                time.sleep(self.window_seconds)
        if not content or not content.strip():
            raise ValueError('Helper model returned an empty response')
        return content.strip()


class LoggedChatLLM:
    def __init__(self, llm, log):
        self.llm, self.log = llm, log

    def generate(self, prompt, system=None, **sampling):
        started = time.perf_counter()
        record = {'status': 'started', 'system': system, 'prompt': prompt, 'sampling': sampling}
        self.log.write('optimizer_calls.jsonl', record)
        output = self.llm.generate(prompt, system=system, **sampling)
        self.log.write('optimizer_calls.jsonl', {**record, 'status': 'completed', 'output': output,
                                                 'seconds': time.perf_counter() - started})
        return output


class MockChatLLM:
    """Scripted helper for dry runs and tests; records every call."""
    def __init__(self, responses=None):
        self.calls = []
        self.responses = iter(responses) if responses is not None else None

    def generate(self, prompt, system=None, **sampling):
        self.calls.append({'prompt': prompt, 'system': system, 'sampling': sampling})
        if self.responses is not None:
            return next(self.responses)
        return (f'Mock prompt {len(self.calls)}: generate a patient-oriented storyboard'
                + (' as JSON with title and scenes.' if len(self.calls) % 2 else '.'))
