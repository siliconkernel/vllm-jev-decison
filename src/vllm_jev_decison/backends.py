"""In-process vLLM backend and an explicitly separate HTTP compatibility backend."""
import asyncio
from pathlib import Path


class BackendError(RuntimeError):
    pass


def check_labels(ids):
    if any(len(item) != 1 for item in ids) or len({item[0] for item in ids}) != len(ids):
        raise BackendError('This tokenizer does not encode candidate letters as distinct single tokens')
    return [item[0] for item in ids]


def usage(prompt, output):
    return {'input_tokens': prompt, 'classification_tokens': output,
            'generated_tokens': 0, 'engine_requests': 1}


class VLLMBackend:
    name = 'vllm_endpoint_plugin'

    def __init__(self, engine, args):
        if engine is None:
            raise BackendError('Decision plugin requires a generation engine')
        if engine.model_config.logprobs_mode != 'raw_logprobs':
            raise BackendError('Decision plugin requires --logprobs-mode raw_logprobs')
        self.engine = engine
        self.tokenizer = engine.renderer.get_tokenizer()
        self.model = engine.model_config.model
        self.template = getattr(args, 'chat_template', None)
        if self.template and not any(c in self.template for c in ('\n', '{', '}')):
            self.template = Path(self.template).read_text()
        self.label_ids = check_labels([self.tokenizer.encode(label, add_special_tokens=False) for label in 'ABCDEFGHIJKLMNOP'])
        if 0 <= engine.model_config.max_logprobs < 16:
            raise BackendError('Set --max-logprobs 16 or higher')

    def _tokens(self, messages):
        if self.template or getattr(self.tokenizer, 'chat_template', None):
            tokens = self.tokenizer.apply_chat_template(messages, chat_template=self.template, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, thinking=False)
        else:
            prompt = '\n\n'.join(message['role'].upper() + ':\n' + message['content'] for message in messages)
            tokens = self.tokenizer.encode(prompt + '\n\nASSISTANT:\n', add_special_tokens=True)
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise BackendError('Tokenizer must produce text token IDs')
        return tokens

    async def _run(self, messages, params, request_id):
        from vllm.inputs import tokens_input
        tokens = await asyncio.to_thread(self._tokens, messages)
        final = None
        try:
            async for result in self.engine.generate(tokens_input(tokens), params, request_id):
                final = result
            if final is None or not final.finished or len(final.outputs) != 1:
                raise BackendError('Engine returned an incomplete result')
            return tokens, final.outputs[0]
        finally:
            # Abort is idempotent for a completed request, and releases work on cancellation.
            await self.engine.abort(request_id)

    async def classify(self, messages, labels, request_id):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        ids = self.label_ids[:len(labels)]
        params = SamplingParams(temperature=0, max_tokens=1, logprob_token_ids=ids, logprobs=len(ids),
                                output_kind=RequestOutputKind.FINAL_ONLY)
        tokens, output = await self._run(messages, params, request_id)
        if len(output.token_ids) != 1 or not output.logprobs:
            raise BackendError('Missing one-step classification logprobs')
        try:
            scores = [output.logprobs[0][token].logprob for token in ids]
        except KeyError as error:
            raise BackendError('Engine omitted requested candidate logprobs') from error
        return {'scores': scores, 'usage': usage(len(tokens), 1)}



class HTTPBackend:
    """Optional bridge: reaches an existing server, not a loaded endpoint plugin."""
    name = 'http_bridge'

    def __init__(self, url, model, api_key=None):
        import httpx
        self.model = model
        headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}
        self.client = httpx.AsyncClient(base_url=url.rstrip('/'), headers=headers, timeout=180, trust_env=False)
        self.label_ids = None
        self.label_lock = asyncio.Lock()

    async def close(self):
        await self.client.aclose()

    async def post(self, path, payload):
        response = await self.client.post(path, json=payload)
        if not response.is_success:
            raise BackendError(f'Upstream {path} returned HTTP {response.status_code}: {response.text[:400]}')
        return response.json()

    async def _labels(self):
        async with self.label_lock:
            if self.label_ids is None:
                results = await asyncio.gather(*(self.post('/tokenize', {'model': self.model, 'prompt': label,
                    'add_special_tokens': False}) for label in 'ABCDEFGHIJKLMNOP'))
                self.label_ids = check_labels([r['tokens'] for r in results])
        return self.label_ids

    async def classify(self, messages, labels, request_id):
        ids = (await self._labels())[:len(labels)]
        prepared = await self.post('/tokenize', {'model': self.model, 'messages': messages,
            'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False, 'thinking': False}})
        result = await self.post('/v1/completions', {'model': self.model, 'prompt': prepared['tokens'],
            'temperature': 0, 'max_tokens': 1, 'logprob_token_ids': ids, 'logprobs': len(ids),
            'return_tokens_as_token_ids': True, 'return_token_ids': True})
        choice = result['choices'][0]
        if len(choice.get('token_ids', [])) != 1:
            raise BackendError('Expected exactly one sampled classification transport token')
        scores = [choice['logprobs']['top_logprobs'][0][f'token_id:{token}'] for token in ids]
        return {'scores': scores, 'usage': usage(result['usage']['prompt_tokens'], result['usage']['completion_tokens'])}
