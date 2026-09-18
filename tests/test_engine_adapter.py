"""Adapter contract tests with a fake engine; not GPU integration evidence."""
import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_jev_decison.backends import BackendError, VLLMBackend


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = ModuleType('vllm')
    params = ModuleType('vllm.sampling_params')
    inputs = ModuleType('vllm.inputs')
    vllm.SamplingParams = SimpleNamespace
    params.RequestOutputKind = SimpleNamespace(FINAL_ONLY='final')
    params.StructuredOutputsParams = SimpleNamespace
    inputs.tokens_input = lambda tokens: {'prompt_token_ids': tokens}
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', params)
    monkeypatch.setitem(sys.modules, 'vllm.inputs', inputs)


class Tokenizer:
    chat_template = 'test'
    def encode(self, text, **kwargs):
        return [ord(text)]
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


class Engine:
    model_config = SimpleNamespace(model='fake', logprobs_mode='raw_logprobs', max_logprobs=20)
    renderer = SimpleNamespace(get_tokenizer=lambda: Tokenizer())
    def __init__(self, finish='stop'):
        self.finish = finish
        self.aborted = []
    async def generate(self, prompt, params, request_id):
        self.params = params
        scores = {65: SimpleNamespace(logprob=-2), 66: SimpleNamespace(logprob=-1)}
        yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=[99], logprobs=[scores],
            finish_reason=self.finish, text='true')])
    async def abort(self, request_id):
        self.aborted.append(request_id)


def test_classification_uses_raw_requested_scores_not_sampled_token(fake_vllm):
    engine = Engine()
    backend = VLLMBackend(engine, SimpleNamespace())
    result = asyncio.run(backend.classify([], ['A', 'B'], 'one'))
    assert result['scores'] == [-2, -1]
    assert engine.params.max_tokens == 1
    assert engine.params.logprob_token_ids == [65, 66]
    assert engine.aborted == ['one']
    assert result['usage']['classification_tokens'] == 1

