import asyncio
import math
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from vllm_decision.backends import BackendError, VLLMBackend, check_labels
from vllm_decision.plugin import DecisionPlugin
from vllm_decision.schema import assemble, plan, strict_json
from vllm_decision.service import DecisionRequest, DecisionService, distribution


class Backend:
    model = 'fake-model'
    name = 'test-only'
    def __init__(self, scores=None, text='"hello"', wait=0):
        self.scores = scores or [-2.0, -1.0]
        self.text = text
        self.wait = wait
        self.active = 0
        self.peak = 0
        self.cancelled = 0
        self.generated = 0
    async def classify(self, messages, labels, request_id):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.wait)
            return {'scores': self.scores, 'usage': {'input_tokens': 10, 'classification_tokens': 1, 'generated_tokens': 0, 'engine_requests': 1}}
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1
    async def generate(self, messages, schema, limit, request_id):
        self.generated += 1
        return {'text': self.text, 'usage': {'input_tokens': 15, 'classification_tokens': 0, 'generated_tokens': 4, 'engine_requests': 1}}


def request(schema, **kwargs):
    return DecisionRequest(state='An example input', schema=schema, **kwargs)


def closed(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def test_finite_nested_plan_and_assembly():
    schema = closed({'urgent': {'type': 'boolean'}, 'nested': closed({'rating': {'type': 'integer', 'minimum': 1, 'maximum': 5}})})
    fields = plan(schema, 'classify')
    assert [f.choices for f in fields] == [[False, True], [1, 2, 3, 4, 5]]
    assert assemble(fields, [True, 3]) == {'urgent': True, 'nested': {'rating': 3}}


@pytest.mark.parametrize('schema', [
    {'type': 'string'}, {'type': 'integer', 'minimum': 0, 'maximum': 100},
    {'type': 'object', 'properties': {'optional': {'type': 'boolean'}}},
    {**closed({'a': {'type': 'boolean'}}), 'if': {'properties': {'a': {'const': True}}}, 'then': {'maxProperties': 0}},
])
def test_nonfinite_or_joint_constraints_generate_as_whole(schema):
    assert plan(schema)[0].choices is None
    with pytest.raises(ValueError):
        plan(schema, 'classify')


@pytest.mark.parametrize('schema', [
    {'$ref': 'https://invalid.example/schema'}, {'$ref': '#/$defs/a', '$defs': {'a': {'type': 'boolean'}}},
    {'type': 'boolean', 'enum': [1]}, {'type': 'object', 'properties': {'x': False}, 'required': ['x'], 'additionalProperties': False},
])
def test_unsupported_or_empty_schema_fails(schema):
    with pytest.raises(ValueError):
        plan(schema)


def test_explicit_generate_does_not_classify():
    assert plan({'type': 'boolean'}, 'generate')[0].choices is None


def test_probabilities_are_conditional_and_mass_is_separate():
    probs, mass = distribution([math.log(0.001), math.log(0.009)])
    assert probs == pytest.approx([0.1, 0.9])
    assert mass == pytest.approx(0.01)


@pytest.mark.parametrize('scores', [[math.nan], [math.inf], [-math.inf], [0.2, -1], [0, 0]])
def test_bad_backend_scores_rejected(scores):
    with pytest.raises(ValueError):
        distribution(scores)


def test_negative_infinity_candidate_has_zero_probability():
    assert distribution([-math.inf, -2])[0] == [0, 1]


def test_parallel_classification_and_usage():
    backend = Backend(wait=0.01)
    schema = closed({'a': {'type': 'boolean'}, 'b': {'enum': ['x', 'y']}})
    result = asyncio.run(DecisionService(backend).decide(request(schema)))
    assert result['value'] == {'a': True, 'b': 'y'}
    assert backend.peak == 2
    assert result['usage'] == {'input_tokens': 20, 'classification_tokens': 2, 'generated_tokens': 0, 'engine_requests': 2}


def test_low_confidence_abstains_without_dispatchable_value():
    result = asyncio.run(DecisionService(Backend()).decide(request({'type': 'boolean'}, min_confidence=0.99)))
    assert not result['accepted'] and result['value'] is None
    assert result['decisions'][0]['value'] is True


def test_mixed_schema_and_no_confidence_for_generation():
    schema = closed({'flag': {'type': 'boolean'}, 'message': {'type': 'string'}})
    result = asyncio.run(DecisionService(Backend()).decide(request(schema)))
    assert result['value'] == {'flag': True, 'message': 'hello'}
    assert result['decisions'][1]['confidence'] is None
    assert result['usage']['generated_tokens'] == 4


def test_constant_does_not_use_model():
    result = asyncio.run(DecisionService(Backend()).decide(request({'const': {'status': 'fixed'}})))
    assert result['value'] == {'status': 'fixed'} and result['usage']['engine_requests'] == 0


def test_timeout_cancels_children():
    async def run():
        backend = Backend(wait=1)
        with pytest.raises(TimeoutError):
            await DecisionService(backend, timeout=0.01).decide(request(closed({'a': {'type': 'boolean'}, 'b': {'type': 'boolean'}})))
        assert backend.active == 0 and backend.cancelled == 2
    asyncio.run(run())


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', 'NaN', 'Infinity', '{"a":NaN}'])
def test_invalid_json_rejected(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_labels_are_checked():
    assert check_labels([[1], [2]]) == [1, 2]
    for ids in ([[1], [1]], [[1, 2], [3]]):
        with pytest.raises(BackendError):
            check_labels(ids)


def test_routes_auth_capabilities_and_schema_validation():
    async def run():
        app = FastAPI()
        DecisionPlugin().attach_router(app)
        app.state.decision_service = DecisionService(Backend())
        app.state.decision_keys = ['test-key']
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.get('/plugins/decision/capabilities')).status_code == 401
            headers = {'Authorization': 'Bearer test-key'}
            assert (await client.get('/plugins/decision/capabilities', headers=headers)).json()['backend'] == 'test-only'
            r = await client.post('/plugins/decision/infer', headers=headers, json={'state': 'x', 'schema': {'type': 'boolean'}})
            assert r.status_code == 200 and r.json()['value'] is True
            r = await client.post('/plugins/decision/infer', headers=headers, json={'state': 'x', 'schema': {'$ref': 'http://no.example'}})
            assert r.status_code == 422
    asyncio.run(run())


def test_plugin_refuses_processed_probabilities():
    engine = SimpleNamespace(model_config=SimpleNamespace(logprobs_mode='processed_logprobs'))
    with pytest.raises(BackendError, match='raw_logprobs'):
        VLLMBackend(engine, SimpleNamespace())


def test_disconnect_cancels_engine_work():
    async def run():
        import json
        app = FastAPI()
        DecisionPlugin().attach_router(app)
        backend = Backend(wait=10)
        app.state.decision_service = DecisionService(backend)
        app.state.decision_keys = []
        sent_body = False
        received = []
        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {'type': 'http.request', 'body': json.dumps({'state': 'x', 'schema': {'type': 'boolean'}}).encode(), 'more_body': False}
            await asyncio.sleep(0.03)
            return {'type': 'http.disconnect'}
        async def send(message):
            received.append(message)
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                 'scheme': 'http', 'path': '/plugins/decision/infer', 'raw_path': b'/plugins/decision/infer',
                 'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                 'client': ('127.0.0.1', 1), 'server': ('test', 80), 'root_path': ''}
        await asyncio.wait_for(app(scope, receive, send), timeout=2)
        assert backend.cancelled == 1 and backend.active == 0
        assert received[0]['status'] == 499
    asyncio.run(run())


def test_generation_is_validated_not_just_parsed():
    from jsonschema.exceptions import ValidationError
    with pytest.raises(ValidationError):
        asyncio.run(DecisionService(Backend(text='123')).decide(request({'type': 'string'})))
