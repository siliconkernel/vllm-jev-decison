import asyncio
import math
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from vllm_jev_decison.backends import BackendError, VLLMBackend, check_labels
from vllm_jev_decison.plugin import DecisionPlugin
from vllm_jev_decison.schema import assemble, plan
from vllm_jev_decison.service import DecisionRequest, DecisionService, distribution


class Backend:
    model = 'fake-model'
    name = 'test-only'
    def __init__(self, scores=None, wait=0):
        self.scores = scores or [-2.0, -1.0]
        self.wait = wait
        self.active = 0
        self.peak = 0
        self.cancelled = 0
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
def test_nonfinite_or_joint_constraints_are_rejected(schema):
    with pytest.raises(ValueError):
        plan(schema, 'classify')


@pytest.mark.parametrize('schema', [
    {'$ref': 'https://invalid.example/schema'}, {'$ref': '#/$defs/a', '$defs': {'a': {'type': 'boolean'}}},
    {'type': 'boolean', 'enum': [1]}, {'type': 'object', 'properties': {'x': False}, 'required': ['x'], 'additionalProperties': False},
])
def test_unsupported_or_empty_schema_fails(schema):
    with pytest.raises(ValueError):
        plan(schema)


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
            assert (await client.get('/plugins/jev-decison/capabilities')).status_code == 401
            headers = {'Authorization': 'Bearer test-key'}
            assert (await client.get('/plugins/jev-decison/capabilities', headers=headers)).json()['backend'] == 'test-only'
            r = await client.post('/plugins/jev-decison/infer', headers=headers, json={'state': 'x', 'schema': {'type': 'boolean'}})
            assert r.status_code == 200 and r.json()['value'] is True
            r = await client.post('/plugins/jev-decison/infer', headers=headers, json={'state': 'x', 'schema': {'$ref': 'http://no.example'}})
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
                 'scheme': 'http', 'path': '/plugins/jev-decison/infer', 'raw_path': b'/plugins/jev-decison/infer',
                 'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                 'client': ('127.0.0.1', 1), 'server': ('test', 80), 'root_path': ''}
        await asyncio.wait_for(app(scope, receive, send), timeout=2)
        assert backend.cancelled == 1 and backend.active == 0
        assert received[0]['status'] == 499
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['auto', 'generate'])
def test_generative_modes_rejected(mode):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        request({'type': 'boolean'}, mode=mode)
    with pytest.raises(ValueError, match='Only classification'):
        plan({'type': 'boolean'}, mode)


def test_mixed_schema_is_rejected_before_any_model_call():
    backend = Backend()
    schema = closed({'flag': {'type': 'boolean'}, 'free_text': {'type': 'string'}})
    with pytest.raises(ValueError, match='finite'):
        asyncio.run(DecisionService(backend).decide(request(schema)))
    assert backend.peak == 0


def test_backends_do_not_expose_generation():
    from vllm_jev_decison.backends import HTTPBackend
    assert not hasattr(VLLMBackend, 'generate')
    assert not hasattr(HTTPBackend, 'generate')


def test_api_rejects_nonclassification_requests_without_inference():
    async def run():
        app = FastAPI()
        DecisionPlugin().attach_router(app)
        backend = Backend()
        app.state.decision_service = DecisionService(backend)
        app.state.decision_keys = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            capabilities = (await client.get('/plugins/jev-decison/capabilities')).json()
            assert capabilities['modes'] == ['classify'] and capabilities['free_form_generation'] is False
            for extra in ({'mode': 'auto'}, {'mode': 'generate'}, {'max_tokens': 100}, {'schema': {'type': 'string'}}):
                payload = {'state': 'hello', 'schema': {'type': 'boolean'}, **extra}
                response = await client.post('/plugins/jev-decison/infer', json=payload)
                assert response.status_code == 422
        assert backend.peak == 0
    asyncio.run(asyncio.wait_for(run(), timeout=3))


def test_unusable_backend_disables_routes_without_stopping_the_server():
    async def run():
        app = FastAPI()
        plugin = DecisionPlugin()
        plugin.attach_router(app)
        # A generation engine is required; init_state must report, not raise.
        await plugin.init_state(None, app.state, SimpleNamespace(api_key=None))
        assert app.state.decision_service is None
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            for response in (await client.get('/plugins/jev-decison/capabilities'),
                             await client.post('/plugins/jev-decison/infer',
                                               json={'state': 'x', 'schema': {'type': 'boolean'}})):
                assert response.status_code == 503
                assert 'generation engine' in response.json()['detail']
    asyncio.run(run())


@pytest.mark.parametrize('value,expected', [('16', 16), ('0', 8), ('999', 8), ('nonsense', 8), (None, 8)])
def test_concurrency_reads_environment_and_rejects_out_of_range(monkeypatch, value, expected):
    monkeypatch.delenv('JEV_DECISON_CONCURRENCY', raising=False)
    if value is not None:
        monkeypatch.setenv('JEV_DECISON_CONCURRENCY', value)
    service = DecisionService(Backend())
    assert service.concurrency == expected
    assert service.gate._value == expected


def test_capabilities_reports_effective_limits(monkeypatch):
    async def run():
        monkeypatch.setenv('JEV_DECISON_TIMEOUT', '42')
        app = FastAPI()
        DecisionPlugin().attach_router(app)
        app.state.decision_service = DecisionService(Backend())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            body = (await client.get('/plugins/jev-decison/capabilities')).json()
        assert body['timeout_seconds'] == 42 and body['concurrency'] == 8
    asyncio.run(run())


def closed_object(count):
    properties = {f'f{i}': {'type': 'boolean'} for i in range(count)}
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def nested_object(depth):
    schema = {'type': 'boolean'}
    for _ in range(depth):
        schema = {'type': 'object', 'properties': {'x': schema}, 'required': ['x'],
                  'additionalProperties': False}
    return schema


@pytest.mark.parametrize('schema,fields', [
    (closed_object(32), 32),
    ({'enum': [f'v{i}' for i in range(16)]}, 1),
    ({'type': 'integer', 'minimum': 0, 'maximum': 15}, 1),
    (nested_object(9), 1),
])
def test_documented_limits_are_accepted(schema, fields):
    assert len(plan(schema)) == fields


@pytest.mark.parametrize('schema', [
    closed_object(33),
    {'enum': [f'v{i}' for i in range(17)]},
    {'type': 'integer', 'minimum': 0, 'maximum': 16},
    nested_object(10),
    {'enum': ['x' * 3000 for _ in range(16)]},
])
def test_beyond_documented_limits_is_rejected(schema):
    with pytest.raises(ValueError):
        plan(schema)


def test_duplicate_enum_values_do_not_split_probability():
    # Three declared values, one repeated: two labels would share one value's mass.
    assert plan({'enum': ['a', 'a', 'b']})[0].choices == ['a', 'b']


def test_json_equality_keeps_true_and_one_apart():
    assert plan({'enum': [True, 1]})[0].choices == [True, 1]


@pytest.mark.parametrize('schema,choices', [
    ({'type': 'integer', 'minimum': 0, 'maximum': 10, 'exclusiveMaximum': 5}, [0, 1, 2, 3, 4]),
    ({'type': 'integer', 'minimum': 0, 'maximum': 10, 'multipleOf': 3}, [0, 3, 6, 9]),
    ({'type': 'integer', 'minimum': -3, 'maximum': 2}, [-3, -2, -1, 0, 1, 2]),
])
def test_joint_numeric_constraints_narrow_the_candidate_set(schema, choices):
    assert plan(schema)[0].choices == choices
