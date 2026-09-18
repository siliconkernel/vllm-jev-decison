"""Official vLLM EndpointPlugin entry point; no engine source mutation."""
import asyncio
import hmac
import logging
import os

from fastapi import APIRouter, FastAPI, HTTPException, Request
from jsonschema.exceptions import SchemaError, ValidationError

from .backends import BackendError, VLLMBackend
from .service import DecisionRequest, DecisionService

PREFIX = '/plugins/jev-decison'
logger = logging.getLogger(__name__)


def attach_routes(app):
    router = APIRouter(prefix=PREFIX)

    def require_service(request):
        service = getattr(request.app.state, 'decision_service', None)
        if service is None:
            raise HTTPException(503, getattr(request.app.state, 'decision_error', None)
                                or 'Decision engine is not initialized')
        return service

    def authorize(request):
        keys = getattr(request.app.state, 'decision_keys', ())
        supplied = request.headers.get('authorization', '')
        if keys and not any(hmac.compare_digest(supplied.encode(), ('Bearer ' + key).encode()) for key in keys):
            raise HTTPException(401, 'Invalid or missing API key', headers={'WWW-Authenticate': 'Bearer'})

    @router.get('/capabilities')
    async def capabilities(request: Request):
        authorize(request)
        service = require_service(request)
        return {'backend': service.backend.name, 'model': service.backend.model, 'max_choices': 16,
                'max_fields': 32, 'concurrency': service.concurrency, 'timeout_seconds': service.timeout,
                'modes': ['classify'], 'free_form_generation': False,
                'schema_draft': '2020-12', 'schema_refs': False, 'input': 'text',
                'probabilities': 'uncalibrated conditional label probabilities',
                'classification': 'one engine output token per nonconstant finite field; normal sampler still used',
                'kv': 'ordinary vLLM prefix caching if configured; no retained-session fusion'}

    @router.post('/infer')
    async def infer(body: DecisionRequest, request: Request):
        authorize(request)
        service = require_service(request)
        async def disconnect():
            while True:
                message = await request.receive()
                if message['type'] == 'http.disconnect':
                    return
        task = asyncio.create_task(service.decide(body))
        disconnected = asyncio.create_task(disconnect())
        try:
            done, _ = await asyncio.wait([task, disconnected], return_when=asyncio.FIRST_COMPLETED)
            if task not in done:
                raise HTTPException(499, 'Client disconnected')
            return task.result()
        except (ValueError, SchemaError, ValidationError) as error:
            raise HTTPException(422, str(error)[:1000]) from error
        except TimeoutError as error:
            raise HTTPException(504, 'Decision request timed out') from error
        except BackendError as error:
            raise HTTPException(502, str(error)) from error
        finally:
            for pending in (task, disconnected):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, disconnected, return_exceptions=True)

    app.include_router(router)


class DecisionPlugin:
    name = 'jev-decison'
    required_tasks = ('generate',)

    def attach_router(self, app):
        attach_routes(app)

    async def init_state(self, engine_client, state, args):
        keys = getattr(args, 'api_key', None) or os.environ.get('VLLM_API_KEY') or []
        state.decision_keys = [keys] if isinstance(keys, str) else list(keys)
        state.decision_service = None
        state.decision_error = None
        try:
            state.decision_service = DecisionService(VLLMBackend(engine_client, args))
        except BackendError as error:
            # This endpoint is an add-on; an unusable backend disables its routes
            # rather than preventing the API server from starting.
            state.decision_error = str(error)
            logger.error('jev-decison endpoint disabled: %s', error)
