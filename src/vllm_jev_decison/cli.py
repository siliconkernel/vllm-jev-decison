import argparse
from importlib.metadata import entry_points, version, PackageNotFoundError
import json
import os

from .service import LABELS


def check_labels_for(model):
    """Candidate labels must be distinct single tokens, or the backend cannot start."""
    from .backends import BackendError, check_labels
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        ids = check_labels([tokenizer.encode(label, add_special_tokens=False) for label in LABELS])
    except BackendError as error:
        return {'model': model, 'single_token_labels': False, 'detail': str(error)}
    except Exception as error:
        return {'model': model, 'single_token_labels': None,
                'detail': f'{type(error).__name__}: {error}'}
    return {'model': model, 'single_token_labels': True, 'label_token_ids': ids}


def doctor(model=None):
    info = {'package': version('vllm-jev-decison'), 'endpoint_entrypoint': any(
        entry.name == 'jev-decison' for entry in entry_points(group='vllm.endpoint_plugins'))}
    try:
        info['vllm_version'] = version('vllm')
        from vllm.plugins.endpoint_plugins.interface import EndpointPlugin
        from .plugin import DecisionPlugin
        info['endpoint_protocol'] = isinstance(DecisionPlugin(), EndpointPlugin)
    except (ImportError, PackageNotFoundError) as error:
        info['endpoint_protocol'] = False
        info['detail'] = f'{type(error).__name__}: install a compatible vLLM; the HTTP bridge is separate.'
    if model is not None:
        info['labels'] = check_labels_for(model)
    print(json.dumps(info, indent=2))
    if model is not None and info['labels']['single_token_labels'] is not True:
        return 1
    return 0 if info.get('endpoint_protocol') and info['endpoint_entrypoint'] else 1


def bridge_app(url, model, upstream_key=None, api_key=None):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from .backends import HTTPBackend
    from .plugin import attach_routes
    from .service import DecisionService
    backend = HTTPBackend(url, model, upstream_key)
    @asynccontextmanager
    async def lifespan(app):
        yield
        await backend.close()
    app = FastAPI(title='vllm-jev-decison HTTP bridge (not the in-process plugin)', lifespan=lifespan)
    app.state.decision_keys = [api_key] if api_key else []
    app.state.decision_service = DecisionService(backend)
    attach_routes(app)
    return app


def main():
    parser = argparse.ArgumentParser(description='vLLM typed decisions plugin utilities')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('doctor', help='Check local vLLM endpoint-plugin availability')
    check.add_argument('--model', default=None,
                       help='Also verify this model encodes candidate labels A-P as distinct single tokens')
    bridge = sub.add_parser('bridge', help='Optional HTTP bridge for an existing vLLM server; no plugin deployment')
    bridge.add_argument('--upstream', required=True)
    bridge.add_argument('--model', required=True)
    bridge.add_argument('--host', default='127.0.0.1')
    bridge.add_argument('--port', type=int, default=18186)
    args = parser.parse_args()
    if args.command == 'doctor':
        raise SystemExit(doctor(args.model))
    import uvicorn
    uvicorn.run(bridge_app(args.upstream, args.model, os.environ.get('JEV_DECISON_UPSTREAM_API_KEY'),
                           os.environ.get('JEV_DECISON_API_KEY')), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
