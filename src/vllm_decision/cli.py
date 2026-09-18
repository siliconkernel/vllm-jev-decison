import argparse
from importlib.metadata import entry_points, version, PackageNotFoundError
import json
import os


def doctor():
    info = {'package': version('vllm-decision'), 'endpoint_entrypoint': any(
        entry.name == 'decision' for entry in entry_points(group='vllm.endpoint_plugins'))}
    try:
        info['vllm_version'] = version('vllm')
        from vllm.plugins.endpoint_plugins.interface import EndpointPlugin
        from .plugin import DecisionPlugin
        info['endpoint_protocol'] = isinstance(DecisionPlugin(), EndpointPlugin)
    except (ImportError, PackageNotFoundError) as error:
        info['endpoint_protocol'] = False
        info['detail'] = f'{type(error).__name__}: install a compatible vLLM; the HTTP bridge is separate.'
    print(json.dumps(info, indent=2))
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
    app = FastAPI(title='vllm-decision HTTP bridge (not the in-process plugin)', lifespan=lifespan)
    app.state.decision_keys = [api_key] if api_key else []
    app.state.decision_service = DecisionService(backend)
    attach_routes(app)
    return app


def main():
    parser = argparse.ArgumentParser(description='vLLM typed decisions plugin utilities')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Check local vLLM endpoint-plugin availability')
    bridge = sub.add_parser('bridge', help='Optional HTTP bridge for an existing vLLM server; no plugin deployment')
    bridge.add_argument('--upstream', required=True)
    bridge.add_argument('--model', required=True)
    bridge.add_argument('--host', default='127.0.0.1')
    bridge.add_argument('--port', type=int, default=18186)
    args = parser.parse_args()
    if args.command == 'doctor':
        raise SystemExit(doctor())
    import uvicorn
    uvicorn.run(bridge_app(args.upstream, args.model, os.environ.get('DECISION_UPSTREAM_API_KEY'),
                           os.environ.get('DECISION_API_KEY')), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
