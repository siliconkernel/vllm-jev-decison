"""Conservative finite-domain planning; unknown constraints stay with generation."""
from dataclasses import dataclass
import json
from typing import Any

from jsonschema import Draft202012Validator

MAX_CHOICES = 16
MAX_FIELDS = 32
ANNOTATIONS = {'title', 'description', '$comment', 'default', 'examples', '$schema'}


@dataclass
class Field:
    path: tuple[str, ...]
    schema: dict
    choices: list[Any] | None


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def validate_schema(schema):
    encoded = json.dumps(schema, allow_nan=False)
    if len(encoded.encode()) > 32_000:
        raise ValueError('Schema exceeds 32 KB')
    def walk(value, depth=0):
        if depth > 20:
            raise ValueError('Schema nesting exceeds 20 levels')
        if isinstance(value, dict):
            if any(key in value for key in ('$ref', '$dynamicRef', '$recursiveRef')):
                raise ValueError('Schema references are not supported in v0.1')
            if '$schema' in value and value['$schema'] != 'https://json-schema.org/draft/2020-12/schema':
                raise ValueError('Only JSON Schema Draft 2020-12 is supported')
            for child in value.values():
                walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth + 1)
    walk(schema)
    Draft202012Validator.check_schema(schema)


def domain(schema):
    validator = Draft202012Validator(schema)
    if 'const' in schema:
        choices = [schema['const']]
    elif 'enum' in schema:
        choices = schema['enum']
    elif schema.get('type') == 'boolean':
        choices = [False, True]
    elif schema.get('type') == 'null':
        choices = [None]
    elif schema.get('type') == 'integer' and type(schema.get('minimum')) is int and type(schema.get('maximum')) is int:
        lo, hi = schema['minimum'], schema['maximum']
        if hi - lo + 1 > MAX_CHOICES or hi < lo:
            return None
        choices = list(range(lo, hi + 1))
    else:
        return None
    choices = [value for value in choices if validator.is_valid(value)]
    if not choices:
        raise ValueError('Finite schema has no valid candidates')
    return choices if len(choices) <= MAX_CHOICES else None


def plan(schema, mode='auto'):
    validate_schema(schema)
    fields = []
    def visit(node, path):
        candidates = domain(node)
        if candidates is not None and mode != 'generate':
            fields.append(Field(path, node, candidates))
            return
        simple = set(node) <= ANNOTATIONS | {'type', 'properties', 'required', 'additionalProperties'}
        properties = node.get('properties', {})
        required = node.get('required', [])
        closed = node.get('additionalProperties') is False and set(required) == set(properties)
        if mode != 'generate' and node.get('type') == 'object' and simple and closed and properties:
            for name, child in properties.items():
                if not isinstance(child, dict):
                    raise ValueError('Boolean property schemas are not supported')
                visit(child, path + (name,))
        else:
            if mode == 'classify':
                raise ValueError('Classification-only mode needs finite fields in required, closed objects')
            fields.append(Field(path, node, None))
    visit(schema, ())
    if len(fields) > MAX_FIELDS:
        raise ValueError(f'Schema requires more than {MAX_FIELDS} inference fields')
    return fields


def assemble(fields, values):
    if len(fields) == 1 and not fields[0].path:
        return values[0]
    result = {}
    for field, value in zip(fields, values, strict=True):
        parent = result
        for part in field.path[:-1]:
            parent = parent.setdefault(part, {})
        parent[field.path[-1]] = value
    return result
