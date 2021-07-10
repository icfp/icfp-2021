import json

from pydantic.json import pydantic_encoder


def to_json(model) -> str:
    return json.dumps(model, default=pydantic_encoder)
