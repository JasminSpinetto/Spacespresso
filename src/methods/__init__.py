from __future__ import annotations

import importlib
from typing import Type

from src.methods.base import BaseMethod


def get_method_class(name: str) -> Type[BaseMethod]:
    module = importlib.import_module(f"src.methods.{name}")
    method_cls = getattr(module, "Method", None)
    if method_cls is None:
        raise ValueError(f"Method module 'src.methods.{name}' does not define Method")
    return method_cls


__all__ = ["BaseMethod", "get_method_class"]

