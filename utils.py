from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import signal
from threading import Thread
import time
import traceback
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING, Type, TypeVar
from uuid import UUID, uuid4

from flask import abort, current_app, Flask, session
from flask.json.provider import JSONProvider
from flask.views import View
from flask_login import UserMixin, current_user
from flask_socketio import disconnect
from google.protobuf.timestamp_pb2 import Timestamp # pylint: disable=no-name-in-module
import numpy as np
import orjson

if TYPE_CHECKING:
    from ai import Pipeline
    from db import DatabaseObject


current_user: UserMixin


class StdOutRedirect():
    def __init__(self, consumer: Callable[[Any], Any]) -> None:
        self.consumer = consumer

    def write(self, text) -> int:
        if text != "\n":
            self.consumer(text)

    def flush(self) -> None:
        pass


class CircularBuffer:
    def __init__(self, size: int, **kwargs) -> None:
        self.size = size
        self.dtype: np.dtype = kwargs.get("dtype", np.float32)
        self.index = 0
        self.buffer = np.empty(self.size, dtype=self.dtype)

    def add(self, data: np.ndarray) -> np.ndarray:
        # Number of return values will be current length of the buffer plus length of data
        # floor divided by buffer capacity
        ret = np.empty(((self.index + len(data)) // self.size, self.size))
        data_index = 0

        if len(ret) > 0:
            # Fill first ret value with a combination of current buffer data and passed in data
            ret[0][:self.index] = self.buffer[:self.index]
            data_index = self.size - self.index
            ret[0][self.index:] = data[:data_index]
            self.index = 0

            # Fill remaining buffers with however much data they can hold
            for i in range(1, len(ret)):
                ret[i] = data[data_index:data_index + self.size]
                data_index += self.size

        # Fill buffer with remaining data in passed in data
        start = self.index
        self.index = start + len(data) - data_index
        self.buffer[start:self.index] = data[data_index:]

        return ret

    def clear(self) -> None:
        self.index = 0

    def get(self) -> np.ndarray:
        return self.buffer[:self.index]


class ORJSONProvider(JSONProvider):
    def loads(self, s, **kwargs):
        return orjson.loads(s)

    def dumps(self, obj, **kwargs):
        return orjson.dumps(obj).decode('utf-8')


T = TypeVar("T")


def abort_if_none(value: T | None) -> T:
    if value is None:
        abort(404, description="Resource not found.")
    return value


# A simple cache that deletes entries after not being updated for a while
class Cache:
    def __init__(self, logger: logging.Logger = None) -> None:
        self.logger = logger or logging.getLogger("cache")
        self.items: Dict[str, Tuple[datetime, Any]] = {}

        self._thread = Thread(target=self._cache_loop)
        self._run_loop = True
        self._original_handler = signal.signal(signal.SIGINT, self._on_exit)

    def __contains__(self, val: Any) -> bool:
        return val in self.items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, indices: Any) -> Tuple[datetime, Any] | None:
        return self.items[indices]

    def get(self, key: str) -> Tuple[datetime, Any] | None:
        return self.items.get(key)

    def add(self, key: str, item: Any, ttl: timedelta = None):
        if ttl is None:
            ttl = timedelta(minutes=15)

        expire_time = datetime.utcnow() + ttl
        self.items[key] = (expire_time, item)
        self.logger.debug("Item with key `%s` added, set to expire at %s.", key, expire_time)

    def start(self) -> None:
        self._run_loop = True
        self._thread.start()

    def close(self) -> None:
        self._run_loop = False
        self._thread.join()

    def _on_exit(self, code, frame) -> None:
        self.close()
        self._original_handler(code, frame)

    def _cache_loop(self) -> None:
        while self._run_loop:
            try:
                if len(self.items) == 0:
                    time.sleep(1)
                    continue

                min_ = min(self.items.items(), key=lambda x: x[1][0])
                if min_[1][0] < datetime.utcnow():
                    # Remove expired items
                    self.items.pop(min_[0])
                    self.logger.debug("Removed expired item with key `%s`.", min_[0])

                time.sleep(1)
            except Exception: # pylint: disable=broad-exception-caught
                self.logger.error(
                    "Cache deletion loop failed with the following error:\n%s",
                    traceback.format_exc()
                )
        self.logger.info("Exited cache loop.")


def list_to_dict(items: List[DatabaseObject]) -> List[Dict]:
    return [item.to_dict() for item in items]


RT = TypeVar("RT")


def socket_login_required(func: Callable[..., RT]):
    def decorator(*args, **kwargs) -> RT | None:
        if current_app.config.get("LOGIN_DISABLED"):
            pass
        elif not current_user.is_authenticated:
            disconnect()
            return None

        return func(*args, **kwargs)

    return decorator


@dataclass
class SessionData:
    sid: str
    pipeline: Pipeline
    user_id: UUID

    def process_data(self, data: str | bytes):
        self.pipeline.process_input(
            data,
            datetime.utcnow(),
            callback_data=(self, current_app.extensions["socketio"], uuid4())
        )

def get_session_data() -> SessionData | None:
    # Session "_id" and "_user_id" are from Flask-Login and the combination of the two *should*
    # ensure key in this dict is a specific user on a specific computer
    return get_session_data_store().get(get_session_key(), None)

def get_session_data_store() -> Dict[Tuple[str, str], SessionData]:
    return current_app.extensions["user_session_ids"]

def get_session_key() -> Tuple[str, str]:
    return (session.get("_id"), session.get("_user_id"))


def add_url_rule_view(
    app: Flask,
    rule: str,
    endpoint: str | None = None,
    view: Type[View] | None = None,
    provide_automatic_options: bool | None = None,
    **options: Any
) -> None:
    return app.add_url_rule(
        rule,
        endpoint=endpoint,
        view_func=view.as_view(view.__name__),
        provide_automatic_options=provide_automatic_options,
        **options
    )


def datetime_to_timestamp(val: datetime) -> Timestamp:
    seconds = val.timestamp()
    nanos = (seconds % 1) * 1_000_000_000

    return Timestamp(seconds=int(seconds), nanos=int(nanos))
