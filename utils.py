from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from datetime import datetime, timedelta, timezone
import logging
from threading import Thread
import time
import traceback
from typing import Any, Callable, Dict, List, Set, Tuple, TYPE_CHECKING, Type, TypeVar
from uuid import UUID

from flask import abort, current_app, Flask, session
from flask.json.provider import JSONProvider
from flask.views import View
from flask_login import UserMixin, current_user
from flask_socketio import disconnect
import orjson


if TYPE_CHECKING:
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

        expire_time = datetime.now(timezone.utc) + ttl
        self.items[key] = (expire_time, item)
        self.logger.debug("Item with key `%s` added, set to expire at %s.", key, expire_time)

    def start(self) -> None:
        self._run_loop = True
        self._thread.start()

    def __enter__(self) -> Cache:
        self.start()
        return self

    def close(self) -> None:
        self._run_loop = False
        self._thread.join()

    def __exit__(self, *args):
        self.close()

    def _cache_loop(self) -> None:
        while self._run_loop:
            try:
                if len(self.items) == 0:
                    time.sleep(1)
                    continue

                min_ = min(self.items.items(), key=lambda x: x[1][0])
                if min_[1][0] < datetime.now(timezone.utc):
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
    user_id: UUID
    sessions: Set[str] = field(default_factory=set)

def get_session_data() -> SessionData | None:
    # Session "_id" and "_user_id" are from Flask-Login and the combination of the two *should*
    # ensure key in this dict is a specific user on a specific computer
    return get_session_data_store().get(session.get("_user_id"), {}).get(session.get("_id"))

def get_session_data_store() -> Dict[str, Dict[str, SessionData]]:
    return current_app.extensions["user_session_ids"]

def add_session_data(session_data: SessionData) -> None:
    store = get_session_data_store()
    user_id = session.get("_user_id")
    session_id = session.get("_id")

    user_sessions = store.get(user_id)
    if user_sessions is not None:
        user_sessions[session_id] = session_data
    else:
        user_sessions = { session_id: session_data }
        store[user_id] = user_sessions

def pop_session_data() -> SessionData:
    store = get_session_data_store()
    user_id = session.get("_user_id")
    session_id = session.get("_id")

    user_sessions = store[user_id]
    session_data = user_sessions[session_id]
    if len(user_sessions) == 1:
        store.pop(user_id)
    else:
        user_sessions.pop(session_id)
    return session_data

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
