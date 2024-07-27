from logging import Logger
import logging
from multiprocessing.connection import Client, Connection
from threading import Thread
import time
import traceback
from typing import Any, Callable
from uuid import UUID

import orjson

from db import ModelPreset


class AIInterface:
    def __init__(
        self,
        host: str,
        port: int,
        authkey: bytes,
        callback: Callable[[bytes], Any],
        logger: Logger = None
    ) -> None:
        self.host = host
        self.port = port
        self.authkey = authkey
        self.callback = callback
        self.logger = logger or logging.getLogger("AI Interface")

        self._client: Connection = None
        self._thread = Thread(target=self._thread_loop)
        self._run_loop = True

    def start(self) -> None:
        self._try_create_client()
        self._thread.start()

    def _try_create_client(self) -> None:
        try:
            self._client = Client((self.host, self.port), authkey=self.authkey)
            self.logger.info("Connected to target address")
        except ConnectionRefusedError:
            self.logger.warning("Interface could not connect to target address")

    def __enter__(self) -> "AIInterface":
        self.start()
        return self

    def close(self) -> None:
        self._run_loop = False
        self._thread.join()
        if self._client is not None:
            self._client.close()

    def __exit__(self, *args) -> None:
        self.close()

    def set_preset(self, user_id: UUID, preset: ModelPreset):
        if self._check_client_is_none():
            return

        self._client.send_bytes(orjson.dumps({
            "op": 1,
            "id": str(user_id),
            "preset": preset.to_dict(),
        }))

    def remove_preset(self, user_id: UUID):
        if self._check_client_is_none():
            return

        self._client.send_bytes(orjson.dumps({
            "op": 2,
            "id": str(user_id)
        }))

    def send_voice_data(self, user_id: UUID, data: bytes):
        if self._check_client_is_none():
            return

        payload = bytearray(1 + 32 + len(data))
        payload[0] = 3
        payload[1:1 + 32] = user_id.bytes
        payload[1 + 32:] = data
        self._client.send_bytes(bytes(payload))

    def send_text_data(self, user_id: UUID, data: str):
        if self._check_client_is_none():
            return
        self._client.send_bytes(orjson.dumps({
            "op": 4,
            "id": str(user_id),
            "data": data
        }))

    def _check_client_is_none(self):
        if self._client is None:
            self.logger.warning("Attempted to use client while it is none")
            return True
        return False

    def _thread_loop(self):
        while True:
            try:
                if not self._run_loop:
                    return

                if self._client is None or self._client.closed:
                    self._try_create_client()
                    time.sleep(1)
                    continue

                if not self._client.poll(1):
                    continue

                try:
                    payload = self._client.recv_bytes()
                except (EOFError, OSError):
                    self._client.close()
                    self._client = None

                try:
                    callback = self.callback
                    callback(payload)
                except Exception: # pylint: disable=broad-exception-caught
                    self.logger.error("Error during callback:\n%s", traceback.format_exc())

            except Exception: # pylint: disable=broad-exception-caught
                self.logger.error("Exception in interface loop:\n%s",traceback.format_exc())


if __name__ == "__main__":
    from constants import AIINTERFACE, DB
    try:
        AIINTERFACE._thread.join()
    except KeyboardInterrupt:
        AIINTERFACE.close()
        DB.close()
