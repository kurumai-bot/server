from logging import Logger
import logging
from multiprocessing.connection import Client, Connection
from threading import Thread
from uuid import UUID

import orjson

from db import ModelPreset


class AIInterface:
    def __init__(self, host: str, port: int, authkey: bytes, logger: Logger = None) -> None:
        self.host = host
        self.port = port
        self.authkey = authkey
        self.logger = logger or logging.getLogger("AI Interface")

        self._client: Connection
        self._thread = Thread(target=self._thread_loop)
        self._run_loop = True

    def start(self) -> None:
        self._client = Client((self.host, self.port), authkey=self.authkey)
        self._thread.start()

    def __enter__(self) -> "AIInterface":
        self.start()
        return self

    def close(self) -> None:
        self._thread.join()
        self._run_loop = False
        self._client.close()

    def __exit__(self, *args) -> None:
        self.close()

    def set_preset(self, user_id: UUID, preset: ModelPreset):
        self._client.send_bytes(orjson.dumps({
            "op": 1,
            "id": user_id,
            "preset": preset.to_dict(),
        }))

    def remove_preset(self, user_id: UUID):
        self._client.send_bytes(orjson.dumps({
            "op": 2,
            "id": user_id
        }))

    def send_voice_data(self, user_id: UUID, data: bytes):
        self._client.send_bytes(bytes(3) + user_id.bytes + data)

    def send_text_data(self, user_id: UUID, data: str):
        self._client.send_bytes(orjson.dumps({
            "op": 4,
            "id": user_id,
            "data": data
        }))

    def _thread_loop(self):
        while True:
            if not self._run_loop:
                break

            if self._client.poll(1):
                print(self._client.recv_bytes())

if __name__ == "__main__":
    from constants import AIINTERFACE, DB
    AIINTERFACE.set_preset(UUID("10a8827c-9187-4dc6-9c48-c883af32ebc4"), DB.get_user_model_presets("10a8827c-9187-4dc6-9c48-c883af32ebc4")[0])
    AIINTERFACE._thread.join()
