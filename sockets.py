from datetime import datetime
from queue import Empty, Queue
import traceback
from typing import Any, Callable, Dict
from uuid import UUID

from flask import Flask, request
from flask_login import current_user
from flask_socketio import join_room, SocketIO # pylint: disable=redefined-builtin
import orjson

from constants import AIINTERFACE, DB, LOGGER
from db import User
from utils import (
    SessionData,
    add_session_data,
    get_session_data,
    pop_session_data,
    socket_login_required
)


current_user: User
logger = LOGGER.getChild("sockets")
# TODO: [Pipeline config]
# TODO: [TTS] Add a processor to handle multispeaker models


# TODO: Add message history to thing
@socket_login_required
def handle_connect():
    session_data = get_session_data()
    if session_data is None:
        add_session_data(SessionData(current_user.id, { request.sid }))
    else:
        session_data.sessions.add(request.sid)

    # Join a room whose id is the same as the user id to make emitting events easier
    join_room(current_user.id)

    logger.info("Socket with session id `%s` connected.", request.sid)


def handle_disconnect():
    # This HAS to run for the session id tracking to work
    session_data = get_session_data()
    if len(session_data.sessions) == 1:
        pop_session_data()
    else:
        session_data.sessions.remove(request.sid)

    logger.info("Socket with session id `%s` disconnected.", request.sid)


@socket_login_required
def handle_mic_packet(data):
    if not isinstance(data, bytes):
        return "Mic packet data should be bytes not string.", 2

    # TODO: [asr] Implement some kind of stop packet to force transcribe what's in the buffer
    AIINTERFACE.send_voice_data(current_user.id, data)


pipeline_complete_queue = Queue()
def poll_pipeline_loop(app: Flask) -> Callable:
    AIINTERFACE.callback = pipeline_complete_queue.put
    def wrapper() -> None:
        socket: SocketIO = app.extensions["socketio"]
        while True:
            try:
                try:
                    # NOTE: I'm not sure it's possible to avoid this weird sleep. Creating a queue
                    # using EngineIO.create_queue will make a queue that never releases its waiters
                    # even if an item is added to the queue, and calling a blocking call on this
                    # thread will completely stop the entire main thread.
                    payload: bytes = pipeline_complete_queue.get_nowait()
                except Empty:
                    socket.sleep(0.05)
                    continue

                # Decode payload
                if payload[0] == 8:
                    opcode = 8
                    user_id: str = payload[1:33]
                    data: memoryview = memoryview(payload[33:])
                else:
                    event = orjson.loads(payload)
                    opcode: int = event["op"]
                    user_id: str = event["id"]
                    timestamp: str = event["timestamp"]
                    data: Any = event["data"]

                # No need to read data in these opcodes, just passthrough to the client
                if opcode in (5, 8, 9):
                    logger.debug("Received %i. Sending...", opcode)
                    socket.emit(event, data, to=user_id)

                # Finished generating AI response and TTS data. This event is called per sentence of
                # the AI response, so it may be called multiple times after `start_gen` is.
                elif opcode == 7:
                    logger.debug("Received generated text. Sending...")
                    _on_finish_gen(timestamp, data, user_id, socket)

                # Finished transcribing user's mic data
                elif opcode == 6:
                    logger.debug("Received transcribed text. Sending...")
                    _on_finish_asr(timestamp, data, user_id, socket)
                else:
                    raise ValueError(f"Op `{opcode}` is not supported by the pipeline poll loop.")

            except Exception: # pylint: disable=broad-exception-caught
                logger.error(
                    "Exception while sending message to user:\n%s", 
                    traceback.format_exc()
                )
    return wrapper

def _on_finish_gen(
    timestamp: datetime,
    data: Dict[str, Any],
    user_id: SessionData,
    socket: SocketIO
) -> None:
    # Convert expression dicts into expression protobuf objects
    expressions = []
    for start_time, visemes in data["expressions"]:
        if not isinstance(visemes, list):
            visemes = [visemes]
        expressions.append({ "visemes": visemes, "start_time": start_time })

    # TODO: [User Management] Temp!!!
    message = DB.send_message(
        UUID("10a8827c-9187-4dc6-9c48-c883af32ebc4"),
        UUID("972878d5-3e81-490a-a19a-69c22f572160"),
        data["text"],
        created_at=timestamp
    )
    socket_event = {
        "message": {
            "id": str(message.id),
            "user_id": str(message.user_id),
            "conversation_id": str(message.conversation_id),
            "content": message.content,
            "created_at": message.created_at
        },
        "expressions": expressions,
        "wav_id": data["wav_id"]
    }

    socket.emit(7, orjson.dumps(socket_event), to=user_id)

def _on_finish_asr(
    timestamp: datetime,
    data: str,
    user_id: str,
    socket: SocketIO
) -> None:
    if data == "" or data.isspace():
        socket.emit(6, "{}", to=user_id)
        return

    # TODO: [User Management] Temp!!!
    message = DB.send_message(
        UUID(user_id),
        UUID("972878d5-3e81-490a-a19a-69c22f572160"),
        data,
        created_at=timestamp
    )
    socket_event = {
        "message": {
            "id": str(message.id),
            "user_id": str(message.user_id),
            "conversation_id": str(message.conversation_id),
            "content": message.content,
            "created_at": message.created_at
        }
    }

    socket.emit(6, orjson.dumps(socket_event), to=user_id)
