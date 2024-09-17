"""
The protocol used in this module is like so:

The client tells the server which conversation it will send mic data to
The server then tells the ai service what the settings it wants are 

When the client send mic information
If the ai service never got settings:
    It will detect that it has no settings and send back an error to the server.
    The server then sends back an error to the client, but does not disconnect it.
If the ai server did get settings:
    It will process the mic information and send back information to the server.
    The server will convert that into messages for the DB and return that information to the user.
"""

from datetime import datetime
from queue import Empty, Queue
import traceback
from typing import Any, Callable, Dict
from uuid import UUID

from flask import Flask, request
from flask_login import current_user
from flask_socketio import join_room, SocketIO, leave_room
import orjson

from constants import AIINTERFACE, DB, LOGGER
from db import User
from utils import (
    add_conversation_to_session,
    add_session_data,
    get_session_data,
    remove_session_data,
    socket_login_required
)

current_user: User
logger = LOGGER.getChild("sockets")
# TODO: [Pipeline config]
# TODO: [TTS] Add a processor to handle multispeaker models


# TODO: Add message history to thing
@socket_login_required
def handle_connect():
    add_session_data(request.sid)

    # Join a room whose id is the same as the user id to make emitting events easier
    join_room(str(current_user.id))

    logger.info("Socket with session id `%s` connected.", request.sid)


def handle_disconnect():
    conversation_id = get_session_data().sessions[request.sid]
    # This HAS to run for the session id tracking to work
    remove_session_data(request.sid)

    AIINTERFACE.remove_preset(str(current_user.id) + str(conversation_id))

    logger.info("Socket with session id `%s` disconnected.", request.sid)


# TODO: Track when pipeline is removed from cache so we know to readd it
@socket_login_required
def handle_mic_packet(data):
    if not isinstance(data, bytes):
        return "Mic packet data should be bytes not string.", 2

    # TODO: [asr] Implement some kind of stop packet to force transcribe what's in the buffer
    conversation_id = get_session_data().sessions[request.sid]
    AIINTERFACE.send_voice_data(str(current_user.id) + str(conversation_id), data)


@socket_login_required
def handle_set_conversation(data):
    try:
        conversation_id = UUID(hex=data)
    except ValueError:
        return "Conversation UUID is malformed.", 2

    old_conversation_id = get_session_data().sessions[request.sid]
    if old_conversation_id is not None:
        old_bot_user = DB.get_conversation_bot_user(old_conversation_id)
        leave_room(old_bot_user.id)

    bot_user = DB.get_conversation_bot_user(conversation_id).get_bot_user()
    join_room(bot_user.id)
    add_conversation_to_session(request.sid, str(conversation_id))
    AIINTERFACE.set_preset(str(current_user.id) + str(conversation_id), bot_user)


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

                _on_payload(payload, socket)

            except Exception: # pylint: disable=broad-exception-caught
                logger.error(
                    "Exception while sending message to user:\n%s", 
                    traceback.format_exc()
                )
    return wrapper

def _on_payload(payload: bytes, socket: SocketIO):
    # Decode payload
    if payload[0] == 8:
        opcode = 8
        user_id: str = payload[1:37].decode()
        conversation_id: str = payload[37:73].decode()
        data: bytes = payload[74:]
    else:
        event = orjson.loads(payload)
        opcode: int = event["op"]
        user_id: str = event["id"][:36]
        conversation_id: str = event["id"][36:]
        timestamp: str = event["timestamp"]
        data: Any = event["data"]

    match opcode:
        case 0:
            logger.debug("Received error %i from ai service", data)
            socket.emit(
                str(0),
                { "conversation_id": conversation_id, "error_code": data },
                to=user_id
            )
        case 5:
            logger.debug("Received 5. Sending to %s", user_id)
            socket.emit(
                str(opcode),
                orjson.dumps(
                    { "type": data[0], "details": data[1], "conversation_id": conversation_id }
                ),
                to=user_id
            )
        case 9:
            logger.debug("Received 9. Sending to %s", user_id)
            socket.emit(
                str(opcode),
                orjson.dumps({ "conversation_id": conversation_id }),
                to=user_id
            )
        case 8:
            logger.debug("Received 8. Sending to %s", user_id)
            socket.emit(str(opcode), data, to=user_id)

        # Finished generating AI response and TTS data. This event is called per
        # sentence of the AI response, so it may be called multiple times after
        # `start_gen` is.
        case 7:
            logger.debug("Received generated text. Sending to %s", user_id)
            _on_finish_gen(timestamp, data, user_id, conversation_id, socket)

        # Finished transcribing user's mic data
        case 6:
            logger.debug("Received transcribed text. Sending to %s", user_id)
            _on_finish_asr(timestamp, data, user_id, conversation_id, socket)
        case _:
            raise ValueError(f"Op `{opcode}` is not supported by the pipeline poll loop.")

def _on_finish_gen(
    timestamp: datetime,
    data: Dict[str, Any],
    user_id: str,
    conversation_id: str,
    socket: SocketIO
) -> None:
    # Convert expression dicts into expression protobuf objects
    expressions = []
    for start_time, visemes in data["expressions"]:
        if not isinstance(visemes, list):
            visemes = [visemes]
        expressions.append({ "visemes": visemes, "start_time": start_time })

    bot_user = DB.get_conversation_bot_user(UUID(conversation_id))
    message = DB.send_message(
        bot_user.id,
        UUID(conversation_id),
        data["text"],
        created_at=timestamp
    )
    socket_event = {
        "message": message.to_dict(),
        "expressions": expressions,
        "emotion": data["emotion"],
        "wav_id": data["wav_id"],
        "conversation_id": conversation_id
    }

    socket.emit(str(7), orjson.dumps(socket_event), to=user_id)

def _on_finish_asr(
    timestamp: datetime,
    data: str,
    user_id: str,
    conversation_id: str,
    socket: SocketIO
) -> None:
    if data == "" or data.isspace():
        socket.emit(str(6), "{}", to=user_id)
        return

    message = DB.send_message(
        UUID(user_id),
        UUID(conversation_id),
        data,
        created_at=timestamp
    )
    socket_event = {
        "message": message.to_dict(),
        "conversation_id": conversation_id
    }

    socket.emit(str(6), orjson.dumps(socket_event), to=user_id)
