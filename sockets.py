from datetime import datetime
from queue import Empty, Queue
import traceback
from typing import Any, Callable, Dict
from uuid import UUID

from TTS.api import TTS
from flask import request, session
from flask_login import current_user
from flask_socketio import ConnectionRefusedError, disconnect, join_room, SocketIO # pylint: disable=redefined-builtin
from google.protobuf.message import DecodeError

from ai import OpenAIChat, Pipeline, Whisper
from constants import DB, LOGGER, OPENAI_API_KEY
from db import User
from messages_pb2 import ( # pylint: disable=no-name-in-module
    Expression,
    Message,
    MicPacket,
    SocketEvent,
    StartMessage,
    TTSMessage,
    Viseme
)
from utils import (
    SessionData,
    add_session_data,
    datetime_to_timestamp,
    get_session_data,
    pop_session_data,
    socket_login_required
)


current_user: User
logger = LOGGER.getChild("sockets")
logger.info("Creating AI inferences.")
# TODO: [Pipeline config]
asr = Whisper("openai/whisper-base.en", device="cuda:0")
# TODO: [TTS] Add a processor to handle multispeaker models
tts = TTS("tts_models/en/vctk/vits", gpu=True)
text_gen = OpenAIChat("gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
logger.info("Finished creating inferences.")


# TODO: Add message history to thing
@socket_login_required
def handle_connect():
    session_data = get_session_data()
    if session_data is None:
        # TODO: [Pipeline config]
        def pipeline_callback(event: str, timestamp: datetime, data: Any, callback_data: Any):
            pipeline_complete_queue.put((event, timestamp, data, callback_data))
        pipeline = Pipeline(
            asr,
            tts,
            text_gen,
            pipeline_callback,
            tts_speaker_name="p300",
            logger=LOGGER.getChild("pipeline"),
            # TODO: [Logging] This may be too verbose
            asr_logger=LOGGER.getChild("pipeline.asr"),
            tts_logger=LOGGER.getChild("pipeline.tts"),
            text_gen_logger=LOGGER.getChild("pipeline.text_gen")
        )
        pipeline.start()

        add_session_data(SessionData(pipeline, current_user.id, { request.sid }))
    else:
        session_data.sessions.add(request.sid)

    # Join a room whose id is the same as the user id to make emitting events easier
    join_room(current_user.id)

    logger.info("Socket with session id `%s` connected.", request.sid)


def handle_disconnect():
    # This HAS to run for the session id tracking to work
    session_data = get_session_data()
    print(session_data.sessions)
    if len(session_data.sessions) == 1:
        pop_session_data()
        session_data.pipeline.stop()
    else:
        session_data.sessions.remove(request.sid)

    logger.info("Socket with session id `%s` disconnected.", request.sid)


@socket_login_required
def handle_mic_packet(data):
    if not isinstance(data, bytes):
        return "Mic packet data should be bytes not string.", 2

    try:
        mic_packet = MicPacket.FromString(data)
    except DecodeError:
        return "Mic packet data is in an incorrect format.", 2

    # TODO: [asr] Implement some kind of stop packet to force transcribe what's in the buffer
    get_session_data().process_data(mic_packet.data)


pipeline_complete_queue = Queue()
def poll_pipeline_loop(sleep: Callable[[float], Any]) -> Callable:
    def wrapper() -> None:
        while True:
            try:
                try:
                    # NOTE: I'm not sure it's possible to avoid this weird sleep. Creating a queue
                    # using EngineIO.create_queue will make a queue that never releases its waiters
                    # even if an item is added to the queue, and calling a blocking call on this
                    # thread will completely stop the entire main thread.
                    event, timestamp, data, (session_data, socket, id) \
                        = pipeline_complete_queue.get_nowait()
                except Empty:
                    sleep(0.05)
                    continue

                # Started either asr or AI generation. Either way, we just notify the client.
                if event == "start":
                    socket_event = SocketEvent(
                        event=event,
                        id=str(id),
                        start_message=StartMessage(type=data[0], details=data[1])
                    )
                    # Cant use the static emit function because this is not in a flask context
                    socket.emit(event, socket_event.SerializeToString(), to=session_data.user_id)

                # Finished generating AI response and TTS data. This event is called per sentence of
                # the AI response, so it may be called multiple times after `start_gen` is.
                elif event == "finish_gen":
                    logger.debug("Received generated text. Sending...")
                    _on_finish_gen(timestamp, data, session_data, socket, id)

                # Finished transcribing user's mic data
                elif event == "finish_asr":
                    logger.debug("Received transcribed text. Sending...")
                    _on_finish_asr(timestamp, data, session_data, socket, id)

                # Pipeline is finished
                elif event == "finish":
                    socket_event = SocketEvent(event="finish", id=str(id))
                    socket.emit(event, socket_event.SerializeToString(), to=session_data.user_id)

                else:
                    raise ValueError(f"Event `{event}` is not supported by the pipeline poll loop.")

            except Exception: # pylint: disable=broad-exception-caught
                logger.error(
                    "Exception while sending message to user:\n%s", 
                    traceback.format_exc()
                )
    return wrapper

def _on_finish_gen(
    timestamp: datetime,
    data: Dict[str, Any],
    session_data: SessionData,
    socket: SocketIO,
    id: UUID
) -> None:
    # Convert expression dicts into expression protobuf objects
    expressions = []
    for start_time, visemes in data["expressions"]:
        if isinstance(visemes, list):
            visemes = [
                Viseme(index=viseme.index, weight=viseme.weight)
                for viseme in visemes
            ]
        else:
            visemes = [Viseme(index=visemes.index, weight=visemes.weight)]
        expressions.append(Expression(visemes=visemes, start_time=start_time))

    # TODO: [User Management] Temp!!!
    message = DB.send_message(
        UUID("10a8827c-9187-4dc6-9c48-c883af32ebc4"),
        UUID("972878d5-3e81-490a-a19a-69c22f572160"),
        data["text"],
        created_at=timestamp
    )
    socket_event = SocketEvent(
        event="finish_gen",
        id=str(id),
        tts_message=TTSMessage(
            message=Message(
                id=str(message.id),
                user_id=str(message.user_id),
                conversation_id=str(message.conversation_id),
                content=message.content,
                created_at=datetime_to_timestamp(message.created_at)
            ),
            expressions=expressions,
            # TODO: Send more complex wav data than just the waveform, since the client is
            # guessing the samplerate
            data=data["wav"].tobytes()
        )
    )

    socket.emit("finish_gen", socket_event.SerializeToString(), to=session_data.user_id)

def _on_finish_asr(
    timestamp: datetime,
    data: str,
    session_data: SessionData,
    socket: SocketIO,
    id: UUID
) -> None:
    if data == "" or data.isspace():
        socket_event = SocketEvent(event="finish_asr", id=str(id))
        socket.emit("finish_asr", socket_event.SerializeToString(), to=session_data.user_id)
        return

    # TODO: [User Management] Temp!!!
    message = DB.send_message(
        session_data.user_id,
        UUID("972878d5-3e81-490a-a19a-69c22f572160"),
        data,
        created_at=timestamp
    )
    socket_event = SocketEvent(
        event="finish_asr",
        id=str(id),
        message=Message(
            id=str(message.id),
            user_id=str(message.user_id),
            conversation_id=str(message.conversation_id),
            content=message.content,
            created_at=datetime_to_timestamp(message.created_at)
        )
    )

    socket.emit("finish_asr", socket_event.SerializeToString(), to=session_data.user_id)
