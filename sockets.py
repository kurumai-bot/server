from datetime import datetime
from queue import Empty, Queue
import traceback
from typing import Any, Callable, Dict
from uuid import UUID

from TTS.api import TTS
from flask import request, session
from flask_login import current_user
from flask_socketio import ConnectionRefusedError, disconnect, SocketIO # pylint: disable=redefined-builtin
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
    datetime_to_timestamp,
    get_session_data,
    get_session_data_store,
    get_session_key,
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


# TODO: Figure out a way to let other clients with the same user know if the user sent a message
# TODO: Add message history to thing
@socket_login_required
def handle_connect():
    # If user already a session then this probably a duplicate so force disconnect them
    # This relies on the disconnect event calling reliably, so let's hope it does
    if get_session_data() is not None:
        logger.info(
            "User with id `%s` attemped to create 2 socketio connections.",
            session.get("_id")
        )

        raise ConnectionRefusedError(
            "A user on the same computer already has socketio instance connected."
        )

    # TODO: [Pipeline config]
    sid = request.sid
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
    get_session_data_store().setdefault(
        get_session_key(),
        SessionData(sid, pipeline, current_user.id)
    )

    logger.info("Socket with session id `%s` connected.", sid)


def handle_disconnect():
    # This HAS to run for the session id tracking to work
    session_ids = get_session_data_store()
    session_data = session_ids.pop(get_session_key(), None)
    if session_data is not None:
        session_data.pipeline.stop()

    logger.info("Socket with session id `%s` disconnected.", request.sid)


@socket_login_required
def handle_mic_packet(data):
    # This in theory shouldn't ever run, but checking doesn't hurt
    session_data = get_session_data()
    if session_data.sid != request.sid:
        disconnect()
        logger.info(
            "Socket cookies on user `%s` mismatched with cached sid",
            session.get("_user_id")
        )
        return None

    if not isinstance(data, bytes):
        return "Mic packet data should be bytes not string.", 2

    try:
        mic_packet = MicPacket.FromString(data)
    except DecodeError:
        return "Mic packet data is in an incorrect format.", 2

    # TODO: [asr] Implement some kind of stop packet to force transcribe what's in the buffer
    session_data.process_data(mic_packet.data)


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
                    socket.emit(event, socket_event.SerializeToString(), to=session_data.sid)

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
                    socket.emit(event, socket_event.SerializeToString(), to=session_data.sid)

                else:
                    raise ValueError(f"Event `{event}` is not supported by the pipeline poll loop.")

            except Exception: # pylint: disable=broad-exception-caught
                logger.error(
                    "Exception while sending message to user:\n%s", 
                    traceback.format_exc()
                )
    return wrapper

# TODO: [generation] Change this function once streaming is supported
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

    socket.emit("finish_gen", socket_event.SerializeToString(), to=session_data.sid)

def _on_finish_asr(
    timestamp: datetime,
    data: str,
    session_data: SessionData,
    socket: SocketIO,
    id: UUID
) -> None:
    if data == "" or data.isspace():
        socket_event = SocketEvent(event="finish_asr", id=str(id))
        socket.emit("finish_asr", socket_event.SerializeToString(), to=session_data.sid)
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

    socket.emit("finish_asr", socket_event.SerializeToString(), to=session_data.sid)
