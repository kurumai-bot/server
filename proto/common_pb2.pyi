from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MicPacket(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class TTSMessage(_message.Message):
    __slots__ = ("message", "expressions", "data")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    message: Message
    expressions: _containers.RepeatedCompositeFieldContainer[Expression]
    data: bytes
    def __init__(self, message: _Optional[_Union[Message, _Mapping]] = ..., expressions: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., data: _Optional[bytes] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ("id", "user_id", "conversation_id", "content", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    user_id: str
    conversation_id: str
    content: str
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., user_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., content: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Expression(_message.Message):
    __slots__ = ("visemes", "start_time")
    VISEMES_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    visemes: _containers.RepeatedCompositeFieldContainer[Viseme]
    start_time: float
    def __init__(self, visemes: _Optional[_Iterable[_Union[Viseme, _Mapping]]] = ..., start_time: _Optional[float] = ...) -> None: ...

class Viseme(_message.Message):
    __slots__ = ("index", "weight")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    index: int
    weight: float
    def __init__(self, index: _Optional[int] = ..., weight: _Optional[float] = ...) -> None: ...

class StartMessage(_message.Message):
    __slots__ = ("type", "details")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    type: str
    details: str
    def __init__(self, type: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class Conversation(_message.Message):
    __slots__ = ("id", "name", "user_id", "bot_user_id", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    BOT_USER_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    user_id: str
    bot_user_id: str
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., user_id: _Optional[str] = ..., bot_user_id: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ModelPreset(_message.Message):
    __slots__ = ("id", "name", "user_id", "text_gen_model_name", "text_gen_starting_context", "tts_model_name", "tts_speaker_name")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_GEN_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TEXT_GEN_STARTING_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TTS_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TTS_SPEAKER_NAME_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    user_id: str
    text_gen_model_name: str
    text_gen_starting_context: str
    tts_model_name: str
    tts_speaker_name: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., user_id: _Optional[str] = ..., text_gen_model_name: _Optional[str] = ..., text_gen_starting_context: _Optional[str] = ..., tts_model_name: _Optional[str] = ..., tts_speaker_name: _Optional[str] = ...) -> None: ...
