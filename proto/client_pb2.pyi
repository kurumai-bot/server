import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SocketEvent(_message.Message):
    __slots__ = ("event", "id", "tts_message", "message", "start_message", "conversation")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    TTS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    START_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_FIELD_NUMBER: _ClassVar[int]
    event: str
    id: str
    tts_message: _common_pb2.TTSMessage
    message: _common_pb2.Message
    start_message: _common_pb2.StartMessage
    conversation: _common_pb2.Conversation
    def __init__(self, event: _Optional[str] = ..., id: _Optional[str] = ..., tts_message: _Optional[_Union[_common_pb2.TTSMessage, _Mapping]] = ..., message: _Optional[_Union[_common_pb2.Message, _Mapping]] = ..., start_message: _Optional[_Union[_common_pb2.StartMessage, _Mapping]] = ..., conversation: _Optional[_Union[_common_pb2.Conversation, _Mapping]] = ...) -> None: ...
