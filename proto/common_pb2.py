# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: common.proto
# Protobuf Python Version: 5.27.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    1,
    '',
    'common.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x63ommon.proto\x12\x07kurumai\x1a\x1fgoogle/protobuf/timestamp.proto\"\'\n\tMicPacket\x12\x11\n\x04\x64\x61ta\x18\x01 \x01(\x0cH\x00\x88\x01\x01\x42\x07\n\x05_data\"\x86\x01\n\nTTSMessage\x12&\n\x07message\x18\x01 \x01(\x0b\x32\x10.kurumai.MessageH\x00\x88\x01\x01\x12(\n\x0b\x65xpressions\x18\x02 \x03(\x0b\x32\x13.kurumai.Expression\x12\x11\n\x04\x64\x61ta\x18\x03 \x01(\x0cH\x01\x88\x01\x01\x42\n\n\x08_messageB\x07\n\x05_data\"\xdb\x01\n\x07Message\x12\x0f\n\x02id\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x14\n\x07user_id\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x1c\n\x0f\x63onversation_id\x18\x03 \x01(\tH\x02\x88\x01\x01\x12\x14\n\x07\x63ontent\x18\x04 \x01(\tH\x03\x88\x01\x01\x12\x33\n\ncreated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampH\x04\x88\x01\x01\x42\x05\n\x03_idB\n\n\x08_user_idB\x12\n\x10_conversation_idB\n\n\x08_contentB\r\n\x0b_created_at\"V\n\nExpression\x12 \n\x07visemes\x18\x01 \x03(\x0b\x32\x0f.kurumai.Viseme\x12\x17\n\nstart_time\x18\x02 \x01(\x02H\x00\x88\x01\x01\x42\r\n\x0b_start_time\"F\n\x06Viseme\x12\x12\n\x05index\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12\x13\n\x06weight\x18\x02 \x01(\x02H\x01\x88\x01\x01\x42\x08\n\x06_indexB\t\n\x07_weight\"L\n\x0cStartMessage\x12\x11\n\x04type\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x14\n\x07\x64\x65tails\x18\x02 \x01(\tH\x01\x88\x01\x01\x42\x07\n\x05_typeB\n\n\x08_details\"\xd2\x01\n\x0c\x43onversation\x12\x0f\n\x02id\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x11\n\x04name\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x14\n\x07user_id\x18\x03 \x01(\tH\x02\x88\x01\x01\x12\x18\n\x0b\x62ot_user_id\x18\x04 \x01(\tH\x03\x88\x01\x01\x12\x33\n\ncreated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampH\x04\x88\x01\x01\x42\x05\n\x03_idB\x07\n\x05_nameB\n\n\x08_user_idB\x0e\n\x0c_bot_user_idB\r\n\x0b_created_at\"\xc7\x02\n\x0bModelPreset\x12\x0f\n\x02id\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x11\n\x04name\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x14\n\x07user_id\x18\x03 \x01(\tH\x02\x88\x01\x01\x12 \n\x13text_gen_model_name\x18\x04 \x01(\tH\x03\x88\x01\x01\x12&\n\x19text_gen_starting_context\x18\x05 \x01(\tH\x04\x88\x01\x01\x12\x1b\n\x0etts_model_name\x18\x06 \x01(\tH\x05\x88\x01\x01\x12\x1d\n\x10tts_speaker_name\x18\x07 \x01(\tH\x06\x88\x01\x01\x42\x05\n\x03_idB\x07\n\x05_nameB\n\n\x08_user_idB\x16\n\x14_text_gen_model_nameB\x1c\n\x1a_text_gen_starting_contextB\x11\n\x0f_tts_model_nameB\x13\n\x11_tts_speaker_nameb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'common_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_MICPACKET']._serialized_start=58
  _globals['_MICPACKET']._serialized_end=97
  _globals['_TTSMESSAGE']._serialized_start=100
  _globals['_TTSMESSAGE']._serialized_end=234
  _globals['_MESSAGE']._serialized_start=237
  _globals['_MESSAGE']._serialized_end=456
  _globals['_EXPRESSION']._serialized_start=458
  _globals['_EXPRESSION']._serialized_end=544
  _globals['_VISEME']._serialized_start=546
  _globals['_VISEME']._serialized_end=616
  _globals['_STARTMESSAGE']._serialized_start=618
  _globals['_STARTMESSAGE']._serialized_end=694
  _globals['_CONVERSATION']._serialized_start=697
  _globals['_CONVERSATION']._serialized_end=907
  _globals['_MODELPRESET']._serialized_start=910
  _globals['_MODELPRESET']._serialized_end=1237
# @@protoc_insertion_point(module_scope)
