# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: messages.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0emessages.proto\x12\x07kurumai\x1a\x1fgoogle/protobuf/timestamp.proto\"\'\n\tMicPacket\x12\x11\n\x04\x64\x61ta\x18\x01 \x01(\x0cH\x00\x88\x01\x01\x42\x07\n\x05_data\"\x86\x01\n\nTTSMessage\x12&\n\x07message\x18\x01 \x01(\x0b\x32\x10.kurumai.MessageH\x00\x88\x01\x01\x12(\n\x0b\x65xpressions\x18\x02 \x03(\x0b\x32\x13.kurumai.Expression\x12\x11\n\x04\x64\x61ta\x18\x03 \x01(\x0cH\x01\x88\x01\x01\x42\n\n\x08_messageB\x07\n\x05_data\"\xdb\x01\n\x07Message\x12\x0f\n\x02id\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x14\n\x07user_id\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x1c\n\x0f\x63onversation_id\x18\x03 \x01(\tH\x02\x88\x01\x01\x12\x14\n\x07\x63ontent\x18\x04 \x01(\tH\x03\x88\x01\x01\x12\x33\n\ncreated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampH\x04\x88\x01\x01\x42\x05\n\x03_idB\n\n\x08_user_idB\x12\n\x10_conversation_idB\n\n\x08_contentB\r\n\x0b_created_at\"V\n\nExpression\x12 \n\x07visemes\x18\x01 \x03(\x0b\x32\x0f.kurumai.Viseme\x12\x17\n\nstart_time\x18\x02 \x01(\x02H\x00\x88\x01\x01\x42\r\n\x0b_start_time\"F\n\x06Viseme\x12\x12\n\x05index\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12\x13\n\x06weight\x18\x02 \x01(\x02H\x01\x88\x01\x01\x42\x08\n\x06_indexB\t\n\x07_weight\"L\n\x0cStartMessage\x12\x11\n\x04type\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x14\n\x07\x64\x65tails\x18\x02 \x01(\tH\x01\x88\x01\x01\x42\x07\n\x05_typeB\n\n\x08_details\"\xcf\x01\n\x0bSocketEvent\x12\x12\n\x05\x65vent\x18\x01 \x01(\tH\x01\x88\x01\x01\x12\x0f\n\x02id\x18\x02 \x01(\tH\x02\x88\x01\x01\x12*\n\x0btts_message\x18\x03 \x01(\x0b\x32\x13.kurumai.TTSMessageH\x00\x12#\n\x07message\x18\x04 \x01(\x0b\x32\x10.kurumai.MessageH\x00\x12.\n\rstart_message\x18\x05 \x01(\x0b\x32\x15.kurumai.StartMessageH\x00\x42\t\n\x07payloadB\x08\n\x06_eventB\x05\n\x03_idb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'messages_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_MICPACKET']._serialized_start=60
  _globals['_MICPACKET']._serialized_end=99
  _globals['_TTSMESSAGE']._serialized_start=102
  _globals['_TTSMESSAGE']._serialized_end=236
  _globals['_MESSAGE']._serialized_start=239
  _globals['_MESSAGE']._serialized_end=458
  _globals['_EXPRESSION']._serialized_start=460
  _globals['_EXPRESSION']._serialized_end=546
  _globals['_VISEME']._serialized_start=548
  _globals['_VISEME']._serialized_end=618
  _globals['_STARTMESSAGE']._serialized_start=620
  _globals['_STARTMESSAGE']._serialized_end=696
  _globals['_SOCKETEVENT']._serialized_start=699
  _globals['_SOCKETEVENT']._serialized_end=906
# @@protoc_insertion_point(module_scope)
