# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modelzoo/protos/services.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2
from modelzoo.protos import model_apis_pb2 as modelzoo_dot_protos_dot_model__apis__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='modelzoo/protos/services.proto',
  package='modelzoo',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1emodelzoo/protos/services.proto\x12\x08modelzoo\x1a\x1cgoogle/api/annotations.proto\x1a modelzoo/protos/model_apis.proto\"\x07\n\x05\x45mpty\"$\n\x06KVPair\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"?\n\x05Model\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\"\n\x08metadata\x18\x03 \x03(\x0b\x32\x10.modelzoo.KVPair\"\'\n\x04User\x12\r\n\x05\x65mail\x18\x01 \x01(\t\x12\x10\n\x08password\x18\x02 \x01(\t\"\x1f\n\x0eRateLimitToken\x12\r\n\x05token\x18\x01 \x01(\t\"5\n\x12ListModelsResponse\x12\x1f\n\x06models\x18\x01 \x03(\x0b\x32\x0f.modelzoo.Model\"#\n\x14ImageDownloadRequest\x12\x0b\n\x03url\x18\x01 \x01(\t\"&\n\x15ImageDownloadResponse\x12\r\n\x05image\x18\x01 \x01(\t\"\xb2\x01\n\x07Payload\x12#\n\x04type\x18\x01 \x01(\x0e\x32\x15.modelzoo.PayloadType\x12 \n\x05image\x18\x02 \x01(\x0b\x32\x0f.modelzoo.ImageH\x00\x12\x1e\n\x04text\x18\x03 \x01(\x0b\x32\x0e.modelzoo.TextH\x00\x12 \n\x05table\x18\x04 \x01(\x0b\x32\x0f.modelzoo.TableH\x00\x12\x13\n\x0bresponse_id\x18\x05 \x01(\rB\t\n\x07payload\"p\n\x0bMetricItems\x12-\n\x07metrics\x18\x04 \x03(\x0b\x32\x1c.modelzoo.MetricItems.Metric\x1a\x32\n\x06Metric\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\x12\x0c\n\x04unit\x18\x03 \x01(\t*-\n\x0bPayloadType\x12\t\n\x05IMAGE\x10\x00\x12\x08\n\x04TEXT\x10\x01\x12\t\n\x05TABLE\x10\x02\x32\xda\x04\n\x0fModelzooService\x12H\n\tInference\x12\x11.modelzoo.Payload\x1a\x11.modelzoo.Payload\"\x15\x82\xd3\xe4\x93\x02\x0f\"\n/inference:\x01*\x12M\n\x08GetImage\x12\x1e.modelzoo.ImageDownloadRequest\x1a\x1f.modelzoo.ImageDownloadResponse\"\x00\x12\x36\n\nGetMetrics\x12\x0f.modelzoo.Empty\x1a\x15.modelzoo.MetricItems\"\x00\x12L\n\x08GetToken\x12\x0f.modelzoo.Empty\x1a\x18.modelzoo.RateLimitToken\"\x15\x82\xd3\xe4\x93\x02\x0f\"\n/get/token:\x01*\x12S\n\nListModels\x12\x0f.modelzoo.Empty\x1a\x1c.modelzoo.ListModelsResponse\"\x16\x82\xd3\xe4\x93\x02\x10\"\x0b/get/models:\x01*\x12\x46\n\nCreateUser\x12\x0e.modelzoo.User\x1a\x0f.modelzoo.Empty\"\x17\x82\xd3\xe4\x93\x02\x11\"\x0c/create/user:\x01*\x12I\n\x0b\x43reateModel\x12\x0f.modelzoo.Model\x1a\x0f.modelzoo.Empty\"\x18\x82\xd3\xe4\x93\x02\x12\"\r/create/model:\x01*\x12@\n\x07GetUser\x12\x0e.modelzoo.User\x1a\x0f.modelzoo.Empty\"\x14\x82\xd3\xe4\x93\x02\x0e\"\t/get/user:\x01*b\x06proto3')
  ,
  dependencies=[google_dot_api_dot_annotations__pb2.DESCRIPTOR,modelzoo_dot_protos_dot_model__apis__pb2.DESCRIPTOR,])

_PAYLOADTYPE = _descriptor.EnumDescriptor(
  name='PayloadType',
  full_name='modelzoo.PayloadType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='IMAGE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEXT', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TABLE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=721,
  serialized_end=766,
)
_sym_db.RegisterEnumDescriptor(_PAYLOADTYPE)

PayloadType = enum_type_wrapper.EnumTypeWrapper(_PAYLOADTYPE)
IMAGE = 0
TEXT = 1
TABLE = 2



_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='modelzoo.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=108,
  serialized_end=115,
)


_KVPAIR = _descriptor.Descriptor(
  name='KVPair',
  full_name='modelzoo.KVPair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='modelzoo.KVPair.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='modelzoo.KVPair.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=153,
)


_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='modelzoo.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_name', full_name='modelzoo.Model.model_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='modelzoo.Model.metadata', index=1,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=155,
  serialized_end=218,
)


_USER = _descriptor.Descriptor(
  name='User',
  full_name='modelzoo.User',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='email', full_name='modelzoo.User.email', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='password', full_name='modelzoo.User.password', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=220,
  serialized_end=259,
)


_RATELIMITTOKEN = _descriptor.Descriptor(
  name='RateLimitToken',
  full_name='modelzoo.RateLimitToken',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='token', full_name='modelzoo.RateLimitToken.token', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=261,
  serialized_end=292,
)


_LISTMODELSRESPONSE = _descriptor.Descriptor(
  name='ListModelsResponse',
  full_name='modelzoo.ListModelsResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='models', full_name='modelzoo.ListModelsResponse.models', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=294,
  serialized_end=347,
)


_IMAGEDOWNLOADREQUEST = _descriptor.Descriptor(
  name='ImageDownloadRequest',
  full_name='modelzoo.ImageDownloadRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='url', full_name='modelzoo.ImageDownloadRequest.url', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=349,
  serialized_end=384,
)


_IMAGEDOWNLOADRESPONSE = _descriptor.Descriptor(
  name='ImageDownloadResponse',
  full_name='modelzoo.ImageDownloadResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='modelzoo.ImageDownloadResponse.image', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=386,
  serialized_end=424,
)


_PAYLOAD = _descriptor.Descriptor(
  name='Payload',
  full_name='modelzoo.Payload',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='modelzoo.Payload.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image', full_name='modelzoo.Payload.image', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='text', full_name='modelzoo.Payload.text', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='table', full_name='modelzoo.Payload.table', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='response_id', full_name='modelzoo.Payload.response_id', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='payload', full_name='modelzoo.Payload.payload',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=427,
  serialized_end=605,
)


_METRICITEMS_METRIC = _descriptor.Descriptor(
  name='Metric',
  full_name='modelzoo.MetricItems.Metric',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='modelzoo.MetricItems.Metric.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='modelzoo.MetricItems.Metric.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unit', full_name='modelzoo.MetricItems.Metric.unit', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=669,
  serialized_end=719,
)

_METRICITEMS = _descriptor.Descriptor(
  name='MetricItems',
  full_name='modelzoo.MetricItems',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='metrics', full_name='modelzoo.MetricItems.metrics', index=0,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_METRICITEMS_METRIC, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=607,
  serialized_end=719,
)

_MODEL.fields_by_name['metadata'].message_type = _KVPAIR
_LISTMODELSRESPONSE.fields_by_name['models'].message_type = _MODEL
_PAYLOAD.fields_by_name['type'].enum_type = _PAYLOADTYPE
_PAYLOAD.fields_by_name['image'].message_type = modelzoo_dot_protos_dot_model__apis__pb2._IMAGE
_PAYLOAD.fields_by_name['text'].message_type = modelzoo_dot_protos_dot_model__apis__pb2._TEXT
_PAYLOAD.fields_by_name['table'].message_type = modelzoo_dot_protos_dot_model__apis__pb2._TABLE
_PAYLOAD.oneofs_by_name['payload'].fields.append(
  _PAYLOAD.fields_by_name['image'])
_PAYLOAD.fields_by_name['image'].containing_oneof = _PAYLOAD.oneofs_by_name['payload']
_PAYLOAD.oneofs_by_name['payload'].fields.append(
  _PAYLOAD.fields_by_name['text'])
_PAYLOAD.fields_by_name['text'].containing_oneof = _PAYLOAD.oneofs_by_name['payload']
_PAYLOAD.oneofs_by_name['payload'].fields.append(
  _PAYLOAD.fields_by_name['table'])
_PAYLOAD.fields_by_name['table'].containing_oneof = _PAYLOAD.oneofs_by_name['payload']
_METRICITEMS_METRIC.containing_type = _METRICITEMS
_METRICITEMS.fields_by_name['metrics'].message_type = _METRICITEMS_METRIC
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['KVPair'] = _KVPAIR
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['User'] = _USER
DESCRIPTOR.message_types_by_name['RateLimitToken'] = _RATELIMITTOKEN
DESCRIPTOR.message_types_by_name['ListModelsResponse'] = _LISTMODELSRESPONSE
DESCRIPTOR.message_types_by_name['ImageDownloadRequest'] = _IMAGEDOWNLOADREQUEST
DESCRIPTOR.message_types_by_name['ImageDownloadResponse'] = _IMAGEDOWNLOADRESPONSE
DESCRIPTOR.message_types_by_name['Payload'] = _PAYLOAD
DESCRIPTOR.message_types_by_name['MetricItems'] = _METRICITEMS
DESCRIPTOR.enum_types_by_name['PayloadType'] = _PAYLOADTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.Empty)
  })
_sym_db.RegisterMessage(Empty)

KVPair = _reflection.GeneratedProtocolMessageType('KVPair', (_message.Message,), {
  'DESCRIPTOR' : _KVPAIR,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.KVPair)
  })
_sym_db.RegisterMessage(KVPair)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), {
  'DESCRIPTOR' : _MODEL,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.Model)
  })
_sym_db.RegisterMessage(Model)

User = _reflection.GeneratedProtocolMessageType('User', (_message.Message,), {
  'DESCRIPTOR' : _USER,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.User)
  })
_sym_db.RegisterMessage(User)

RateLimitToken = _reflection.GeneratedProtocolMessageType('RateLimitToken', (_message.Message,), {
  'DESCRIPTOR' : _RATELIMITTOKEN,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.RateLimitToken)
  })
_sym_db.RegisterMessage(RateLimitToken)

ListModelsResponse = _reflection.GeneratedProtocolMessageType('ListModelsResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTMODELSRESPONSE,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.ListModelsResponse)
  })
_sym_db.RegisterMessage(ListModelsResponse)

ImageDownloadRequest = _reflection.GeneratedProtocolMessageType('ImageDownloadRequest', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEDOWNLOADREQUEST,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.ImageDownloadRequest)
  })
_sym_db.RegisterMessage(ImageDownloadRequest)

ImageDownloadResponse = _reflection.GeneratedProtocolMessageType('ImageDownloadResponse', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEDOWNLOADRESPONSE,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.ImageDownloadResponse)
  })
_sym_db.RegisterMessage(ImageDownloadResponse)

Payload = _reflection.GeneratedProtocolMessageType('Payload', (_message.Message,), {
  'DESCRIPTOR' : _PAYLOAD,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.Payload)
  })
_sym_db.RegisterMessage(Payload)

MetricItems = _reflection.GeneratedProtocolMessageType('MetricItems', (_message.Message,), {

  'Metric' : _reflection.GeneratedProtocolMessageType('Metric', (_message.Message,), {
    'DESCRIPTOR' : _METRICITEMS_METRIC,
    '__module__' : 'modelzoo.protos.services_pb2'
    # @@protoc_insertion_point(class_scope:modelzoo.MetricItems.Metric)
    })
  ,
  'DESCRIPTOR' : _METRICITEMS,
  '__module__' : 'modelzoo.protos.services_pb2'
  # @@protoc_insertion_point(class_scope:modelzoo.MetricItems)
  })
_sym_db.RegisterMessage(MetricItems)
_sym_db.RegisterMessage(MetricItems.Metric)



_MODELZOOSERVICE = _descriptor.ServiceDescriptor(
  name='ModelzooService',
  full_name='modelzoo.ModelzooService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=769,
  serialized_end=1371,
  methods=[
  _descriptor.MethodDescriptor(
    name='Inference',
    full_name='modelzoo.ModelzooService.Inference',
    index=0,
    containing_service=None,
    input_type=_PAYLOAD,
    output_type=_PAYLOAD,
    serialized_options=_b('\202\323\344\223\002\017\"\n/inference:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='GetImage',
    full_name='modelzoo.ModelzooService.GetImage',
    index=1,
    containing_service=None,
    input_type=_IMAGEDOWNLOADREQUEST,
    output_type=_IMAGEDOWNLOADRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetMetrics',
    full_name='modelzoo.ModelzooService.GetMetrics',
    index=2,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_METRICITEMS,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetToken',
    full_name='modelzoo.ModelzooService.GetToken',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_RATELIMITTOKEN,
    serialized_options=_b('\202\323\344\223\002\017\"\n/get/token:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='ListModels',
    full_name='modelzoo.ModelzooService.ListModels',
    index=4,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_LISTMODELSRESPONSE,
    serialized_options=_b('\202\323\344\223\002\020\"\013/get/models:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='CreateUser',
    full_name='modelzoo.ModelzooService.CreateUser',
    index=5,
    containing_service=None,
    input_type=_USER,
    output_type=_EMPTY,
    serialized_options=_b('\202\323\344\223\002\021\"\014/create/user:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='CreateModel',
    full_name='modelzoo.ModelzooService.CreateModel',
    index=6,
    containing_service=None,
    input_type=_MODEL,
    output_type=_EMPTY,
    serialized_options=_b('\202\323\344\223\002\022\"\r/create/model:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='GetUser',
    full_name='modelzoo.ModelzooService.GetUser',
    index=7,
    containing_service=None,
    input_type=_USER,
    output_type=_EMPTY,
    serialized_options=_b('\202\323\344\223\002\016\"\t/get/user:\001*'),
  ),
])
_sym_db.RegisterServiceDescriptor(_MODELZOOSERVICE)

DESCRIPTOR.services_by_name['ModelzooService'] = _MODELZOOSERVICE

# @@protoc_insertion_point(module_scope)
