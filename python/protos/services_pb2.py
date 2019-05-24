# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/services.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/services.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x15protos/services.proto\"\x0e\n\x0cGetModelsReq\"z\n\rGetModelsResp\x12$\n\x06models\x18\x01 \x03(\x0b\x32\x14.GetModelsResp.Model\x1a\x43\n\x05Model\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12&\n\x0emodel_category\x18\x02 \x01(\x0e\x32\x0e.ModelCategory\"V\n\x15TextGenerationRequest\x12\x14\n\x0cinput_phrase\x18\x01 \x01(\t\x12\x13\n\x0btemperature\x18\x02 \x01(\x02\x12\x12\n\nmodel_name\x18\x03 \x01(\t\"1\n\x16TextGenerationResponse\x12\x17\n\x0fgenerated_texts\x18\x01 \x03(\t\"[\n\x1bVisionClassificationRequest\x12\x13\n\x0binput_image\x18\x01 \x01(\t\x12\x13\n\x0bnum_returns\x18\x02 \x01(\r\x12\x12\n\nmodel_name\x18\x03 \x01(\t\"\x8e\x01\n\x1cVisionClassificationResponse\x12\x35\n\x07results\x18\x01 \x03(\x0b\x32$.VisionClassificationResponse.Result\x1a\x37\n\x06Result\x12\x0c\n\x04rank\x18\x01 \x01(\r\x12\x10\n\x08\x63\x61tegory\x18\x02 \x01(\t\x12\r\n\x05proba\x18\x03 \x01(\x02\"#\n\x14ImageDownloadRequest\x12\x0b\n\x03url\x18\x01 \x01(\t\"&\n\x15ImageDownloadResponse\x12\r\n\x05image\x18\x01 \x01(\t*=\n\rModelCategory\x12\x18\n\x14VisionClassification\x10\x00\x12\x12\n\x0eTextGeneration\x10\x01\x32\x8f\x02\n\x05Model\x12U\n\x14VisionClassification\x12\x1c.VisionClassificationRequest\x1a\x1d.VisionClassificationResponse\"\x00\x12\x43\n\x0eTextGeneration\x12\x16.TextGenerationRequest\x1a\x17.TextGenerationResponse\"\x00\x12;\n\x08GetImage\x12\x15.ImageDownloadRequest\x1a\x16.ImageDownloadResponse\"\x00\x12-\n\nListModels\x12\r.GetModelsReq\x1a\x0e.GetModelsResp\"\x00\x62\x06proto3')
)

_MODELCATEGORY = _descriptor.EnumDescriptor(
  name='ModelCategory',
  full_name='ModelCategory',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VisionClassification', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TextGeneration', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=619,
  serialized_end=680,
)
_sym_db.RegisterEnumDescriptor(_MODELCATEGORY)

ModelCategory = enum_type_wrapper.EnumTypeWrapper(_MODELCATEGORY)
VisionClassification = 0
TextGeneration = 1



_GETMODELSREQ = _descriptor.Descriptor(
  name='GetModelsReq',
  full_name='GetModelsReq',
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
  serialized_start=25,
  serialized_end=39,
)


_GETMODELSRESP_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='GetModelsResp.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_name', full_name='GetModelsResp.Model.model_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_category', full_name='GetModelsResp.Model.model_category', index=1,
      number=2, type=14, cpp_type=8, label=1,
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
  ],
  serialized_start=96,
  serialized_end=163,
)

_GETMODELSRESP = _descriptor.Descriptor(
  name='GetModelsResp',
  full_name='GetModelsResp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='models', full_name='GetModelsResp.models', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GETMODELSRESP_MODEL, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=41,
  serialized_end=163,
)


_TEXTGENERATIONREQUEST = _descriptor.Descriptor(
  name='TextGenerationRequest',
  full_name='TextGenerationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_phrase', full_name='TextGenerationRequest.input_phrase', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='temperature', full_name='TextGenerationRequest.temperature', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_name', full_name='TextGenerationRequest.model_name', index=2,
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
  serialized_start=165,
  serialized_end=251,
)


_TEXTGENERATIONRESPONSE = _descriptor.Descriptor(
  name='TextGenerationResponse',
  full_name='TextGenerationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='generated_texts', full_name='TextGenerationResponse.generated_texts', index=0,
      number=1, type=9, cpp_type=9, label=3,
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
  serialized_start=253,
  serialized_end=302,
)


_VISIONCLASSIFICATIONREQUEST = _descriptor.Descriptor(
  name='VisionClassificationRequest',
  full_name='VisionClassificationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_image', full_name='VisionClassificationRequest.input_image', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_returns', full_name='VisionClassificationRequest.num_returns', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_name', full_name='VisionClassificationRequest.model_name', index=2,
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
  serialized_start=304,
  serialized_end=395,
)


_VISIONCLASSIFICATIONRESPONSE_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='VisionClassificationResponse.Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rank', full_name='VisionClassificationResponse.Result.rank', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='category', full_name='VisionClassificationResponse.Result.category', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='proba', full_name='VisionClassificationResponse.Result.proba', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=485,
  serialized_end=540,
)

_VISIONCLASSIFICATIONRESPONSE = _descriptor.Descriptor(
  name='VisionClassificationResponse',
  full_name='VisionClassificationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='results', full_name='VisionClassificationResponse.results', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_VISIONCLASSIFICATIONRESPONSE_RESULT, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=398,
  serialized_end=540,
)


_IMAGEDOWNLOADREQUEST = _descriptor.Descriptor(
  name='ImageDownloadRequest',
  full_name='ImageDownloadRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='url', full_name='ImageDownloadRequest.url', index=0,
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
  serialized_start=542,
  serialized_end=577,
)


_IMAGEDOWNLOADRESPONSE = _descriptor.Descriptor(
  name='ImageDownloadResponse',
  full_name='ImageDownloadResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='ImageDownloadResponse.image', index=0,
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
  serialized_start=579,
  serialized_end=617,
)

_GETMODELSRESP_MODEL.fields_by_name['model_category'].enum_type = _MODELCATEGORY
_GETMODELSRESP_MODEL.containing_type = _GETMODELSRESP
_GETMODELSRESP.fields_by_name['models'].message_type = _GETMODELSRESP_MODEL
_VISIONCLASSIFICATIONRESPONSE_RESULT.containing_type = _VISIONCLASSIFICATIONRESPONSE
_VISIONCLASSIFICATIONRESPONSE.fields_by_name['results'].message_type = _VISIONCLASSIFICATIONRESPONSE_RESULT
DESCRIPTOR.message_types_by_name['GetModelsReq'] = _GETMODELSREQ
DESCRIPTOR.message_types_by_name['GetModelsResp'] = _GETMODELSRESP
DESCRIPTOR.message_types_by_name['TextGenerationRequest'] = _TEXTGENERATIONREQUEST
DESCRIPTOR.message_types_by_name['TextGenerationResponse'] = _TEXTGENERATIONRESPONSE
DESCRIPTOR.message_types_by_name['VisionClassificationRequest'] = _VISIONCLASSIFICATIONREQUEST
DESCRIPTOR.message_types_by_name['VisionClassificationResponse'] = _VISIONCLASSIFICATIONRESPONSE
DESCRIPTOR.message_types_by_name['ImageDownloadRequest'] = _IMAGEDOWNLOADREQUEST
DESCRIPTOR.message_types_by_name['ImageDownloadResponse'] = _IMAGEDOWNLOADRESPONSE
DESCRIPTOR.enum_types_by_name['ModelCategory'] = _MODELCATEGORY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GetModelsReq = _reflection.GeneratedProtocolMessageType('GetModelsReq', (_message.Message,), dict(
  DESCRIPTOR = _GETMODELSREQ,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:GetModelsReq)
  ))
_sym_db.RegisterMessage(GetModelsReq)

GetModelsResp = _reflection.GeneratedProtocolMessageType('GetModelsResp', (_message.Message,), dict(

  Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), dict(
    DESCRIPTOR = _GETMODELSRESP_MODEL,
    __module__ = 'protos.services_pb2'
    # @@protoc_insertion_point(class_scope:GetModelsResp.Model)
    ))
  ,
  DESCRIPTOR = _GETMODELSRESP,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:GetModelsResp)
  ))
_sym_db.RegisterMessage(GetModelsResp)
_sym_db.RegisterMessage(GetModelsResp.Model)

TextGenerationRequest = _reflection.GeneratedProtocolMessageType('TextGenerationRequest', (_message.Message,), dict(
  DESCRIPTOR = _TEXTGENERATIONREQUEST,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:TextGenerationRequest)
  ))
_sym_db.RegisterMessage(TextGenerationRequest)

TextGenerationResponse = _reflection.GeneratedProtocolMessageType('TextGenerationResponse', (_message.Message,), dict(
  DESCRIPTOR = _TEXTGENERATIONRESPONSE,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:TextGenerationResponse)
  ))
_sym_db.RegisterMessage(TextGenerationResponse)

VisionClassificationRequest = _reflection.GeneratedProtocolMessageType('VisionClassificationRequest', (_message.Message,), dict(
  DESCRIPTOR = _VISIONCLASSIFICATIONREQUEST,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:VisionClassificationRequest)
  ))
_sym_db.RegisterMessage(VisionClassificationRequest)

VisionClassificationResponse = _reflection.GeneratedProtocolMessageType('VisionClassificationResponse', (_message.Message,), dict(

  Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), dict(
    DESCRIPTOR = _VISIONCLASSIFICATIONRESPONSE_RESULT,
    __module__ = 'protos.services_pb2'
    # @@protoc_insertion_point(class_scope:VisionClassificationResponse.Result)
    ))
  ,
  DESCRIPTOR = _VISIONCLASSIFICATIONRESPONSE,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:VisionClassificationResponse)
  ))
_sym_db.RegisterMessage(VisionClassificationResponse)
_sym_db.RegisterMessage(VisionClassificationResponse.Result)

ImageDownloadRequest = _reflection.GeneratedProtocolMessageType('ImageDownloadRequest', (_message.Message,), dict(
  DESCRIPTOR = _IMAGEDOWNLOADREQUEST,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:ImageDownloadRequest)
  ))
_sym_db.RegisterMessage(ImageDownloadRequest)

ImageDownloadResponse = _reflection.GeneratedProtocolMessageType('ImageDownloadResponse', (_message.Message,), dict(
  DESCRIPTOR = _IMAGEDOWNLOADRESPONSE,
  __module__ = 'protos.services_pb2'
  # @@protoc_insertion_point(class_scope:ImageDownloadResponse)
  ))
_sym_db.RegisterMessage(ImageDownloadResponse)



_MODEL = _descriptor.ServiceDescriptor(
  name='Model',
  full_name='Model',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=683,
  serialized_end=954,
  methods=[
  _descriptor.MethodDescriptor(
    name='VisionClassification',
    full_name='Model.VisionClassification',
    index=0,
    containing_service=None,
    input_type=_VISIONCLASSIFICATIONREQUEST,
    output_type=_VISIONCLASSIFICATIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='TextGeneration',
    full_name='Model.TextGeneration',
    index=1,
    containing_service=None,
    input_type=_TEXTGENERATIONREQUEST,
    output_type=_TEXTGENERATIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetImage',
    full_name='Model.GetImage',
    index=2,
    containing_service=None,
    input_type=_IMAGEDOWNLOADREQUEST,
    output_type=_IMAGEDOWNLOADRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ListModels',
    full_name='Model.ListModels',
    index=3,
    containing_service=None,
    input_type=_GETMODELSREQ,
    output_type=_GETMODELSRESP,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_MODEL)

DESCRIPTOR.services_by_name['Model'] = _MODEL

# @@protoc_insertion_point(module_scope)
