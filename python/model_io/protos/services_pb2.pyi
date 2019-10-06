# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    List as typing___List,
    Mapping as typing___Mapping,
    MutableMapping as typing___MutableMapping,
    Optional as typing___Optional,
    Text as typing___Text,
    Tuple as typing___Tuple,
    cast as typing___cast,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


class PayloadType(int):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    @classmethod
    def Name(cls, number: int) -> str: ...
    @classmethod
    def Value(cls, name: str) -> PayloadType: ...
    @classmethod
    def keys(cls) -> typing___List[str]: ...
    @classmethod
    def values(cls) -> typing___List[PayloadType]: ...
    @classmethod
    def items(cls) -> typing___List[typing___Tuple[str, PayloadType]]: ...
IMAGE = typing___cast(PayloadType, 0)
TEXT = typing___cast(PayloadType, 1)
TABLE = typing___cast(PayloadType, 2)

class Image(google___protobuf___message___Message):
    class MetadataEntry(google___protobuf___message___Message):
        key = ... # type: typing___Text
        value = ... # type: typing___Text

        def __init__(self,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> Image.MetadataEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[b"key",b"value"]) -> None: ...

    image_data_url = ... # type: typing___Text
    model_name = ... # type: typing___Text
    access_token = ... # type: typing___Text

    @property
    def metadata(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    def __init__(self,
        metadata : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        image_data_url : typing___Optional[typing___Text] = None,
        model_name : typing___Optional[typing___Text] = None,
        access_token : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Image: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",u"image_data_url",u"metadata",u"model_name"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"access_token",b"image_data_url",b"metadata",b"model_name"]) -> None: ...

class Text(google___protobuf___message___Message):
    class MetadataEntry(google___protobuf___message___Message):
        key = ... # type: typing___Text
        value = ... # type: typing___Text

        def __init__(self,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> Text.MetadataEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[b"key",b"value"]) -> None: ...

    texts = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text]
    model_name = ... # type: typing___Text
    access_token = ... # type: typing___Text

    @property
    def metadata(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    def __init__(self,
        metadata : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        texts : typing___Optional[typing___Iterable[typing___Text]] = None,
        model_name : typing___Optional[typing___Text] = None,
        access_token : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Text: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",u"metadata",u"model_name",u"texts"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"access_token",b"metadata",b"model_name",b"texts"]) -> None: ...

class Table(google___protobuf___message___Message):
    class MetadataEntry(google___protobuf___message___Message):
        key = ... # type: typing___Text
        value = ... # type: typing___Text

        def __init__(self,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> Table.MetadataEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[b"key",b"value"]) -> None: ...

    class Row(google___protobuf___message___Message):
        class ColumnToValueEntry(google___protobuf___message___Message):
            key = ... # type: typing___Text
            value = ... # type: typing___Text

            def __init__(self,
                key : typing___Optional[typing___Text] = None,
                value : typing___Optional[typing___Text] = None,
                ) -> None: ...
            @classmethod
            def FromString(cls, s: bytes) -> Table.Row.ColumnToValueEntry: ...
            def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
            def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
            if sys.version_info >= (3,):
                def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
            else:
                def ClearField(self, field_name: typing_extensions___Literal[b"key",b"value"]) -> None: ...


        @property
        def column_to_value(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

        def __init__(self,
            column_to_value : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> Table.Row: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"column_to_value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[b"column_to_value"]) -> None: ...

    class TableEntry(google___protobuf___message___Message):
        key = ... # type: typing___Text

        @property
        def value(self) -> Table.Row: ...

        def __init__(self,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[Table.Row] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> Table.TableEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def HasField(self, field_name: typing_extensions___Literal[u"value"]) -> bool: ...
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def HasField(self, field_name: typing_extensions___Literal[u"value",b"value"]) -> bool: ...
            def ClearField(self, field_name: typing_extensions___Literal[b"key",b"value"]) -> None: ...

    model_name = ... # type: typing___Text
    access_token = ... # type: typing___Text
    column_names = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text]

    @property
    def metadata(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    @property
    def table(self) -> typing___MutableMapping[typing___Text, Table.Row]: ...

    def __init__(self,
        metadata : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        model_name : typing___Optional[typing___Text] = None,
        access_token : typing___Optional[typing___Text] = None,
        table : typing___Optional[typing___Mapping[typing___Text, Table.Row]] = None,
        column_names : typing___Optional[typing___Iterable[typing___Text]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Table: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",u"column_names",u"metadata",u"model_name",u"table"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"access_token",b"column_names",b"metadata",b"model_name",b"table"]) -> None: ...

class Empty(google___protobuf___message___Message):

    def __init__(self,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Empty: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class KVPair(google___protobuf___message___Message):
    key = ... # type: typing___Text
    value = ... # type: typing___Text

    def __init__(self,
        key : typing___Optional[typing___Text] = None,
        value : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> KVPair: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"key",b"value"]) -> None: ...

class Model(google___protobuf___message___Message):
    model_name = ... # type: typing___Text

    @property
    def metadata(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[KVPair]: ...

    def __init__(self,
        model_name : typing___Optional[typing___Text] = None,
        metadata : typing___Optional[typing___Iterable[KVPair]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Model: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"metadata",u"model_name"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"metadata",b"model_name"]) -> None: ...

class User(google___protobuf___message___Message):
    email = ... # type: typing___Text
    password = ... # type: typing___Text

    def __init__(self,
        email : typing___Optional[typing___Text] = None,
        password : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> User: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"email",u"password"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"email",b"password"]) -> None: ...

class RateLimitToken(google___protobuf___message___Message):
    token = ... # type: typing___Text

    def __init__(self,
        token : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> RateLimitToken: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"token"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"token"]) -> None: ...

class ListModelsResponse(google___protobuf___message___Message):

    @property
    def models(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Model]: ...

    def __init__(self,
        models : typing___Optional[typing___Iterable[Model]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ListModelsResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"models"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"models"]) -> None: ...

class ImageDownloadRequest(google___protobuf___message___Message):
    url = ... # type: typing___Text

    def __init__(self,
        url : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ImageDownloadRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"url"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"url"]) -> None: ...

class ImageDownloadResponse(google___protobuf___message___Message):
    image = ... # type: typing___Text

    def __init__(self,
        image : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ImageDownloadResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"image"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"image"]) -> None: ...

class Payload(google___protobuf___message___Message):
    type = ... # type: PayloadType

    @property
    def image(self) -> Image: ...

    @property
    def text(self) -> Text: ...

    @property
    def table(self) -> Table: ...

    def __init__(self,
        type : typing___Optional[PayloadType] = None,
        image : typing___Optional[Image] = None,
        text : typing___Optional[Text] = None,
        table : typing___Optional[Table] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Payload: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"image",u"payload",u"table",u"text"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"image",u"payload",u"table",u"text",u"type"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"image",b"image",u"payload",b"payload",u"table",b"table",u"text",b"text"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[b"image",b"payload",b"table",b"text",b"type"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions___Literal[u"payload",b"payload"]) -> typing_extensions___Literal["image","text","table"]: ...
