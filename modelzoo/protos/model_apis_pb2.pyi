# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.internal.containers import (
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    Mapping as typing___Mapping,
    MutableMapping as typing___MutableMapping,
    Optional as typing___Optional,
    Text as typing___Text,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


class Image(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class MetadataEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key = ... # type: typing___Text
        value = ... # type: typing___Text

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: builtin___bytes) -> Image.MetadataEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...

    image_data_url = ... # type: typing___Text
    model_name = ... # type: typing___Text
    access_token = ... # type: typing___Text

    @property
    def metadata(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    def __init__(self,
        *,
        metadata : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        image_data_url : typing___Optional[typing___Text] = None,
        model_name : typing___Optional[typing___Text] = None,
        access_token : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> Image: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",u"image_data_url",u"metadata",u"model_name"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",b"access_token",u"image_data_url",b"image_data_url",u"metadata",b"metadata",u"model_name",b"model_name"]) -> None: ...

class Text(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class MetadataEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key = ... # type: typing___Text
        value = ... # type: typing___Text

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: builtin___bytes) -> Text.MetadataEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...

    texts = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text]
    model_name = ... # type: typing___Text
    access_token = ... # type: typing___Text

    @property
    def metadata(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    def __init__(self,
        *,
        metadata : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        texts : typing___Optional[typing___Iterable[typing___Text]] = None,
        model_name : typing___Optional[typing___Text] = None,
        access_token : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> Text: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",u"metadata",u"model_name",u"texts"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",b"access_token",u"metadata",b"metadata",u"model_name",b"model_name",u"texts",b"texts"]) -> None: ...

class Table(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class MetadataEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key = ... # type: typing___Text
        value = ... # type: typing___Text

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: builtin___bytes) -> Table.MetadataEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...

    class Row(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        class ColumnToValueEntry(google___protobuf___message___Message):
            DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
            key = ... # type: typing___Text
            value = ... # type: typing___Text

            def __init__(self,
                *,
                key : typing___Optional[typing___Text] = None,
                value : typing___Optional[typing___Text] = None,
                ) -> None: ...
            @classmethod
            def FromString(cls, s: builtin___bytes) -> Table.Row.ColumnToValueEntry: ...
            def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
            def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
            if sys.version_info >= (3,):
                def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
            else:
                def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...


        @property
        def column_to_value(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

        def __init__(self,
            *,
            column_to_value : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: builtin___bytes) -> Table.Row: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(self, field_name: typing_extensions___Literal[u"column_to_value"]) -> None: ...
        else:
            def ClearField(self, field_name: typing_extensions___Literal[u"column_to_value",b"column_to_value"]) -> None: ...

    class TableEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key = ... # type: typing___Text

        @property
        def value(self) -> Table.Row: ...

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[Table.Row] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: builtin___bytes) -> Table.TableEntry: ...
        def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
        if sys.version_info >= (3,):
            def HasField(self, field_name: typing_extensions___Literal[u"value"]) -> builtin___bool: ...
            def ClearField(self, field_name: typing_extensions___Literal[u"key",u"value"]) -> None: ...
        else:
            def HasField(self, field_name: typing_extensions___Literal[u"value",b"value"]) -> builtin___bool: ...
            def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...

    model_name = ... # type: typing___Text
    access_token = ... # type: typing___Text
    column_names = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text]

    @property
    def metadata(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    @property
    def table(self) -> typing___MutableMapping[typing___Text, Table.Row]: ...

    def __init__(self,
        *,
        metadata : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        model_name : typing___Optional[typing___Text] = None,
        access_token : typing___Optional[typing___Text] = None,
        table : typing___Optional[typing___Mapping[typing___Text, Table.Row]] = None,
        column_names : typing___Optional[typing___Iterable[typing___Text]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> Table: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",u"column_names",u"metadata",u"model_name",u"table"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"access_token",b"access_token",u"column_names",b"column_names",u"metadata",b"metadata",u"model_name",b"model_name",u"table",b"table"]) -> None: ...