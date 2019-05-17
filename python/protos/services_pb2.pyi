# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.message import Message as google___protobuf___message___Message

from typing import (
    Iterable as typing___Iterable,
    Optional as typing___Optional,
    Text as typing___Text,
)

from typing_extensions import Literal as typing_extensions___Literal

class VisionClassificationGetModelsReq(google___protobuf___message___Message):
    def __init__(self,) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> VisionClassificationGetModelsReq: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class VisionClassificationGetModelsResp(google___protobuf___message___Message):
    models = (
        ...
    )  # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text]
    def __init__(
        self, models: typing___Optional[typing___Iterable[typing___Text]] = None
    ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> VisionClassificationGetModelsResp: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(
            self, field_name: typing_extensions___Literal[u"models"]
        ) -> None: ...
    else:
        def ClearField(
            self, field_name: typing_extensions___Literal[b"models"]
        ) -> None: ...

class VisionClassificationRequest(google___protobuf___message___Message):
    input_image = ...  # type: typing___Text
    num_returns = ...  # type: int
    model_name = ...  # type: typing___Text
    def __init__(
        self,
        input_image: typing___Optional[typing___Text] = None,
        num_returns: typing___Optional[int] = None,
        model_name: typing___Optional[typing___Text] = None,
    ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> VisionClassificationRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(
            self,
            field_name: typing_extensions___Literal[
                u"input_image", u"model_name", u"num_returns"
            ],
        ) -> None: ...
    else:
        def ClearField(
            self,
            field_name: typing_extensions___Literal[
                b"input_image", b"model_name", b"num_returns"
            ],
        ) -> None: ...

class VisionClassificationResponse(google___protobuf___message___Message):
    class Result(google___protobuf___message___Message):
        rank = ...  # type: int
        category = ...  # type: typing___Text
        proba = ...  # type: float
        def __init__(
            self,
            rank: typing___Optional[int] = None,
            category: typing___Optional[typing___Text] = None,
            proba: typing___Optional[float] = None,
        ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> VisionClassificationResponse.Result: ...
        def MergeFrom(
            self, other_msg: google___protobuf___message___Message
        ) -> None: ...
        def CopyFrom(
            self, other_msg: google___protobuf___message___Message
        ) -> None: ...
        if sys.version_info >= (3,):
            def ClearField(
                self,
                field_name: typing_extensions___Literal[u"category", u"proba", u"rank"],
            ) -> None: ...
        else:
            def ClearField(
                self,
                field_name: typing_extensions___Literal[b"category", b"proba", b"rank"],
            ) -> None: ...
    @property
    def results(
        self
    ) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[
        VisionClassificationResponse.Result
    ]: ...
    def __init__(
        self,
        results: typing___Optional[
            typing___Iterable[VisionClassificationResponse.Result]
        ] = None,
    ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> VisionClassificationResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(
            self, field_name: typing_extensions___Literal[u"results"]
        ) -> None: ...
    else:
        def ClearField(
            self, field_name: typing_extensions___Literal[b"results"]
        ) -> None: ...

class ImageDownloadRequest(google___protobuf___message___Message):
    url = ...  # type: typing___Text
    def __init__(self, url: typing___Optional[typing___Text] = None) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ImageDownloadRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(
            self, field_name: typing_extensions___Literal[u"url"]
        ) -> None: ...
    else:
        def ClearField(
            self, field_name: typing_extensions___Literal[b"url"]
        ) -> None: ...

class ImageDownloadResponse(google___protobuf___message___Message):
    image = ...  # type: typing___Text
    def __init__(self, image: typing___Optional[typing___Text] = None) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ImageDownloadResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(
            self, field_name: typing_extensions___Literal[u"image"]
        ) -> None: ...
    else:
        def ClearField(
            self, field_name: typing_extensions___Literal[b"image"]
        ) -> None: ...
