# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from protos import services_pb2 as protos_dot_services__pb2


class ModelzooServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Inference = channel.unary_unary(
        '/modelzoo.ModelzooService/Inference',
        request_serializer=protos_dot_services__pb2.Payload.SerializeToString,
        response_deserializer=protos_dot_services__pb2.Payload.FromString,
        )
    self.GetImage = channel.unary_unary(
        '/modelzoo.ModelzooService/GetImage',
        request_serializer=protos_dot_services__pb2.ImageDownloadRequest.SerializeToString,
        response_deserializer=protos_dot_services__pb2.ImageDownloadResponse.FromString,
        )
    self.GetMetrics = channel.unary_unary(
        '/modelzoo.ModelzooService/GetMetrics',
        request_serializer=protos_dot_services__pb2.Empty.SerializeToString,
        response_deserializer=protos_dot_services__pb2.MetricItems.FromString,
        )
    self.GetToken = channel.unary_unary(
        '/modelzoo.ModelzooService/GetToken',
        request_serializer=protos_dot_services__pb2.Empty.SerializeToString,
        response_deserializer=protos_dot_services__pb2.RateLimitToken.FromString,
        )
    self.ListModels = channel.unary_unary(
        '/modelzoo.ModelzooService/ListModels',
        request_serializer=protos_dot_services__pb2.Empty.SerializeToString,
        response_deserializer=protos_dot_services__pb2.ListModelsResponse.FromString,
        )
    self.CreateUser = channel.unary_unary(
        '/modelzoo.ModelzooService/CreateUser',
        request_serializer=protos_dot_services__pb2.User.SerializeToString,
        response_deserializer=protos_dot_services__pb2.Empty.FromString,
        )
    self.CreateModel = channel.unary_unary(
        '/modelzoo.ModelzooService/CreateModel',
        request_serializer=protos_dot_services__pb2.Model.SerializeToString,
        response_deserializer=protos_dot_services__pb2.Empty.FromString,
        )
    self.GetUser = channel.unary_unary(
        '/modelzoo.ModelzooService/GetUser',
        request_serializer=protos_dot_services__pb2.User.SerializeToString,
        response_deserializer=protos_dot_services__pb2.Empty.FromString,
        )


class ModelzooServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Inference(self, request, context):
    """Inference
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetImage(self, request, context):
    """Website utils
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetMetrics(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetToken(self, request, context):
    """Rate limiting
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ListModels(self, request, context):
    """Database
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CreateUser(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CreateModel(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetUser(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ModelzooServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Inference': grpc.unary_unary_rpc_method_handler(
          servicer.Inference,
          request_deserializer=protos_dot_services__pb2.Payload.FromString,
          response_serializer=protos_dot_services__pb2.Payload.SerializeToString,
      ),
      'GetImage': grpc.unary_unary_rpc_method_handler(
          servicer.GetImage,
          request_deserializer=protos_dot_services__pb2.ImageDownloadRequest.FromString,
          response_serializer=protos_dot_services__pb2.ImageDownloadResponse.SerializeToString,
      ),
      'GetMetrics': grpc.unary_unary_rpc_method_handler(
          servicer.GetMetrics,
          request_deserializer=protos_dot_services__pb2.Empty.FromString,
          response_serializer=protos_dot_services__pb2.MetricItems.SerializeToString,
      ),
      'GetToken': grpc.unary_unary_rpc_method_handler(
          servicer.GetToken,
          request_deserializer=protos_dot_services__pb2.Empty.FromString,
          response_serializer=protos_dot_services__pb2.RateLimitToken.SerializeToString,
      ),
      'ListModels': grpc.unary_unary_rpc_method_handler(
          servicer.ListModels,
          request_deserializer=protos_dot_services__pb2.Empty.FromString,
          response_serializer=protos_dot_services__pb2.ListModelsResponse.SerializeToString,
      ),
      'CreateUser': grpc.unary_unary_rpc_method_handler(
          servicer.CreateUser,
          request_deserializer=protos_dot_services__pb2.User.FromString,
          response_serializer=protos_dot_services__pb2.Empty.SerializeToString,
      ),
      'CreateModel': grpc.unary_unary_rpc_method_handler(
          servicer.CreateModel,
          request_deserializer=protos_dot_services__pb2.Model.FromString,
          response_serializer=protos_dot_services__pb2.Empty.SerializeToString,
      ),
      'GetUser': grpc.unary_unary_rpc_method_handler(
          servicer.GetUser,
          request_deserializer=protos_dot_services__pb2.User.FromString,
          response_serializer=protos_dot_services__pb2.Empty.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'modelzoo.ModelzooService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
