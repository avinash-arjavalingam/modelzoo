package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/hex"
	"fmt"
	"math/rand"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/harbor-ml/modelzoo/go/schema"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	modelzoo "github.com/harbor-ml/modelzoo/go/modelzoo/protos"
	log "github.com/sirupsen/logrus"
	cb_client "github.com/cloudburstclient"
	cb_proto "github.com/proto/common"
)

func (s *ProxyServer) getModelVersion(modelName string) schema.ModelVersion {
	modelRecord := schema.ModelVersion{}
	s.db.Where("name = ?", modelName).Find(&modelRecord)
	return modelRecord
}

func (s *ProxyServer) getAccesToken(token string) schema.Token {
	accessToken := schema.Token{}
	s.db.Where("secret = ?", token).Find(&accessToken)
	return accessToken
}

func (s *ProxyServer) validateModel(msg proto.Message, modelName string) (map[string]string, []byte, error) {
	modelRecord := schema.ModelVersion{}
	if err := s.db.Where("name = ?", modelName).Find(&modelRecord).Error; err != nil {
		return nil, nil, status.Error(codes.Internal, fmt.Sprint(err))
	}

	metadata, err := schema.GetMetadataMap(s.db, &modelRecord)
	if err != nil {
		return nil, nil, err
	}

	if metadata["service_type"] != "clipper" {
		return nil, nil, status.Error(codes.Internal, "modelzoo only support clipper for now")
	}

	marshaled, err := proto.Marshal(msg)
	if err != nil {
		return nil, nil, err
	}
	return metadata, marshaled, nil
}

func (s *ProxyServer) saveQueryResult(modelName string, token string,
	startTime time.Time, endTime time.Time, exitStatus int) {
	queryEntry := schema.Query{
		ModelVersion: s.getModelVersion(modelName),
		Token:        s.getAccesToken(token),
		StartedAt:    startTime,
		EndedAt:      endTime,
		Status:       exitStatus,
	}
	s.db.Save(&queryEntry)
}

func (s *ProxyServer) Inference(ctx context.Context, payload *modelzoo.Payload) (*modelzoo.Payload, error) {
	var metadata map[string]string
	var serializedPayload []byte
	var err error
	var token string
	var modelName string

	switch payload.Type {
	case modelzoo.PayloadType_IMAGE:
		item := payload.GetImage()
		modelName = item.ModelName
		token = item.AccessToken
		metadata, serializedPayload, err = s.validateModel(item, item.ModelName)
	case modelzoo.PayloadType_TEXT:
		item := payload.GetText()
		modelName = item.ModelName
		token = item.AccessToken
		metadata, serializedPayload, err = s.validateModel(item, item.ModelName)
	case modelzoo.PayloadType_TABLE:
		item := payload.GetTable()
		modelName = item.ModelName
		token = item.AccessToken
		metadata, serializedPayload, err = s.validateModel(item, item.ModelName)
	default:
		return nil, status.Error(codes.Internal, "wrong payload type")
	}
	if err != nil {
		return nil, err
	}

	_, err = schema.PerformRateLimit(s.db, token)

	rateLimitSuccess := true
	if err != nil {
		rateLimitSuccess = false
	}

	defer s.logger.WithFields(log.Fields{
		"input_type":           payload.Type,
		"model_name":           modelName,
		"rate_limit_success":   rateLimitSuccess,
		"expected_output_type": metadata["output_type"],
	}).Info("New inference request")

	if err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprint(err))
	}

	// url := metadata["clipper_url"]
	encodedReq := base64.StdEncoding.EncodeToString(serializedPayload)
	httpPayload := map[string]string{"input": encodedReq}
	// httpPayload := map[string]string{"hello": "world"}

	testing_client := cb_client.NewCloudburstClient("127.0.0.1", "127.0.0.1", true)
	argsMap := map[string]*cb_proto.Arguments{}
    args := &cb_proto.Arguments{}
    // httpPayloadBytes := []byte(encodedReq)
    httpPayloadBytes, _ := json.Marshal(httpPayload)
    args.Values = append(args.Values, &cb_proto.Value{Body: httpPayloadBytes, Type: cb_proto.SerializerType_STRING})
    argsMap["torch_class"] = args

	startTime := time.Now()


	respFuture := testing_client.CallDag("torch_dag", argsMap, true)
	respRaw :=  (*[]byte)(respFuture.Get())
	var dag_resp map[string]interface{}
	json.Unmarshal(*respRaw, &dag_resp)
	
	// resp := postJSON(url, httpPayload)
	defer s.logger.WithFields(log.Fields{
		"client":           testing_client,
		// "argsMap":          argsMap,
		"respType":         fmt.Sprintf("%T", encodedReq),
		"respRawLen":       len(*respRaw),
		"respRaw":          hex.EncodeToString(*respRaw),
		"resp":             dag_resp,
	}).Info("New inference request")
	endTime := time.Now()

	resp := dag_resp
	
	if resp["default"].(bool) == true {
		s.saveQueryResult(modelName, token, startTime, endTime, 1)
		return nil, status.Error(
			codes.Internal, fmt.Sprintf("Query failed: %s", resp["default_explanation"].(string)))
	}
	

	decoded, err := base64.StdEncoding.DecodeString(resp["output"].(string))
	// decoded, err := base64.StdEncoding.DecodeString(resp)
	if err != nil {
		s.saveQueryResult(modelName, token, startTime, endTime, 1)
		return nil, err
	}

	var val modelzoo.Payload
	resp_id := rand.Uint32()
	switch metadata["output_type"] {
	case "image":
		data := &modelzoo.Image{}
		if err := proto.Unmarshal(decoded, data); err != nil {
			s.saveQueryResult(modelName, token, startTime, endTime, 1)
			return nil, status.Error(codes.Internal, "failed to unmarshal proto from clipper")
		}
		val = modelzoo.Payload{
			Payload:    &modelzoo.Payload_Image{Image: data},
			Type:       modelzoo.PayloadType_IMAGE,
			ResponseId: resp_id,
		}
	case "table":
		data := &modelzoo.Table{}
		if err := proto.Unmarshal(decoded, data); err != nil {
			s.saveQueryResult(modelName, token, startTime, endTime, 1)
			return nil, status.Error(codes.Internal, "failed to unmarshal proto from clipper")
		}
		val = modelzoo.Payload{
			Payload:    &modelzoo.Payload_Table{Table: data},
			Type:       modelzoo.PayloadType_TABLE,
			ResponseId: resp_id,
		}
	case "text":
		data := &modelzoo.Text{}
		if err := proto.Unmarshal(decoded, data); err != nil {
			s.saveQueryResult(modelName, token, startTime, endTime, 1)
			return nil, status.Error(codes.Internal, "failed to unmarshal proto from clipper")
		}
		val = modelzoo.Payload{
			Payload:    &modelzoo.Payload_Text{Text: data},
			Type:       modelzoo.PayloadType_TEXT,
			ResponseId: resp_id,
		}
	default:
		return nil, status.Error(codes.Internal, "model doesn't have output_type field")
	}

	s.saveQueryResult(modelName, token, startTime, endTime, 0)

	return &val, nil
}
