// Code generated by protoc-gen-go. DO NOT EDIT.
// source: modelzoo/protos/services.proto

package modelzoo

import (
	context "context"
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	_ "google.golang.org/genproto/googleapis/api/annotations"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type PayloadType int32

const (
	PayloadType_IMAGE PayloadType = 0
	PayloadType_TEXT  PayloadType = 1
	PayloadType_TABLE PayloadType = 2
)

var PayloadType_name = map[int32]string{
	0: "IMAGE",
	1: "TEXT",
	2: "TABLE",
}

var PayloadType_value = map[string]int32{
	"IMAGE": 0,
	"TEXT":  1,
	"TABLE": 2,
}

func (x PayloadType) String() string {
	return proto.EnumName(PayloadType_name, int32(x))
}

func (PayloadType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{0}
}

// Web
type Empty struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Empty) Reset()         { *m = Empty{} }
func (m *Empty) String() string { return proto.CompactTextString(m) }
func (*Empty) ProtoMessage()    {}
func (*Empty) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{0}
}

func (m *Empty) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Empty.Unmarshal(m, b)
}
func (m *Empty) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Empty.Marshal(b, m, deterministic)
}
func (m *Empty) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Empty.Merge(m, src)
}
func (m *Empty) XXX_Size() int {
	return xxx_messageInfo_Empty.Size(m)
}
func (m *Empty) XXX_DiscardUnknown() {
	xxx_messageInfo_Empty.DiscardUnknown(m)
}

var xxx_messageInfo_Empty proto.InternalMessageInfo

type KVPair struct {
	Key                  string   `protobuf:"bytes,1,opt,name=key,proto3" json:"key,omitempty"`
	Value                string   `protobuf:"bytes,2,opt,name=value,proto3" json:"value,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *KVPair) Reset()         { *m = KVPair{} }
func (m *KVPair) String() string { return proto.CompactTextString(m) }
func (*KVPair) ProtoMessage()    {}
func (*KVPair) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{1}
}

func (m *KVPair) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_KVPair.Unmarshal(m, b)
}
func (m *KVPair) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_KVPair.Marshal(b, m, deterministic)
}
func (m *KVPair) XXX_Merge(src proto.Message) {
	xxx_messageInfo_KVPair.Merge(m, src)
}
func (m *KVPair) XXX_Size() int {
	return xxx_messageInfo_KVPair.Size(m)
}
func (m *KVPair) XXX_DiscardUnknown() {
	xxx_messageInfo_KVPair.DiscardUnknown(m)
}

var xxx_messageInfo_KVPair proto.InternalMessageInfo

func (m *KVPair) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *KVPair) GetValue() string {
	if m != nil {
		return m.Value
	}
	return ""
}

type Model struct {
	ModelName            string    `protobuf:"bytes,1,opt,name=model_name,json=modelName,proto3" json:"model_name,omitempty"`
	Metadata             []*KVPair `protobuf:"bytes,3,rep,name=metadata,proto3" json:"metadata,omitempty"`
	XXX_NoUnkeyedLiteral struct{}  `json:"-"`
	XXX_unrecognized     []byte    `json:"-"`
	XXX_sizecache        int32     `json:"-"`
}

func (m *Model) Reset()         { *m = Model{} }
func (m *Model) String() string { return proto.CompactTextString(m) }
func (*Model) ProtoMessage()    {}
func (*Model) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{2}
}

func (m *Model) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Model.Unmarshal(m, b)
}
func (m *Model) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Model.Marshal(b, m, deterministic)
}
func (m *Model) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Model.Merge(m, src)
}
func (m *Model) XXX_Size() int {
	return xxx_messageInfo_Model.Size(m)
}
func (m *Model) XXX_DiscardUnknown() {
	xxx_messageInfo_Model.DiscardUnknown(m)
}

var xxx_messageInfo_Model proto.InternalMessageInfo

func (m *Model) GetModelName() string {
	if m != nil {
		return m.ModelName
	}
	return ""
}

func (m *Model) GetMetadata() []*KVPair {
	if m != nil {
		return m.Metadata
	}
	return nil
}

type User struct {
	Email                string   `protobuf:"bytes,1,opt,name=email,proto3" json:"email,omitempty"`
	Password             string   `protobuf:"bytes,2,opt,name=password,proto3" json:"password,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *User) Reset()         { *m = User{} }
func (m *User) String() string { return proto.CompactTextString(m) }
func (*User) ProtoMessage()    {}
func (*User) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{3}
}

func (m *User) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_User.Unmarshal(m, b)
}
func (m *User) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_User.Marshal(b, m, deterministic)
}
func (m *User) XXX_Merge(src proto.Message) {
	xxx_messageInfo_User.Merge(m, src)
}
func (m *User) XXX_Size() int {
	return xxx_messageInfo_User.Size(m)
}
func (m *User) XXX_DiscardUnknown() {
	xxx_messageInfo_User.DiscardUnknown(m)
}

var xxx_messageInfo_User proto.InternalMessageInfo

func (m *User) GetEmail() string {
	if m != nil {
		return m.Email
	}
	return ""
}

func (m *User) GetPassword() string {
	if m != nil {
		return m.Password
	}
	return ""
}

type RateLimitToken struct {
	Token                string   `protobuf:"bytes,1,opt,name=token,proto3" json:"token,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *RateLimitToken) Reset()         { *m = RateLimitToken{} }
func (m *RateLimitToken) String() string { return proto.CompactTextString(m) }
func (*RateLimitToken) ProtoMessage()    {}
func (*RateLimitToken) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{4}
}

func (m *RateLimitToken) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RateLimitToken.Unmarshal(m, b)
}
func (m *RateLimitToken) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RateLimitToken.Marshal(b, m, deterministic)
}
func (m *RateLimitToken) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RateLimitToken.Merge(m, src)
}
func (m *RateLimitToken) XXX_Size() int {
	return xxx_messageInfo_RateLimitToken.Size(m)
}
func (m *RateLimitToken) XXX_DiscardUnknown() {
	xxx_messageInfo_RateLimitToken.DiscardUnknown(m)
}

var xxx_messageInfo_RateLimitToken proto.InternalMessageInfo

func (m *RateLimitToken) GetToken() string {
	if m != nil {
		return m.Token
	}
	return ""
}

type ListModelsResponse struct {
	Models               []*Model `protobuf:"bytes,1,rep,name=models,proto3" json:"models,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ListModelsResponse) Reset()         { *m = ListModelsResponse{} }
func (m *ListModelsResponse) String() string { return proto.CompactTextString(m) }
func (*ListModelsResponse) ProtoMessage()    {}
func (*ListModelsResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{5}
}

func (m *ListModelsResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ListModelsResponse.Unmarshal(m, b)
}
func (m *ListModelsResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ListModelsResponse.Marshal(b, m, deterministic)
}
func (m *ListModelsResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ListModelsResponse.Merge(m, src)
}
func (m *ListModelsResponse) XXX_Size() int {
	return xxx_messageInfo_ListModelsResponse.Size(m)
}
func (m *ListModelsResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ListModelsResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ListModelsResponse proto.InternalMessageInfo

func (m *ListModelsResponse) GetModels() []*Model {
	if m != nil {
		return m.Models
	}
	return nil
}

// Downloader
type ImageDownloadRequest struct {
	Url                  string   `protobuf:"bytes,1,opt,name=url,proto3" json:"url,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ImageDownloadRequest) Reset()         { *m = ImageDownloadRequest{} }
func (m *ImageDownloadRequest) String() string { return proto.CompactTextString(m) }
func (*ImageDownloadRequest) ProtoMessage()    {}
func (*ImageDownloadRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{6}
}

func (m *ImageDownloadRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ImageDownloadRequest.Unmarshal(m, b)
}
func (m *ImageDownloadRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ImageDownloadRequest.Marshal(b, m, deterministic)
}
func (m *ImageDownloadRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ImageDownloadRequest.Merge(m, src)
}
func (m *ImageDownloadRequest) XXX_Size() int {
	return xxx_messageInfo_ImageDownloadRequest.Size(m)
}
func (m *ImageDownloadRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_ImageDownloadRequest.DiscardUnknown(m)
}

var xxx_messageInfo_ImageDownloadRequest proto.InternalMessageInfo

func (m *ImageDownloadRequest) GetUrl() string {
	if m != nil {
		return m.Url
	}
	return ""
}

type ImageDownloadResponse struct {
	Image                string   `protobuf:"bytes,1,opt,name=image,proto3" json:"image,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ImageDownloadResponse) Reset()         { *m = ImageDownloadResponse{} }
func (m *ImageDownloadResponse) String() string { return proto.CompactTextString(m) }
func (*ImageDownloadResponse) ProtoMessage()    {}
func (*ImageDownloadResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{7}
}

func (m *ImageDownloadResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ImageDownloadResponse.Unmarshal(m, b)
}
func (m *ImageDownloadResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ImageDownloadResponse.Marshal(b, m, deterministic)
}
func (m *ImageDownloadResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ImageDownloadResponse.Merge(m, src)
}
func (m *ImageDownloadResponse) XXX_Size() int {
	return xxx_messageInfo_ImageDownloadResponse.Size(m)
}
func (m *ImageDownloadResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ImageDownloadResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ImageDownloadResponse proto.InternalMessageInfo

func (m *ImageDownloadResponse) GetImage() string {
	if m != nil {
		return m.Image
	}
	return ""
}

type Payload struct {
	Type PayloadType `protobuf:"varint,1,opt,name=type,proto3,enum=modelzoo.PayloadType" json:"type,omitempty"`
	// Types that are valid to be assigned to Payload:
	//	*Payload_Image
	//	*Payload_Text
	//	*Payload_Table
	Payload              isPayload_Payload `protobuf_oneof:"payload"`
	ResponseId           uint32            `protobuf:"varint,5,opt,name=response_id,json=responseId,proto3" json:"response_id,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *Payload) Reset()         { *m = Payload{} }
func (m *Payload) String() string { return proto.CompactTextString(m) }
func (*Payload) ProtoMessage()    {}
func (*Payload) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{8}
}

func (m *Payload) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Payload.Unmarshal(m, b)
}
func (m *Payload) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Payload.Marshal(b, m, deterministic)
}
func (m *Payload) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Payload.Merge(m, src)
}
func (m *Payload) XXX_Size() int {
	return xxx_messageInfo_Payload.Size(m)
}
func (m *Payload) XXX_DiscardUnknown() {
	xxx_messageInfo_Payload.DiscardUnknown(m)
}

var xxx_messageInfo_Payload proto.InternalMessageInfo

func (m *Payload) GetType() PayloadType {
	if m != nil {
		return m.Type
	}
	return PayloadType_IMAGE
}

type isPayload_Payload interface {
	isPayload_Payload()
}

type Payload_Image struct {
	Image *Image `protobuf:"bytes,2,opt,name=image,proto3,oneof"`
}

type Payload_Text struct {
	Text *Text `protobuf:"bytes,3,opt,name=text,proto3,oneof"`
}

type Payload_Table struct {
	Table *Table `protobuf:"bytes,4,opt,name=table,proto3,oneof"`
}

func (*Payload_Image) isPayload_Payload() {}

func (*Payload_Text) isPayload_Payload() {}

func (*Payload_Table) isPayload_Payload() {}

func (m *Payload) GetPayload() isPayload_Payload {
	if m != nil {
		return m.Payload
	}
	return nil
}

func (m *Payload) GetImage() *Image {
	if x, ok := m.GetPayload().(*Payload_Image); ok {
		return x.Image
	}
	return nil
}

func (m *Payload) GetText() *Text {
	if x, ok := m.GetPayload().(*Payload_Text); ok {
		return x.Text
	}
	return nil
}

func (m *Payload) GetTable() *Table {
	if x, ok := m.GetPayload().(*Payload_Table); ok {
		return x.Table
	}
	return nil
}

func (m *Payload) GetResponseId() uint32 {
	if m != nil {
		return m.ResponseId
	}
	return 0
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*Payload) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*Payload_Image)(nil),
		(*Payload_Text)(nil),
		(*Payload_Table)(nil),
	}
}

type MetricItems struct {
	Metrics              []*MetricItems_Metric `protobuf:"bytes,4,rep,name=metrics,proto3" json:"metrics,omitempty"`
	XXX_NoUnkeyedLiteral struct{}              `json:"-"`
	XXX_unrecognized     []byte                `json:"-"`
	XXX_sizecache        int32                 `json:"-"`
}

func (m *MetricItems) Reset()         { *m = MetricItems{} }
func (m *MetricItems) String() string { return proto.CompactTextString(m) }
func (*MetricItems) ProtoMessage()    {}
func (*MetricItems) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{9}
}

func (m *MetricItems) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MetricItems.Unmarshal(m, b)
}
func (m *MetricItems) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MetricItems.Marshal(b, m, deterministic)
}
func (m *MetricItems) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MetricItems.Merge(m, src)
}
func (m *MetricItems) XXX_Size() int {
	return xxx_messageInfo_MetricItems.Size(m)
}
func (m *MetricItems) XXX_DiscardUnknown() {
	xxx_messageInfo_MetricItems.DiscardUnknown(m)
}

var xxx_messageInfo_MetricItems proto.InternalMessageInfo

func (m *MetricItems) GetMetrics() []*MetricItems_Metric {
	if m != nil {
		return m.Metrics
	}
	return nil
}

type MetricItems_Metric struct {
	Key                  string   `protobuf:"bytes,1,opt,name=key,proto3" json:"key,omitempty"`
	Value                string   `protobuf:"bytes,2,opt,name=value,proto3" json:"value,omitempty"`
	Unit                 string   `protobuf:"bytes,3,opt,name=unit,proto3" json:"unit,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *MetricItems_Metric) Reset()         { *m = MetricItems_Metric{} }
func (m *MetricItems_Metric) String() string { return proto.CompactTextString(m) }
func (*MetricItems_Metric) ProtoMessage()    {}
func (*MetricItems_Metric) Descriptor() ([]byte, []int) {
	return fileDescriptor_65095a69d1aa27c9, []int{9, 0}
}

func (m *MetricItems_Metric) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MetricItems_Metric.Unmarshal(m, b)
}
func (m *MetricItems_Metric) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MetricItems_Metric.Marshal(b, m, deterministic)
}
func (m *MetricItems_Metric) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MetricItems_Metric.Merge(m, src)
}
func (m *MetricItems_Metric) XXX_Size() int {
	return xxx_messageInfo_MetricItems_Metric.Size(m)
}
func (m *MetricItems_Metric) XXX_DiscardUnknown() {
	xxx_messageInfo_MetricItems_Metric.DiscardUnknown(m)
}

var xxx_messageInfo_MetricItems_Metric proto.InternalMessageInfo

func (m *MetricItems_Metric) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *MetricItems_Metric) GetValue() string {
	if m != nil {
		return m.Value
	}
	return ""
}

func (m *MetricItems_Metric) GetUnit() string {
	if m != nil {
		return m.Unit
	}
	return ""
}

func init() {
	proto.RegisterEnum("modelzoo.PayloadType", PayloadType_name, PayloadType_value)
	proto.RegisterType((*Empty)(nil), "modelzoo.Empty")
	proto.RegisterType((*KVPair)(nil), "modelzoo.KVPair")
	proto.RegisterType((*Model)(nil), "modelzoo.Model")
	proto.RegisterType((*User)(nil), "modelzoo.User")
	proto.RegisterType((*RateLimitToken)(nil), "modelzoo.RateLimitToken")
	proto.RegisterType((*ListModelsResponse)(nil), "modelzoo.ListModelsResponse")
	proto.RegisterType((*ImageDownloadRequest)(nil), "modelzoo.ImageDownloadRequest")
	proto.RegisterType((*ImageDownloadResponse)(nil), "modelzoo.ImageDownloadResponse")
	proto.RegisterType((*Payload)(nil), "modelzoo.Payload")
	proto.RegisterType((*MetricItems)(nil), "modelzoo.MetricItems")
	proto.RegisterType((*MetricItems_Metric)(nil), "modelzoo.MetricItems.Metric")
}

func init() { proto.RegisterFile("modelzoo/protos/services.proto", fileDescriptor_65095a69d1aa27c9) }

var fileDescriptor_65095a69d1aa27c9 = []byte{
	// 724 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x94, 0xcf, 0x6f, 0xeb, 0x44,
	0x10, 0xc7, 0xe3, 0xc6, 0xf9, 0x35, 0xa1, 0x69, 0xba, 0x4a, 0xc0, 0xb2, 0x4a, 0x1b, 0xad, 0x10,
	0x0d, 0x15, 0xad, 0x51, 0x91, 0x2a, 0x84, 0x84, 0x44, 0x4b, 0x43, 0x1a, 0x91, 0xa0, 0xca, 0x35,
	0x88, 0x5b, 0xb5, 0x4d, 0x86, 0xc8, 0x6a, 0xfc, 0x03, 0xef, 0xa6, 0x6d, 0x38, 0x72, 0xe5, 0xc8,
	0x9f, 0xc6, 0xe5, 0xe9, 0x9d, 0xdf, 0x1f, 0xf2, 0xb4, 0x63, 0xbb, 0x49, 0x93, 0xf7, 0xa4, 0xde,
	0x76, 0x66, 0xbe, 0xfb, 0xd9, 0xef, 0xce, 0x7a, 0x0c, 0xfb, 0x41, 0x34, 0xc1, 0xd9, 0xdf, 0x51,
	0xe4, 0xc4, 0x49, 0xa4, 0x22, 0xe9, 0x48, 0x4c, 0x1e, 0xfc, 0x31, 0xca, 0x13, 0x8a, 0x59, 0x35,
	0xaf, 0xdb, 0x7b, 0xd3, 0x28, 0x9a, 0xce, 0xd0, 0x11, 0xb1, 0xef, 0x88, 0x30, 0x8c, 0x94, 0x50,
	0x7e, 0x14, 0x66, 0x3a, 0xbb, 0xb3, 0xce, 0xa1, 0xf8, 0x56, 0xc4, 0x7e, 0xa6, 0xe0, 0x15, 0x28,
	0xf5, 0x82, 0x58, 0x2d, 0xf8, 0x37, 0x50, 0xfe, 0xe5, 0xf7, 0x6b, 0xe1, 0x27, 0xac, 0x09, 0xc5,
	0x7b, 0x5c, 0x58, 0x46, 0xc7, 0xe8, 0xd6, 0x5c, 0xbd, 0x64, 0x2d, 0x28, 0x3d, 0x88, 0xd9, 0x1c,
	0xad, 0x2d, 0xca, 0xa5, 0x01, 0xf7, 0xa0, 0x34, 0xd2, 0x38, 0xf6, 0x39, 0x40, 0xca, 0x0d, 0x45,
	0x80, 0xd9, 0xbe, 0x1a, 0x65, 0x7e, 0x15, 0x01, 0xb2, 0xaf, 0xa1, 0x1a, 0xa0, 0x12, 0x13, 0xa1,
	0x84, 0x55, 0xec, 0x14, 0xbb, 0xf5, 0xd3, 0xe6, 0x49, 0xee, 0xeb, 0x24, 0x3d, 0xd3, 0x7d, 0x56,
	0xf0, 0xef, 0xc0, 0xfc, 0x4d, 0x62, 0xa2, 0xcf, 0xc4, 0x40, 0xf8, 0xb3, 0x8c, 0x97, 0x06, 0xcc,
	0x86, 0x6a, 0x2c, 0xa4, 0x7c, 0x8c, 0x92, 0x49, 0x66, 0xe6, 0x39, 0xe6, 0x5f, 0x42, 0xc3, 0x15,
	0x0a, 0x87, 0x7e, 0xe0, 0x2b, 0x2f, 0xba, 0xc7, 0x50, 0x33, 0x94, 0x5e, 0xe4, 0x0c, 0x0a, 0xf8,
	0x0f, 0xc0, 0x86, 0xbe, 0x54, 0xe4, 0x5d, 0xba, 0x28, 0xe3, 0x28, 0x94, 0xc8, 0x0e, 0xa1, 0x4c,
	0xa6, 0xa4, 0x65, 0x90, 0xc7, 0x9d, 0xa5, 0x47, 0x52, 0xba, 0x59, 0x99, 0x77, 0xa1, 0x35, 0x08,
	0xc4, 0x14, 0x2f, 0xa3, 0xc7, 0x70, 0x16, 0x89, 0x89, 0x8b, 0x7f, 0xcd, 0x51, 0x2a, 0xdd, 0xb6,
	0x79, 0x92, 0xdb, 0xd5, 0x4b, 0x7e, 0x0c, 0xed, 0x35, 0x65, 0x76, 0x56, 0x0b, 0x4a, 0xbe, 0x2e,
	0xe4, 0xbe, 0x28, 0xe0, 0x6f, 0x0c, 0xa8, 0x5c, 0x8b, 0x85, 0x56, 0xb2, 0xaf, 0xc0, 0x54, 0x8b,
	0x38, 0x15, 0x34, 0x4e, 0xdb, 0x4b, 0x2f, 0x99, 0xc0, 0x5b, 0xc4, 0xe8, 0x92, 0x84, 0x1d, 0xe6,
	0x30, 0xdd, 0x8f, 0x17, 0xbe, 0xe9, 0xf0, 0xab, 0x42, 0xc6, 0x67, 0x5f, 0x80, 0xa9, 0xf0, 0x49,
	0x59, 0x45, 0xd2, 0x35, 0x96, 0x3a, 0x0f, 0x9f, 0xd4, 0x55, 0xc1, 0xa5, 0xaa, 0xc6, 0x29, 0x71,
	0x37, 0x43, 0xcb, 0x5c, 0xc7, 0x79, 0x3a, 0xad, 0x71, 0x54, 0x67, 0x07, 0x50, 0x4f, 0xb2, 0x0b,
	0xdd, 0xfa, 0x13, 0xab, 0xd4, 0x31, 0xba, 0xdb, 0x2e, 0xe4, 0xa9, 0xc1, 0xe4, 0xa2, 0x06, 0x95,
	0x38, 0x75, 0xcb, 0xff, 0x35, 0xa0, 0x3e, 0x42, 0x95, 0xf8, 0xe3, 0x81, 0xc2, 0x40, 0xb2, 0x33,
	0xa8, 0x04, 0x14, 0x4a, 0xcb, 0xa4, 0x6e, 0xef, 0xad, 0x74, 0x7b, 0xa9, 0xcb, 0xd6, 0x6e, 0x2e,
	0xb6, 0x2f, 0xa1, 0x9c, 0xa6, 0x5e, 0xfb, 0x91, 0x32, 0x06, 0xe6, 0x3c, 0xf4, 0xd3, 0x4b, 0xd7,
	0x5c, 0x5a, 0x1f, 0x1d, 0x43, 0x7d, 0xa5, 0x8d, 0xac, 0x06, 0xa5, 0xc1, 0xe8, 0xbc, 0xdf, 0x6b,
	0x16, 0x58, 0x15, 0x4c, 0xaf, 0xf7, 0x87, 0xd7, 0x34, 0x74, 0xd2, 0x3b, 0xbf, 0x18, 0xf6, 0x9a,
	0x5b, 0xa7, 0x6f, 0x4d, 0xd8, 0x19, 0x65, 0xee, 0x6e, 0xd2, 0x39, 0x64, 0x57, 0x50, 0x1b, 0x84,
	0x7f, 0x62, 0x82, 0xe1, 0x18, 0xd9, 0xee, 0xc6, 0xf3, 0xd8, 0x9b, 0x29, 0xde, 0xfe, 0xe7, 0xff,
	0x77, 0xff, 0x6d, 0xed, 0x70, 0x70, 0xfc, 0x7c, 0xe7, 0xf7, 0xc6, 0x11, 0x1b, 0x41, 0xb5, 0x8f,
	0x8a, 0x9e, 0x8a, 0xed, 0xaf, 0xbd, 0xdd, 0xda, 0x27, 0x66, 0x1f, 0x7c, 0xb4, 0x9e, 0x36, 0x9d,
	0x17, 0xd8, 0x19, 0x40, 0x1f, 0x55, 0xda, 0x24, 0xc9, 0x56, 0x5e, 0x8f, 0xa6, 0xdc, 0x6e, 0x7f,
	0xb0, 0xcf, 0xbc, 0xc0, 0x86, 0x64, 0x23, 0x1d, 0x9b, 0x8d, 0x5d, 0xd6, 0x32, 0xf1, 0x72, 0xc2,
	0x56, 0x2e, 0x35, 0x45, 0xe5, 0xd0, 0x7c, 0xe9, 0x4b, 0xdd, 0x00, 0x2c, 0x47, 0x6c, 0x93, 0xb7,
	0xf2, 0xda, 0x9b, 0x93, 0xc8, 0x3f, 0x25, 0x66, 0x93, 0xd7, 0x89, 0x99, 0x4e, 0x9d, 0x86, 0xfe,
	0x0c, 0xf0, 0x53, 0x82, 0x42, 0x21, 0xfd, 0x1f, 0x56, 0xbe, 0x5f, 0x1d, 0xdb, 0xeb, 0x87, 0xf0,
	0xcf, 0x08, 0xb3, 0xcb, 0x3f, 0x71, 0xc6, 0xb4, 0xcb, 0x99, 0x4b, 0x4c, 0x34, 0x67, 0x00, 0xf5,
	0x94, 0x93, 0xfe, 0xbd, 0xd6, 0x07, 0x7d, 0x93, 0x64, 0x11, 0x89, 0xf1, 0xed, 0x9c, 0x44, 0x75,
	0x8d, 0xfa, 0x11, 0x2a, 0x7d, 0x54, 0xaf, 0xf3, 0xd3, 0x22, 0x4a, 0x83, 0xd7, 0xe8, 0x5a, 0x99,
	0x99, 0xbb, 0x32, 0xfd, 0x86, 0xbf, 0x7d, 0x1f, 0x00, 0x00, 0xff, 0xff, 0x87, 0x38, 0x86, 0xdb,
	0xf2, 0x05, 0x00, 0x00,
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// ModelzooServiceClient is the client API for ModelzooService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type ModelzooServiceClient interface {
	// Inference
	Inference(ctx context.Context, in *Payload, opts ...grpc.CallOption) (*Payload, error)
	// Website utils
	GetImage(ctx context.Context, in *ImageDownloadRequest, opts ...grpc.CallOption) (*ImageDownloadResponse, error)
	GetMetrics(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*MetricItems, error)
	// Rate limiting
	GetToken(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*RateLimitToken, error)
	// Database
	ListModels(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*ListModelsResponse, error)
	CreateUser(ctx context.Context, in *User, opts ...grpc.CallOption) (*Empty, error)
	CreateModel(ctx context.Context, in *Model, opts ...grpc.CallOption) (*Empty, error)
	GetUser(ctx context.Context, in *User, opts ...grpc.CallOption) (*Empty, error)
}

type modelzooServiceClient struct {
	cc *grpc.ClientConn
}

func NewModelzooServiceClient(cc *grpc.ClientConn) ModelzooServiceClient {
	return &modelzooServiceClient{cc}
}

func (c *modelzooServiceClient) Inference(ctx context.Context, in *Payload, opts ...grpc.CallOption) (*Payload, error) {
	out := new(Payload)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/Inference", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelzooServiceClient) GetImage(ctx context.Context, in *ImageDownloadRequest, opts ...grpc.CallOption) (*ImageDownloadResponse, error) {
	out := new(ImageDownloadResponse)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/GetImage", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelzooServiceClient) GetMetrics(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*MetricItems, error) {
	out := new(MetricItems)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/GetMetrics", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelzooServiceClient) GetToken(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*RateLimitToken, error) {
	out := new(RateLimitToken)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/GetToken", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelzooServiceClient) ListModels(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*ListModelsResponse, error) {
	out := new(ListModelsResponse)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/ListModels", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelzooServiceClient) CreateUser(ctx context.Context, in *User, opts ...grpc.CallOption) (*Empty, error) {
	out := new(Empty)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/CreateUser", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelzooServiceClient) CreateModel(ctx context.Context, in *Model, opts ...grpc.CallOption) (*Empty, error) {
	out := new(Empty)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/CreateModel", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelzooServiceClient) GetUser(ctx context.Context, in *User, opts ...grpc.CallOption) (*Empty, error) {
	out := new(Empty)
	err := c.cc.Invoke(ctx, "/modelzoo.ModelzooService/GetUser", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ModelzooServiceServer is the server API for ModelzooService service.
type ModelzooServiceServer interface {
	// Inference
	Inference(context.Context, *Payload) (*Payload, error)
	// Website utils
	GetImage(context.Context, *ImageDownloadRequest) (*ImageDownloadResponse, error)
	GetMetrics(context.Context, *Empty) (*MetricItems, error)
	// Rate limiting
	GetToken(context.Context, *Empty) (*RateLimitToken, error)
	// Database
	ListModels(context.Context, *Empty) (*ListModelsResponse, error)
	CreateUser(context.Context, *User) (*Empty, error)
	CreateModel(context.Context, *Model) (*Empty, error)
	GetUser(context.Context, *User) (*Empty, error)
}

// UnimplementedModelzooServiceServer can be embedded to have forward compatible implementations.
type UnimplementedModelzooServiceServer struct {
}

func (*UnimplementedModelzooServiceServer) Inference(ctx context.Context, req *Payload) (*Payload, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Inference not implemented")
}
func (*UnimplementedModelzooServiceServer) GetImage(ctx context.Context, req *ImageDownloadRequest) (*ImageDownloadResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetImage not implemented")
}
func (*UnimplementedModelzooServiceServer) GetMetrics(ctx context.Context, req *Empty) (*MetricItems, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetMetrics not implemented")
}
func (*UnimplementedModelzooServiceServer) GetToken(ctx context.Context, req *Empty) (*RateLimitToken, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetToken not implemented")
}
func (*UnimplementedModelzooServiceServer) ListModels(ctx context.Context, req *Empty) (*ListModelsResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ListModels not implemented")
}
func (*UnimplementedModelzooServiceServer) CreateUser(ctx context.Context, req *User) (*Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method CreateUser not implemented")
}
func (*UnimplementedModelzooServiceServer) CreateModel(ctx context.Context, req *Model) (*Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method CreateModel not implemented")
}
func (*UnimplementedModelzooServiceServer) GetUser(ctx context.Context, req *User) (*Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetUser not implemented")
}

func RegisterModelzooServiceServer(s *grpc.Server, srv ModelzooServiceServer) {
	s.RegisterService(&_ModelzooService_serviceDesc, srv)
}

func _ModelzooService_Inference_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(Payload)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).Inference(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/Inference",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).Inference(ctx, req.(*Payload))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelzooService_GetImage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ImageDownloadRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).GetImage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/GetImage",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).GetImage(ctx, req.(*ImageDownloadRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelzooService_GetMetrics_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(Empty)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).GetMetrics(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/GetMetrics",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).GetMetrics(ctx, req.(*Empty))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelzooService_GetToken_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(Empty)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).GetToken(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/GetToken",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).GetToken(ctx, req.(*Empty))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelzooService_ListModels_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(Empty)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).ListModels(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/ListModels",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).ListModels(ctx, req.(*Empty))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelzooService_CreateUser_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(User)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).CreateUser(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/CreateUser",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).CreateUser(ctx, req.(*User))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelzooService_CreateModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(Model)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).CreateModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/CreateModel",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).CreateModel(ctx, req.(*Model))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelzooService_GetUser_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(User)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelzooServiceServer).GetUser(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/modelzoo.ModelzooService/GetUser",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelzooServiceServer).GetUser(ctx, req.(*User))
	}
	return interceptor(ctx, in, info, handler)
}

var _ModelzooService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "modelzoo.ModelzooService",
	HandlerType: (*ModelzooServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Inference",
			Handler:    _ModelzooService_Inference_Handler,
		},
		{
			MethodName: "GetImage",
			Handler:    _ModelzooService_GetImage_Handler,
		},
		{
			MethodName: "GetMetrics",
			Handler:    _ModelzooService_GetMetrics_Handler,
		},
		{
			MethodName: "GetToken",
			Handler:    _ModelzooService_GetToken_Handler,
		},
		{
			MethodName: "ListModels",
			Handler:    _ModelzooService_ListModels_Handler,
		},
		{
			MethodName: "CreateUser",
			Handler:    _ModelzooService_CreateUser_Handler,
		},
		{
			MethodName: "CreateModel",
			Handler:    _ModelzooService_CreateModel_Handler,
		},
		{
			MethodName: "GetUser",
			Handler:    _ModelzooService_GetUser_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "modelzoo/protos/services.proto",
}
