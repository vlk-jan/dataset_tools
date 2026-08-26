// Minimal C ABI wrapper around cloudini_lib for use from Python (ctypes).
// Operates either on the raw CDR-serialized point_cloud_interfaces/msg/
// CompressedPointCloud2 message ("dds" variants) or directly on the
// cloudini-encoded blob (the msg.compressed_data field).
//
// Build: see build_shim.sh in this directory.

#include <cstring>
#include <string>

#include "cloudini_lib/cloudini.hpp"
#include "cloudini_lib/ros_msg_utils.hpp"

extern "C" {

// Writes the EncodingInfo of an encoded blob as YAML into `out` (at most
// `cap` bytes). Returns the YAML length, or 0 on failure.
uint32_t cldn_shim_header_yaml(const uint8_t* data, uint64_t size, char* out, uint32_t cap) {
  try {
    Cloudini::ConstBufferView view(data, size);
    const auto info = Cloudini::DecodeHeader(view);
    const std::string yaml = Cloudini::EncodingInfoToYAML(info);
    if (yaml.size() > cap) {
      return 0;
    }
    std::memcpy(out, yaml.data(), yaml.size());
    return static_cast<uint32_t>(yaml.size());
  } catch (const std::exception&) {
    return 0;
  }
}

// Decoded size (width * height * point_step) of an encoded blob, 0 on failure.
uint64_t cldn_shim_decoded_size(const uint8_t* data, uint64_t size) {
  try {
    Cloudini::ConstBufferView view(data, size);
    const auto info = Cloudini::DecodeHeader(view);
    return static_cast<uint64_t>(info.width) * info.height * info.point_step;
  } catch (const std::exception&) {
    return 0;
  }
}

// Decodes an encoded blob into `out` (`cap` bytes available). Returns the
// number of bytes written, or 0 on failure.
uint64_t cldn_shim_decode(const uint8_t* data, uint64_t size, uint8_t* out, uint64_t cap) {
  try {
    Cloudini::ConstBufferView view(data, size);
    Cloudini::EncodingInfo info = Cloudini::DecodeHeader(view);
    const uint64_t decoded_size =
        static_cast<uint64_t>(info.width) * info.height * info.point_step;
    if (decoded_size > cap) {
      return 0;
    }
    Cloudini::BufferView out_view(out, decoded_size);
    Cloudini::PointcloudDecoder decoder;
    decoder.decode(info, view, out_view);
    return decoded_size;
  } catch (const std::exception&) {
    return 0;
  }
}

// Same as above, but starting from the raw CDR-serialized
// CompressedPointCloud2 message instead of the blob.
uint64_t cldn_shim_decoded_size_dds(const uint8_t* data, uint64_t size) {
  try {
    const auto msg = cloudini_ros::getDeserializedPointCloudMessage({data, size});
    return static_cast<uint64_t>(msg.width) * msg.height * msg.point_step;
  } catch (const std::exception&) {
    return 0;
  }
}

uint64_t cldn_shim_decode_dds(const uint8_t* data, uint64_t size, uint8_t* out, uint64_t cap) {
  try {
    const auto msg = cloudini_ros::getDeserializedPointCloudMessage({data, size});
    return cldn_shim_decode(msg.data.data(), msg.data.size(), out, cap);
  } catch (const std::exception&) {
    return 0;
  }
}

uint32_t cldn_shim_header_yaml_dds(const uint8_t* data, uint64_t size, char* out, uint32_t cap) {
  try {
    const auto msg = cloudini_ros::getDeserializedPointCloudMessage({data, size});
    return cldn_shim_header_yaml(msg.data.data(), msg.data.size(), out, cap);
  } catch (const std::exception&) {
    return 0;
  }
}

}  // extern "C"
