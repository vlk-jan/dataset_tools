// Minimal C ABI wrapper around cloudini_lib for use from Python (ctypes).
// Operates on the cloudini-encoded blob (the msg.compressed_data field of a
// point_cloud_interfaces/msg/CompressedPointCloud2).
//
// Build: see build_shim.sh in this directory.

#include "cloudini_lib/cloudini.hpp"

extern "C" {

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

}  // extern "C"
