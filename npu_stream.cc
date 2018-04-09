#include "npu_stream.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace npu {

    bool NpuStream::Init() {
        return true;
    }

    void NpuStream::Destroy() {
    }

    bool NpuStream::IsIdle() const {
        return false;
    }

    NpuStream *AsNpuStream(Stream *stream) {
        DCHECK(stream != nullptr);
        return static_cast<NpuStream *>(stream->implementation());
    }

}  // namespace npu
