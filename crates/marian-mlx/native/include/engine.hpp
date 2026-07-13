#pragma once

#include "rust/cxx.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace marian_mlx {

struct BatchOutput;

class Engine {
 public:
  virtual ~Engine() = default;
  virtual BatchOutput translate(rust::Slice<const std::int32_t> tokens,
                                rust::Slice<const std::uint32_t> offsets,
                                std::size_t max_output_tokens) = 0;
  virtual void warmup() = 0;
  virtual rust::String device_name() const = 0;
};

void validate_shortlist(rust::Str shortlist_path);

std::unique_ptr<Engine> new_engine(rust::Str weights_path,
                                   rust::Str shortlist_path,
                                   rust::Str metallib_path,
                                   std::size_t max_length_factor);

}  // namespace marian_mlx
