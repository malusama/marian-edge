#include "marian-mlx/src/ffi.rs.h"

#include "mlx/mlx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace marian_mlx {
namespace {

namespace mx = mlx::core;
using WeightMap = std::unordered_map<std::string, mx::array>;

constexpr int kModelDim = 384;
constexpr int kHeads = 8;
constexpr int kHeadDim = 48;
constexpr int kSourceVocab = 32000;
constexpr int kTargetVocab = 32000;
constexpr float kAttentionScale = 0.14433756729740643f;  // 1 / sqrt(48)
constexpr float kEmbeddingScale = 19.595917942265423f;  // sqrt(384)
constexpr float kLayerNormEpsilon = 1e-6f;
constexpr int kMaximumPosition = 4096;

mx::array require_weight(const WeightMap& weights, const std::string& name) {
  const auto found = weights.find(name);
  if (found == weights.end()) {
    throw std::runtime_error("missing model tensor: " + name);
  }
  return found->second;
}

struct AttentionWeights {
  mx::array wq;
  mx::array wk;
  mx::array wv;
  mx::array wo;
  mx::array bq;
  mx::array bk;
  mx::array bv;
  mx::array bo;
  mx::array norm_scale;
  mx::array norm_bias;

  AttentionWeights(const WeightMap& weights, const std::string& prefix)
      : wq(require_weight(weights, prefix + "_Wq")),
        wk(require_weight(weights, prefix + "_Wk")),
        wv(require_weight(weights, prefix + "_Wv")),
        wo(require_weight(weights, prefix + "_Wo")),
        bq(require_weight(weights, prefix + "_bq")),
        bk(require_weight(weights, prefix + "_bk")),
        bv(require_weight(weights, prefix + "_bv")),
        bo(require_weight(weights, prefix + "_bo")),
        norm_scale(require_weight(weights, prefix + "_Wo_ln_scale")),
        norm_bias(require_weight(weights, prefix + "_Wo_ln_bias")) {}
};

struct FeedForwardWeights {
  mx::array w1;
  mx::array w2;
  mx::array b1;
  mx::array b2;
  mx::array norm_scale;
  mx::array norm_bias;

  FeedForwardWeights(const WeightMap& weights, const std::string& prefix)
      : w1(require_weight(weights, prefix + "_W1")),
        w2(require_weight(weights, prefix + "_W2")),
        b1(require_weight(weights, prefix + "_b1")),
        b2(require_weight(weights, prefix + "_b2")),
        norm_scale(require_weight(weights, prefix + "_ffn_ln_scale")),
        norm_bias(require_weight(weights, prefix + "_ffn_ln_bias")) {}
};

struct SsruWeights {
  mx::array w;
  mx::array wf;
  mx::array bf;
  mx::array norm_scale;
  mx::array norm_bias;

  SsruWeights(const WeightMap& weights, const std::string& prefix)
      : w(require_weight(weights, prefix + "_W")),
        wf(require_weight(weights, prefix + "_Wf")),
        bf(require_weight(weights, prefix + "_bf")),
        norm_scale(require_weight(weights, prefix + "_ffn_ln_scale")),
        norm_bias(require_weight(weights, prefix + "_ffn_ln_bias")) {}
};

struct EncoderLayer {
  AttentionWeights self;
  FeedForwardWeights ffn;

  EncoderLayer(const WeightMap& weights, int layer)
      : self(weights, "encoder_l" + std::to_string(layer) + "_self"),
        ffn(weights, "encoder_l" + std::to_string(layer) + "_ffn") {}
};

struct DecoderLayer {
  SsruWeights ssru;
  AttentionWeights context;
  FeedForwardWeights ffn;

  DecoderLayer(const WeightMap& weights, int layer)
      : ssru(weights, "decoder_l" + std::to_string(layer) + "_rnn"),
        context(weights, "decoder_l" + std::to_string(layer) + "_context"),
        ffn(weights, "decoder_l" + std::to_string(layer) + "_ffn") {}
};

mx::array affine(const mx::array& input,
                 const mx::array& weight,
                 const mx::array& bias) {
  return mx::matmul(input, weight) + bias;
}

mx::array relu(const mx::array& input) {
  return mx::maximum(input, mx::array(0.0f));
}

mx::array layer_norm(const mx::array& input,
                     const mx::array& scale,
                     const mx::array& bias) {
  const auto flat_scale = mx::reshape(scale, {kModelDim});
  const auto flat_bias = mx::reshape(bias, {kModelDim});
  return mx::fast::layer_norm(input, flat_scale, flat_bias,
                              kLayerNormEpsilon);
}

mx::array project_heads(const mx::array& input,
                        const mx::array& weight,
                        const mx::array& bias,
                        int batch,
                        int sequence) {
  auto projected = affine(input, weight, bias);
  projected = mx::reshape(projected, {batch, sequence, kHeads, kHeadDim});
  return mx::transpose(projected, {0, 2, 1, 3});
}

mx::array attend(const mx::array& query,
                 const mx::array& key,
                 const mx::array& value,
                 const mx::array& mask,
                 const AttentionWeights& weights,
                 int batch,
                 int query_length) {
  const auto key_t = mx::transpose(key, {0, 1, 3, 2});
  auto scores = mx::matmul(query, key_t) * kAttentionScale + mask;
  // Marian evaluates attention softmax in fp32. `precise=true` keeps that
  // behavior if an FP16 model is produced later.
  auto probabilities = mx::softmax(scores, -1, true);
  auto context = mx::matmul(probabilities, value);
  context = mx::transpose(context, {0, 2, 1, 3});
  context = mx::reshape(context, {batch, query_length, kModelDim});
  return affine(context, weights.wo, weights.bo);
}

mx::array self_attention(const mx::array& input,
                         const mx::array& mask,
                         const AttentionWeights& weights,
                         int batch,
                         int sequence) {
  const auto query = project_heads(input, weights.wq, weights.bq, batch, sequence);
  const auto key = project_heads(input, weights.wk, weights.bk, batch, sequence);
  const auto value = project_heads(input, weights.wv, weights.bv, batch, sequence);
  return attend(query, key, value, mask, weights, batch, sequence);
}

mx::array feed_forward(const mx::array& input,
                       const FeedForwardWeights& weights) {
  return affine(relu(affine(input, weights.w1, weights.b1)), weights.w2,
                weights.b2);
}

mx::array make_positions(int maximum_position) {
  std::vector<float> values(static_cast<std::size_t>(maximum_position) *
                            kModelDim);
  constexpr int half = kModelDim / 2;
  for (int position = 0; position < maximum_position; ++position) {
    for (int index = 0; index < half; ++index) {
      const float frequency =
          std::exp(-static_cast<float>(index) * std::log(10000.0f) /
                   static_cast<float>(half - 1));
      values[static_cast<std::size_t>(position) * kModelDim + index] =
          std::sin(static_cast<float>(position) * frequency);
      values[static_cast<std::size_t>(position) * kModelDim + half + index] =
          std::cos(static_cast<float>(position) * frequency);
    }
  }
  return mx::array(values.begin(), {maximum_position, kModelDim}, mx::float32);
}

std::uint64_t read_u64(std::ifstream& input) {
  std::array<unsigned char, 8> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) {
    throw std::runtime_error("truncated lexical shortlist header");
  }
  std::uint64_t value = 0;
  for (int index = 7; index >= 0; --index) {
    value = (value << 8) | bytes[static_cast<std::size_t>(index)];
  }
  return value;
}

std::uint32_t read_u32(std::ifstream& input) {
  std::array<unsigned char, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) {
    throw std::runtime_error("truncated lexical shortlist body");
  }
  std::uint32_t value = 0;
  for (int index = 3; index >= 0; --index) {
    value = (value << 8) | bytes[static_cast<std::size_t>(index)];
  }
  return value;
}

class LexicalShortlist {
 public:
  explicit LexicalShortlist(const std::string& path) {
    if (path.empty()) {
      return;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("failed to open lexical shortlist: " + path);
    }
    const auto magic = read_u64(input);
    (void)read_u64(input);  // checksum is verified when the release is fetched.
    first_num_ = read_u64(input);
    (void)read_u64(input);  // bestNum
    const auto offset_count = read_u64(input);
    const auto target_count = read_u64(input);
    if (magic != 17373278592220534773ULL || offset_count != kSourceVocab + 1 ||
        first_num_ > kTargetVocab || target_count > 100000000ULL) {
      throw std::runtime_error("unsupported lexical shortlist format");
    }
    offsets_.reserve(offset_count);
    for (std::uint64_t index = 0; index < offset_count; ++index) {
      offsets_.push_back(read_u64(input));
    }
    targets_.reserve(target_count);
    for (std::uint64_t index = 0; index < target_count; ++index) {
      const auto target = read_u32(input);
      if (target >= kTargetVocab) {
        throw std::runtime_error("lexical shortlist contains invalid target id");
      }
      targets_.push_back(target);
    }
    for (std::size_t index = 0; index < offsets_.size(); ++index) {
      if (offsets_[index] > targets_.size()) {
        throw std::runtime_error(
            "lexical shortlist offset is outside its target body");
      }
      if (index > 0 && offsets_[index] < offsets_[index - 1]) {
        throw std::runtime_error(
            "lexical shortlist offsets are not monotonic");
      }
    }
    if (offsets_.back() != targets_.size()) {
      throw std::runtime_error("lexical shortlist offsets do not match its body");
    }
  }

  std::vector<std::uint32_t> candidates(rust::Slice<const std::int32_t> tokens) const {
    if (offsets_.empty()) {
      std::vector<std::uint32_t> full;
      full.reserve(kTargetVocab - 33);
      for (std::uint32_t id = 0; id < kTargetVocab; ++id) {
        if (id != 1 && !(id >= 7 && id <= 38)) {
          full.push_back(id);
        }
      }
      return full;
    }

    std::vector<bool> present(kTargetVocab, false);
    for (std::uint32_t id = 0; id < first_num_; ++id) {
      present[id] = true;
    }
    for (const auto token : tokens) {
      if (token < 0 || token >= kSourceVocab) {
        throw std::runtime_error("source token id is outside the model vocabulary");
      }
      for (auto index = offsets_[token]; index < offsets_[token + 1]; ++index) {
        present[targets_[index]] = true;
      }
    }

    // Marian disallows UNK and the control-character byte-fallback symbols.
    present[1] = false;
    for (int id = 7; id <= 38; ++id) {
      present[id] = false;
    }
    std::vector<std::uint32_t> result;
    for (std::uint32_t id = 0; id < kTargetVocab; ++id) {
      if (present[id]) {
        result.push_back(id);
      }
    }
    for (std::uint32_t id = 50; result.size() % 8 != 0 && id < kTargetVocab; ++id) {
      if (!present[id]) {
        present[id] = true;
        result.push_back(id);
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  }

 private:
  std::uint64_t first_num_ = 0;
  std::vector<std::uint64_t> offsets_;
  std::vector<std::uint32_t> targets_;
};

mx::ThreadLocalStream initialize_device(const std::string& metallib_path) {
  if (!metallib_path.empty()) {
    mx::metal::set_metallib_path(metallib_path);
  }
  if (!mx::metal::is_available()) {
    throw std::runtime_error("MLX Metal backend is unavailable");
  }
  mx::set_default_device(mx::Device(mx::Device::gpu));
  return mx::new_thread_local_stream(mx::Device(mx::Device::gpu));
}

class EngineImpl final : public Engine {
 public:
  EngineImpl(const std::string& weights_path,
             const std::string& shortlist_path,
             const std::string& metallib_path,
             std::size_t max_length_factor)
      : thread_stream_(initialize_device(metallib_path)),
        stream_(mx::stream_from_thread_local_stream(thread_stream_)),
        shortlist_(shortlist_path),
        max_length_factor_(max_length_factor),
        position_(make_positions(kMaximumPosition)),
        encoder_embedding_(mx::array(0.0f)),
        decoder_embedding_(mx::array(0.0f)),
        output_bias_(mx::array(0.0f)) {
    mx::StreamContext scope(stream_);
    // MLX's file Load primitive is CPU-only. Once materialized, unified-memory
    // arrays are consumed by the GPU graph without reparsing the file.
    auto loaded = mx::load_safetensors(
        weights_path, mx::Device(mx::Device::cpu));
    auto& weights = loaded.first;
    encoder_embedding_ = require_weight(weights, "encoder_Wemb");
    decoder_embedding_ = require_weight(weights, "decoder_Wemb");
    output_bias_ = require_weight(weights, "decoder_ff_logit_out_b");
    for (int layer = 1; layer <= 6; ++layer) {
      encoder_layers_.emplace_back(weights, layer);
    }
    for (int layer = 1; layer <= 4; ++layer) {
      decoder_layers_.emplace_back(weights, layer);
    }

    if (encoder_embedding_.shape() != mx::Shape{kSourceVocab, kModelDim} ||
        decoder_embedding_.shape() != mx::Shape{kTargetVocab, kModelDim}) {
      throw std::runtime_error("model embedding dimensions do not match manifest");
    }
    std::vector<mx::array> materialize;
    materialize.reserve(weights.size() + 1);
    for (const auto& [name, weight] : weights) {
      (void)name;
      materialize.push_back(weight);
    }
    materialize.push_back(position_);
    mx::eval(materialize);
  }

  ~EngineImpl() override {
    mx::synchronize(stream_);
    mx::clear_streams();
  }

  BatchOutput translate(rust::Slice<const std::int32_t> tokens,
                        rust::Slice<const std::uint32_t> offsets,
                        std::size_t max_output_tokens) override {
    mx::StreamContext scope(stream_);
    validate_batch(tokens, offsets);
    const int batch = static_cast<int>(offsets.size() - 1);
    int source_length = 0;
    std::vector<int> lengths(batch);
    for (int index = 0; index < batch; ++index) {
      lengths[index] = static_cast<int>(offsets[index + 1] - offsets[index]);
      source_length = std::max(source_length, lengths[index]);
    }
    if (source_length > kMaximumPosition) {
      throw std::runtime_error("source exceeds the 4096-token position limit");
    }

    std::vector<std::int32_t> padded(
        static_cast<std::size_t>(batch) * source_length, 0);
    std::vector<float> mask_values(
        static_cast<std::size_t>(batch) * source_length, -100000000.0f);
    for (int row = 0; row < batch; ++row) {
      const auto begin = offsets[row];
      for (int column = 0; column < lengths[row]; ++column) {
        padded[static_cast<std::size_t>(row) * source_length + column] =
            tokens[begin + column];
        mask_values[static_cast<std::size_t>(row) * source_length + column] = 0.0f;
      }
    }

    const auto input_ids = mx::array(padded.begin(), {batch, source_length}, mx::int32);
    auto encoder = mx::take(encoder_embedding_, input_ids, 0) * kEmbeddingScale;
    const auto positions =
        mx::slice(position_, {0, 0}, {source_length, kModelDim});
    encoder = encoder + positions;
    const auto mask = mx::array(mask_values.begin(), {batch, 1, 1, source_length},
                                mx::float32);

    for (const auto& layer : encoder_layers_) {
      auto attention = self_attention(encoder, mask, layer.self, batch, source_length);
      encoder = layer_norm(encoder + attention, layer.self.norm_scale,
                           layer.self.norm_bias);
      auto ffn = feed_forward(encoder, layer.ffn);
      encoder = layer_norm(encoder + ffn, layer.ffn.norm_scale,
                           layer.ffn.norm_bias);
    }

    std::vector<mx::array> cross_keys;
    std::vector<mx::array> cross_values;
    cross_keys.reserve(decoder_layers_.size());
    cross_values.reserve(decoder_layers_.size());
    for (const auto& layer : decoder_layers_) {
      cross_keys.push_back(project_heads(encoder, layer.context.wk, layer.context.bk,
                                         batch, source_length));
      cross_values.push_back(project_heads(encoder, layer.context.wv, layer.context.bv,
                                           batch, source_length));
    }
    std::vector<mx::array> cache_arrays = cross_keys;
    cache_arrays.insert(cache_arrays.end(), cross_values.begin(), cross_values.end());
    mx::eval(cache_arrays);

    const auto candidate_ids = shortlist_.candidates(tokens);
    const auto candidate_array = mx::array(candidate_ids.begin(),
                                            {static_cast<int>(candidate_ids.size())},
                                            mx::uint32);
    const auto candidate_embeddings = mx::take(decoder_embedding_, candidate_array, 0);
    const auto candidate_output = mx::transpose(candidate_embeddings);
    const auto candidate_bias = mx::take(output_bias_, candidate_array, 1);
    mx::eval(candidate_embeddings, candidate_output, candidate_bias);

    std::vector<mx::array> states;
    states.reserve(decoder_layers_.size());
    for (std::size_t layer = 0; layer < decoder_layers_.size(); ++layer) {
      states.push_back(mx::zeros({batch, kModelDim}, mx::float32));
    }
    std::vector<std::int32_t> previous(batch, 0);
    std::vector<bool> finished(batch, false);
    std::vector<std::vector<std::int32_t>> generated(batch);
    std::size_t finished_count = 0;
    const auto factor_limit = static_cast<std::size_t>(source_length) *
                              std::max<std::size_t>(max_length_factor_, 1);
    const auto generation_limit = std::min(
        {max_output_tokens, factor_limit,
         static_cast<std::size_t>(kMaximumPosition)});

    for (std::size_t step = 0; step < generation_limit && finished_count < generated.size();
         ++step) {
      mx::array decoder = mx::zeros({batch, kModelDim}, mx::float32);
      if (step > 0) {
        const auto previous_array = mx::array(previous.begin(), {batch}, mx::int32);
        decoder = mx::take(decoder_embedding_, previous_array, 0) * kEmbeddingScale;
      }
      decoder = decoder + mx::take(position_, static_cast<int>(step), 0);

      for (std::size_t index = 0; index < decoder_layers_.size(); ++index) {
        const auto& layer = decoder_layers_[index];
        const auto update = mx::matmul(decoder, layer.ssru.w);
        const auto gate = mx::sigmoid(affine(decoder, layer.ssru.wf, layer.ssru.bf));
        states[index] = gate * states[index] + (1.0f - gate) * update;
        decoder = layer_norm(decoder + relu(states[index]), layer.ssru.norm_scale,
                             layer.ssru.norm_bias);

        auto query = project_heads(mx::reshape(decoder, {batch, 1, kModelDim}),
                                   layer.context.wq, layer.context.bq, batch, 1);
        auto context = attend(query, cross_keys[index], cross_values[index], mask,
                              layer.context, batch, 1);
        context = mx::reshape(context, {batch, kModelDim});
        decoder = layer_norm(decoder + context, layer.context.norm_scale,
                             layer.context.norm_bias);
        auto ffn = feed_forward(decoder, layer.ffn);
        decoder = layer_norm(decoder + ffn, layer.ffn.norm_scale,
                             layer.ffn.norm_bias);
      }

      const auto logits = mx::matmul(decoder, candidate_output) + candidate_bias;
      auto selected = mx::argmax(logits, -1);
      std::vector<mx::array> step_outputs{selected};
      step_outputs.insert(step_outputs.end(), states.begin(), states.end());
      mx::eval(step_outputs);
      const auto* indices = selected.data<std::uint32_t>();
      for (int row = 0; row < batch; ++row) {
        if (finished[row]) {
          continue;
        }
        const auto candidate_index = indices[row];
        if (candidate_index >= candidate_ids.size()) {
          throw std::runtime_error("decoder produced an invalid candidate index");
        }
        const auto token = static_cast<std::int32_t>(candidate_ids[candidate_index]);
        previous[row] = token;
        generated[row].push_back(token);
        if (token == 0) {
          finished[row] = true;
          ++finished_count;
        }
      }
    }

    BatchOutput output;
    output.offsets.reserve(generated.size() + 1);
    output.scores.reserve(generated.size());
    output.offsets.push_back(0);
    for (const auto& sentence : generated) {
      for (const auto token : sentence) {
        output.tokens.push_back(token);
      }
      output.offsets.push_back(static_cast<std::uint32_t>(output.tokens.size()));
      output.scores.push_back(0.0f);
    }
    return output;
  }

  void warmup() override {
    const std::array<std::int32_t, 2> tokens{2, 0};
    const std::array<std::uint32_t, 2> offsets{0, 2};
    (void)translate(
        rust::Slice<const std::int32_t>(tokens.data(), tokens.size()),
        rust::Slice<const std::uint32_t>(offsets.data(), offsets.size()), 4);
  }

  rust::String device_name() const override {
    const auto& info = mx::device_info(mx::Device(mx::Device::gpu));
    const auto found = info.find("device_name");
    if (found != info.end() && std::holds_alternative<std::string>(found->second)) {
      return rust::String(std::get<std::string>(found->second));
    }
    return rust::String("Apple GPU (Metal)");
  }

 private:
  static void validate_batch(rust::Slice<const std::int32_t> tokens,
                             rust::Slice<const std::uint32_t> offsets) {
    if (offsets.size() < 2 || offsets[0] != 0 || offsets[offsets.size() - 1] != tokens.size()) {
      throw std::runtime_error("invalid packed batch offsets");
    }
    if (offsets.size() - 1 > 256) {
      throw std::runtime_error("batch contains more than 256 sentences");
    }
    for (std::size_t index = 0; index + 1 < offsets.size(); ++index) {
      if (offsets[index + 1] <= offsets[index]) {
        throw std::runtime_error("each input must contain at least one token");
      }
    }
    for (const auto token : tokens) {
      if (token < 0 || token >= kSourceVocab) {
        throw std::runtime_error("source token id is outside the model vocabulary");
      }
    }
  }

  mx::ThreadLocalStream thread_stream_;
  mx::Stream stream_;
  LexicalShortlist shortlist_;
  std::size_t max_length_factor_;
  mx::array position_;
  mx::array encoder_embedding_;
  mx::array decoder_embedding_;
  mx::array output_bias_;
  std::vector<EncoderLayer> encoder_layers_;
  std::vector<DecoderLayer> decoder_layers_;
};

}  // namespace

void validate_shortlist(rust::Str shortlist_path) {
  (void)LexicalShortlist(
      std::string(shortlist_path.data(), shortlist_path.size()));
}

std::unique_ptr<Engine> new_engine(rust::Str weights_path,
                                   rust::Str shortlist_path,
                                   rust::Str metallib_path,
                                   std::size_t max_length_factor) {
  return std::make_unique<EngineImpl>(
      std::string(weights_path.data(), weights_path.size()),
      std::string(shortlist_path.data(), shortlist_path.size()),
      std::string(metallib_path.data(), metallib_path.size()), max_length_factor);
}

}  // namespace marian_mlx
