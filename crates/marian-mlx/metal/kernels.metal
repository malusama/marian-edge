#include <metal_stdlib>
using namespace metal;

inline float model_value(device const uchar* values, uint index, uint storage) {
  if (storage == 0) {
    return reinterpret_cast<device const float*>(values)[index];
  }
  return float(reinterpret_cast<device const half*>(values)[index]);
}

constant uint TILE = 16;
constant uint REDUCTION_THREADS = 128;
constant float EMBEDDING_SCALE = 19.595917942265423f;
constant float ATTENTION_SCALE = 0.14433756729740643f;

struct MatMulParams {
  uint rows;
  uint cols;
  uint inner;
  uint has_bias;
  uint activation;
  uint storage;
};

kernel void matmul_f32(
    device const float* lhs [[buffer(0)]],
    device const uchar* rhs [[buffer(1)]],
    device const uchar* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant MatMulParams& p [[buffer(4)]],
    ushort2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup float lhs_tile[TILE][TILE];
  threadgroup float rhs_tile[TILE][TILE];
  const uint row = group.y * TILE + local.y;
  const uint col = group.x * TILE + local.x;
  float value = 0.0f;

  for (uint start = 0; start < p.inner; start += TILE) {
    const uint lhs_col = start + local.x;
    const uint rhs_row = start + local.y;
    lhs_tile[local.y][local.x] =
        row < p.rows && lhs_col < p.inner ? lhs[row * p.inner + lhs_col] : 0.0f;
    rhs_tile[local.y][local.x] =
        rhs_row < p.inner && col < p.cols
            ? model_value(rhs, rhs_row * p.cols + col, p.storage) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = 0; index < TILE; ++index) {
      value += lhs_tile[local.y][index] * rhs_tile[index][local.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < p.rows && col < p.cols) {
    value += p.has_bias != 0 ? model_value(bias, col, p.storage) : 0.0f;
    output[row * p.cols + col] = p.activation != 0 ? max(value, 0.0f) : value;
  }
}

struct EmbeddingParams {
  uint batch;
  uint sequence;
  uint dim;
  uint storage;
};

kernel void embedding_positions_f32(
    device const int* token_ids [[buffer(0)]],
    device const uchar* embedding [[buffer(1)]],
    device const float* positions [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant EmbeddingParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.dim || gid.y >= p.sequence || gid.z >= p.batch) return;
  const uint token_offset = gid.z * p.sequence + gid.y;
  const uint token = uint(token_ids[token_offset]);
  const uint output_offset = token_offset * p.dim + gid.x;
  output[output_offset] = model_value(embedding, token * p.dim + gid.x, p.storage) * EMBEDDING_SCALE
      + positions[gid.y * p.dim + gid.x];
}

struct DecoderInputParams {
  uint batch;
  uint dim;
  uint position;
  uint storage;
};

kernel void decoder_input_f32(
    device const int* previous [[buffer(0)]],
    device const uchar* embedding [[buffer(1)]],
    device const float* positions [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant DecoderInputParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.dim || gid.y >= p.batch) return;
  const int token = previous[gid.y];
  const float embedded = token >= 0
      ? model_value(embedding, uint(token) * p.dim + gid.x, p.storage) : 0.0f;
  output[gid.y * p.dim + gid.x] = embedded * EMBEDDING_SCALE
      + positions[p.position * p.dim + gid.x];
}

struct NormParams {
  uint rows;
  uint dim;
  uint storage;
};

kernel void residual_layer_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const uchar* scale [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant NormParams& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= p.rows) return;
  const uint base = row * p.dim;
  threadgroup float reduction[REDUCTION_THREADS];
  float local_sum = 0.0f;
  for (uint index = tid; index < p.dim; index += REDUCTION_THREADS) {
    const float value = input[base + index] + residual[base + index];
    output[base + index] = value;
    local_sum += value;
  }
  reduction[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = reduction[0] / float(p.dim);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_variance = 0.0f;
  for (uint index = tid; index < p.dim; index += REDUCTION_THREADS) {
    const float centered = output[base + index] - mean;
    local_variance += centered * centered;
  }
  reduction[tid] = local_variance;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_std = rsqrt(reduction[0] / float(p.dim) + 1.0e-6f);
  for (uint index = tid; index < p.dim; index += REDUCTION_THREADS) {
    output[base + index] = (output[base + index] - mean) * inv_std
        * model_value(scale, index, p.storage) + model_value(bias, index, p.storage);
  }
}

kernel void ssru_update_layer_norm_f32(
    device const float* candidate [[buffer(0)]],
    device const float* forget_pre [[buffer(1)]],
    device float* state [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device const uchar* scale [[buffer(4)]],
    device const uchar* bias [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant NormParams& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= p.rows) return;
  const uint base = row * p.dim;
  threadgroup float reduction[REDUCTION_THREADS];
  float local_sum = 0.0f;
  for (uint index = tid; index < p.dim; index += REDUCTION_THREADS) {
    const uint offset = base + index;
    const float gate = 1.0f / (1.0f + exp(-forget_pre[offset]));
    const float next = gate * state[offset] + (1.0f - gate) * candidate[offset];
    state[offset] = next;
    const float value = residual[offset] + max(next, 0.0f);
    output[offset] = value;
    local_sum += value;
  }
  reduction[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = reduction[0] / float(p.dim);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_variance = 0.0f;
  for (uint index = tid; index < p.dim; index += REDUCTION_THREADS) {
    const float centered = output[base + index] - mean;
    local_variance += centered * centered;
  }
  reduction[tid] = local_variance;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_std = rsqrt(reduction[0] / float(p.dim) + 1.0e-6f);
  for (uint index = tid; index < p.dim; index += REDUCTION_THREADS) {
    output[base + index] = (output[base + index] - mean) * inv_std
        * model_value(scale, index, p.storage) + model_value(bias, index, p.storage);
  }
}

struct AttentionParams {
  uint batch;
  uint query_length;
  uint key_length;
  uint dim;
  uint heads;
};

kernel void attention_scores_f32(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const uint* lengths [[buffer(2)]],
    device float* scores [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint key_index = gid.x;
  const uint query_index = gid.y;
  const uint batch_head = gid.z;
  if (key_index >= p.key_length || query_index >= p.query_length
      || batch_head >= p.batch * p.heads) return;
  const uint batch = batch_head / p.heads;
  const uint head = batch_head % p.heads;
  const uint head_dim = p.dim / p.heads;
  const uint query_base = (batch * p.query_length + query_index) * p.dim
      + head * head_dim;
  const uint key_base = (batch * p.key_length + key_index) * p.dim
      + head * head_dim;
  float value = 0.0f;
  for (uint index = 0; index < head_dim; ++index) {
    value += queries[query_base + index] * keys[key_base + index];
  }
  if (key_index >= lengths[batch]) value = -100000000.0f;
  else value *= ATTENTION_SCALE;
  const uint offset = ((batch_head * p.query_length + query_index) * p.key_length)
      + key_index;
  scores[offset] = value;
}

kernel void attention_softmax_f32(
    device float* scores [[buffer(0)]],
    constant AttentionParams& p [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const uint query_index = group.x;
  const uint batch_head = group.y;
  if (query_index >= p.query_length || batch_head >= p.batch * p.heads) return;
  const uint base = (batch_head * p.query_length + query_index) * p.key_length;
  threadgroup float reduction[REDUCTION_THREADS];
  float local_maximum = -INFINITY;
  for (uint index = tid; index < p.key_length; index += REDUCTION_THREADS) {
    local_maximum = max(local_maximum, scores[base + index]);
  }
  reduction[tid] = local_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] = max(reduction[tid], reduction[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float maximum = reduction[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_sum = 0.0f;
  for (uint index = tid; index < p.key_length; index += REDUCTION_THREADS) {
    const float value = exp(scores[base + index] - maximum);
    scores[base + index] = value;
    local_sum += value;
  }
  reduction[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inverse_sum = 1.0f / reduction[0];
  for (uint index = tid; index < p.key_length; index += REDUCTION_THREADS) {
    scores[base + index] *= inverse_sum;
  }
}

kernel void attention_apply_f32(
    device const float* scores [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant AttentionParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint dim_index = gid.x;
  const uint query_index = gid.y;
  const uint batch = gid.z;
  if (dim_index >= p.dim || query_index >= p.query_length || batch >= p.batch) return;
  const uint head_dim = p.dim / p.heads;
  const uint head = dim_index / head_dim;
  const uint score_base = (((batch * p.heads + head) * p.query_length + query_index)
      * p.key_length);
  float value = 0.0f;
  for (uint key_index = 0; key_index < p.key_length; ++key_index) {
    value += scores[score_base + key_index]
        * values[(batch * p.key_length + key_index) * p.dim + dim_index];
  }
  output[(batch * p.query_length + query_index) * p.dim + dim_index] = value;
}

struct OutputParams {
  uint batch;
  uint candidates;
  uint dim;
  uint storage;
};

kernel void output_logits_f32(
    device const float* decoder [[buffer(0)]],
    device const uchar* embedding [[buffer(1)]],
    device const uchar* bias [[buffer(2)]],
    device const uint* candidate_ids [[buffer(3)]],
    device const uint* candidate_counts [[buffer(4)]],
    device float* logits [[buffer(5)]],
    constant OutputParams& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint candidate_index = gid.x;
  const uint batch = gid.y;
  if (candidate_index >= p.candidates || batch >= p.batch) return;
  if (candidate_index >= candidate_counts[batch]) {
    logits[batch * p.candidates + candidate_index] = -INFINITY;
    return;
  }
  const uint token = candidate_ids[batch * p.candidates + candidate_index];
  float value = model_value(bias, token, p.storage);
  for (uint index = 0; index < p.dim; ++index) {
    value += decoder[batch * p.dim + index]
        * model_value(embedding, token * p.dim + index, p.storage);
  }
  logits[batch * p.candidates + candidate_index] = value;
}

kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    device const uint* candidate_counts [[buffer(1)]],
    device uint* selected [[buffer(2)]],
    constant OutputParams& p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
  if (batch >= p.batch) return;
  const uint base = batch * p.candidates;
  threadgroup float values[REDUCTION_THREADS];
  threadgroup uint indices[REDUCTION_THREADS];
  float best_value = -INFINITY;
  uint best_index = 0xffffffffu;
  for (uint index = tid; index < candidate_counts[batch]; index += REDUCTION_THREADS) {
    const float value = logits[base + index];
    if (value > best_value || (value == best_value && index < best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_value = values[tid + stride];
      const uint other_index = indices[tid + stride];
      if (other_value > values[tid]
          || (other_value == values[tid] && other_index < indices[tid])) {
        values[tid] = other_value;
        indices[tid] = other_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) selected[batch] = indices[0];
}
