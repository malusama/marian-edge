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
constant uint FLASH_ATTENTION_THREADS = 32;
constant uint FLASH_ATTENTION_MAX_HEAD_DIM = 64;
constant uint FLASH_ATTENTION_QUERY_TILE = 4;
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

// Forward-only FlashAttention-style kernel for the Marian inference graph.
// Each SIMD group owns four query rows for one (batch, head) pair. It streams
// K/V tiles, maintains four online softmax states, and never writes the O(N^2)
// score matrix to device memory. Blocking queries lets the group reuse every
// K/V load four times. The current model has head_dim=48, so each lane owns at
// most two dimensions per query and reductions use SIMD-group intrinsics.
kernel void attention_flash_f32(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const uint* lengths [[buffer(2)]],
    device const float* values [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant AttentionParams& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const uint query_start = group.x * FLASH_ATTENTION_QUERY_TILE;
  const uint batch_head = group.y;
  if (query_start >= p.query_length || batch_head >= p.batch * p.heads) return;

  const uint batch = batch_head / p.heads;
  const uint head = batch_head % p.heads;
  const uint head_dim = p.dim / p.heads;
  if (head_dim > FLASH_ATTENTION_MAX_HEAD_DIM) return;

  const bool has_query1 = query_start + 1 < p.query_length;
  const bool has_query2 = query_start + 2 < p.query_length;
  const bool has_query3 = query_start + 3 < p.query_length;
  const uint active_keys = min(lengths[batch], p.key_length);
  const uint query_base0 = (batch * p.query_length + query_start) * p.dim
      + head * head_dim;
  const uint query_base1 = query_base0 + p.dim;
  const uint query_base2 = query_base1 + p.dim;
  const uint query_base3 = query_base2 + p.dim;
  threadgroup float query0[64];
  threadgroup float query1[64];
  threadgroup float query2[64];
  threadgroup float query3[64];
  threadgroup float probability0[FLASH_ATTENTION_THREADS];
  threadgroup float probability1[FLASH_ATTENTION_THREADS];
  threadgroup float probability2[FLASH_ATTENTION_THREADS];
  threadgroup float probability3[FLASH_ATTENTION_THREADS];

  query0[tid] = tid < head_dim ? queries[query_base0 + tid] : 0.0f;
  query1[tid] = has_query1 && tid < head_dim ? queries[query_base1 + tid] : 0.0f;
  query2[tid] = has_query2 && tid < head_dim ? queries[query_base2 + tid] : 0.0f;
  query3[tid] = has_query3 && tid < head_dim ? queries[query_base3 + tid] : 0.0f;
  if (tid + FLASH_ATTENTION_THREADS < head_dim) {
    query0[tid + FLASH_ATTENTION_THREADS] =
        queries[query_base0 + tid + FLASH_ATTENTION_THREADS];
    query1[tid + FLASH_ATTENTION_THREADS] = has_query1
        ? queries[query_base1 + tid + FLASH_ATTENTION_THREADS] : 0.0f;
    query2[tid + FLASH_ATTENTION_THREADS] = has_query2
        ? queries[query_base2 + tid + FLASH_ATTENTION_THREADS] : 0.0f;
    query3[tid + FLASH_ATTENTION_THREADS] = has_query3
        ? queries[query_base3 + tid + FLASH_ATTENTION_THREADS] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float maximum0 = -INFINITY, maximum1 = -INFINITY;
  float maximum2 = -INFINITY, maximum3 = -INFINITY;
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  float low0 = 0.0f, low1 = 0.0f, low2 = 0.0f, low3 = 0.0f;
  float high0 = 0.0f, high1 = 0.0f, high2 = 0.0f, high3 = 0.0f;

  for (uint start = 0; start < active_keys; start += FLASH_ATTENTION_THREADS) {
    const uint tile_keys = min(FLASH_ATTENTION_THREADS, active_keys - start);
    float score0 = -INFINITY, score1 = -INFINITY;
    float score2 = -INFINITY, score3 = -INFINITY;
    if (tid < tile_keys) {
      const uint key_base = (batch * p.key_length + start + tid) * p.dim
          + head * head_dim;
      score0 = 0.0f;
      score1 = 0.0f;
      score2 = 0.0f;
      score3 = 0.0f;
      for (uint index = 0; index < head_dim; ++index) {
        const float key_value = keys[key_base + index];
        score0 += query0[index] * key_value;
        score1 += query1[index] * key_value;
        score2 += query2[index] * key_value;
        score3 += query3[index] * key_value;
      }
      score0 *= ATTENTION_SCALE;
      score1 *= ATTENTION_SCALE;
      score2 *= ATTENTION_SCALE;
      score3 *= ATTENTION_SCALE;
    }

    const float next_maximum0 = max(maximum0, simd_max(score0));
    const float next_maximum1 = max(maximum1, simd_max(score1));
    const float next_maximum2 = max(maximum2, simd_max(score2));
    const float next_maximum3 = max(maximum3, simd_max(score3));
    const float scale0 = maximum0 == -INFINITY ? 0.0f : exp(maximum0 - next_maximum0);
    const float scale1 = maximum1 == -INFINITY ? 0.0f : exp(maximum1 - next_maximum1);
    const float scale2 = maximum2 == -INFINITY ? 0.0f : exp(maximum2 - next_maximum2);
    const float scale3 = maximum3 == -INFINITY ? 0.0f : exp(maximum3 - next_maximum3);
    const float probability_value0 = tid < tile_keys ? exp(score0 - next_maximum0) : 0.0f;
    const float probability_value1 = has_query1 && tid < tile_keys
        ? exp(score1 - next_maximum1) : 0.0f;
    const float probability_value2 = has_query2 && tid < tile_keys
        ? exp(score2 - next_maximum2) : 0.0f;
    const float probability_value3 = has_query3 && tid < tile_keys
        ? exp(score3 - next_maximum3) : 0.0f;
    probability0[tid] = probability_value0;
    probability1[tid] = probability_value1;
    probability2[tid] = probability_value2;
    probability3[tid] = probability_value3;
    sum0 = sum0 * scale0 + simd_sum(probability_value0);
    sum1 = sum1 * scale1 + simd_sum(probability_value1);
    sum2 = sum2 * scale2 + simd_sum(probability_value2);
    sum3 = sum3 * scale3 + simd_sum(probability_value3);
    maximum0 = next_maximum0;
    maximum1 = next_maximum1;
    maximum2 = next_maximum2;
    maximum3 = next_maximum3;
    low0 *= scale0;
    low1 *= scale1;
    low2 *= scale2;
    low3 *= scale3;
    high0 *= scale0;
    high1 *= scale1;
    high2 *= scale2;
    high3 *= scale3;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
      for (uint index = 0; index < tile_keys; ++index) {
        const uint value_index = (batch * p.key_length + start + index) * p.dim
            + head * head_dim + tid;
        const float value = values[value_index];
        low0 += probability0[index] * value;
        low1 += probability1[index] * value;
        low2 += probability2[index] * value;
        low3 += probability3[index] * value;
      }
    }
    const uint high_dimension = tid + FLASH_ATTENTION_THREADS;
    if (high_dimension < head_dim) {
      for (uint index = 0; index < tile_keys; ++index) {
        const uint value_index = (batch * p.key_length + start + index) * p.dim
            + head * head_dim + high_dimension;
        const float value = values[value_index];
        high0 += probability0[index] * value;
        high1 += probability1[index] * value;
        high2 += probability2[index] * value;
        high3 += probability3[index] * value;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid < head_dim) {
    output[query_base0 + tid] = low0 / sum0;
    if (has_query1) output[query_base1 + tid] = low1 / sum1;
    if (has_query2) output[query_base2 + tid] = low2 / sum2;
    if (has_query3) output[query_base3 + tid] = low3 / sum3;
  }
  if (tid + FLASH_ATTENTION_THREADS < head_dim) {
    const uint high_dimension = tid + FLASH_ATTENTION_THREADS;
    output[query_base0 + high_dimension] = high0 / sum0;
    if (has_query1) output[query_base1 + high_dimension] = high1 / sum1;
    if (has_query2) output[query_base2 + high_dimension] = high2 / sum2;
    if (has_query3) output[query_base3 + high_dimension] = high3 / sum3;
  }
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
