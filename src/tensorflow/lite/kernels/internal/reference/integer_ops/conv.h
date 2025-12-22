/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include <cstdio>
#include "cfu.h"
#include "perf.h"

#define UNUSED(x) (void)(x)

namespace tflite {
namespace reference_integer_ops {

static inline uint32_t pack4_u8(uint8_t b3, uint8_t b2, uint8_t b1, uint8_t b0) {
  return (uint32_t(b3) << 24) | (uint32_t(b2) << 16) | (uint32_t(b1) << 8) | uint32_t(b0);
}

constexpr int kMaxIm2ColRows = 148;
constexpr int kMaxIm2ColCols = 8000;
constexpr int kMaxOutputDepth = 2000;

static int32_t mm_result[kMaxIm2ColRows][kMaxOutputDepth];

struct Im2ColPacker1D {
  int M;
  int input_width;
  int pad_width;
  int8_t neg_in_off;
  const RuntimeShape* input_shape;
  const int8_t* input_data;
};

// stride=1
inline uint32_t pack_A_s1(
    const Im2ColPacker1D& p, int mg, int in_channel, int fx) {

  const int r0 = (mg << 2);
  const int r1 = r0 + 1;
  const int r2 = r0 + 2;
  const int r3 = r0 + 3;

  const int base = -p.pad_width + fx;   // in_x = out_x + base

  auto load = [&](int out_x)->uint8_t {
    if (out_x >= p.M) [[unlikely]] return 0;           // tail padding
    const int in_x = out_x + base;
    if (!((uint32_t)in_x < (uint32_t)p.input_width)) [[unlikely]]
      return (uint8_t)p.neg_in_off;                    // image padding
    return (uint8_t)p.input_data[Offset(*p.input_shape, 0, 0, in_x, in_channel)];
  };

  return pack4_u8(load(r0), load(r1), load(r2), load(r3));
}

// stride=2
inline uint32_t pack_A_s2(
    const Im2ColPacker1D& p, int mg, int in_channel, int fx) {

  const int r0 = (mg << 2);
  const int r1 = r0 + 1;
  const int r2 = r0 + 2;
  const int r3 = r0 + 3;

  const int base = -p.pad_width + fx;   // in_x = 2*out_x + base

  auto load = [&](int out_x)->uint8_t {
    if (out_x >= p.M) [[unlikely]] return 0;
    const int in_x = (out_x << 1) + base;
    if (!((uint32_t)in_x < (uint32_t)p.input_width)) [[unlikely]]
      return (uint8_t)p.neg_in_off;
    return (uint8_t)p.input_data[Offset(*p.input_shape, 0, 0, in_x, in_channel)];
  };

  return pack4_u8(load(r0), load(r1), load(r2), load(r3));
}


// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int pad_width = params.padding_values.width;
  // const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  // const int32_t output_activation_min = -128;
  // const int32_t output_activation_max =  127;

  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

  // Check dimensions of the tensors.
  // const int input_height = 1;
  const int input_width = input_shape.Dims(2);
  const int filter_height = 1;
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int output_height = 1;
  const int output_width = output_shape.Dims(2);

  const int8_t neg_in_off = static_cast<int8_t>(-input_offset);

  const int M = output_height * output_width;                         // im2col_rows
  const int K = filter_height * filter_width * filter_input_depth;    // kernel_rows
  // const int img_off = filter_height * filter_width;
  const int M4 = (M + 3) >> 2;
  const int N  = output_depth;
  const int N4 = (N + 3) >> 2;

  // perf_enable_counter(3);
  // for (int k = 0; k < K; ++k) {
  //   // k -> (in_channel, filter_y, filter_x)
  //   const int in_channel = k / img_off;
  //   const int rem = k - in_channel * img_off;
  //   const int filter_y = rem / filter_width;
  //   const int filter_x = rem - filter_y * filter_width;

  //   #pragma GCC unroll 4
  //   for (int mg = 0; mg < M4; ++mg) {
  //     const int r0 = (mg << 2) + 0;
  //     const int r1 = (mg << 2) + 1;
  //     const int r2 = (mg << 2) + 2;
  //     const int r3 = (mg << 2) + 3;

  //     auto load_row = [&](int r)->uint8_t {
  //       if (r >= M) [[unlikely]] return 0;  // GEMM tail padding -> 0

  //       const int out_y = r / output_width;
  //       const int out_x = r - out_y * output_width;

  //       const int in_y_origin = out_y - pad_height;
  //       const int in_x_origin = (out_x * stride_width) - pad_width;

  //       const int in_y = in_y_origin + filter_y;
  //       const int in_x = in_x_origin + filter_x;

  //       const bool inside =
  //         ((uint32_t)in_x < (uint32_t)input_width) &&
  //         ((uint32_t)in_y < (uint32_t)input_height);

  //       if (!inside) [[unlikely]] return (uint8_t)neg_in_off; // image padding -> neg_in_off
  //       return (uint8_t)input_data[Offset(input_shape, 0, in_y, in_x, in_channel)];
  //     };

  //     const uint8_t a3 = load_row(r0);
  //     const uint8_t a2 = load_row(r1);
  //     const uint8_t a1 = load_row(r2);
  //     const uint8_t a0 = load_row(r3);

  //     // Transposed packing: [mg][k]
  //     m_im2col_packed[mg][k] = pack4_u8(a3,a2,a1,a0);
  //   }
  // }
  // perf_disable_counter(3);

  // Shape of matrices:
  // m_im2col_packed: [M/4][K]
  // packed_weights:  [N/4][K]
  // mm_result:       [M][N]
  const int TILE_M = 148; // im2col_rows
  const int TILE_K = 256; // kernel_rows
  const int TILE_N = 256; // output_depth

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      mm_result[row][col] = 0;
    }
  }

  // Assume weights are always packed
  const uint32_t* __restrict__ packed_weights_ptr = reinterpret_cast<const uint32_t*>(filter_data);

  // perf_enable_counter(2);
  for (int krnl_y = 0; krnl_y < K; krnl_y += TILE_K) {
    const int kk = std::min(TILE_K, K - krnl_y);
    const int ky = krnl_y + kk;

    const int img_y = 0;
    const int mm  = std::min(TILE_M, M - img_y);
    const int my  = img_y + mm;
    const int mg0 = (img_y >> 2);
    const int mg1 = std::min(M4, (my + 3) >> 2); // ceil(my/4)

    // ===== Initialize the Im2Col packer (once per function call) =====
    Im2ColPacker1D packer;
    packer.M = M;
    packer.input_width = input_width;
    packer.pad_width = pad_width;
    packer.neg_in_off = neg_in_off;
    packer.input_shape = &input_shape;
    packer.input_data  = input_data;

    // ===== A tile load: ONCE per K-tile =====
    cfu_op0(2, 0, 0);  // reset A stream counter

    // The loop iterates over the range mg0..mg1 and k=krnl_y..ky.
    // Optimization: Generate k from (in_channel, fx) to avoid expensive division operations.
    const int k0 = krnl_y;
    const int k1 = ky;

    // Map the current k-range [k0, k1) to the corresponding input channel range.
    const int ic0 = k0 / filter_width;
    const int ic1 = (k1 + filter_width - 1) / filter_width; // ceil
    // Note: ic1 is clamped by the tile boundary, so it may be less than filter_input_depth.

    if (stride_width == 1) [[likely]] {
      for (int mg = mg0; mg < mg1; ++mg) {

        // Iterate over the input channels within the current tile.
        for (int in_channel = ic0; in_channel < ic1; ++in_channel) {

          // Calculate the base index in the flattened kernel matrix (K) for this channel.
          const int k_base = in_channel * filter_width;

          // Iterate through the filter width. We only process indices that fall within the current tile's k-range [k0, k1).
          // This check is typically optimized away or handled efficiently by branch prediction.
          #pragma GCC unroll 4
          for (int fx = 0; fx < filter_width; ++fx) {
            const int k = k_base + fx;
            if ((unsigned)(k - k0) >= (unsigned)(k1 - k0)) [[unlikely]] continue;

            cfu_op0(3, 0, pack_A_s1(packer, mg, in_channel, fx));
          }
        }
      }
    } else { // stride_width == 2
      for (int mg = mg0; mg < mg1; ++mg) {
        for (int in_channel = ic0; in_channel < ic1; ++in_channel) {
          const int k_base = in_channel * filter_width;

          #pragma GCC unroll 4
          for (int fx = 0; fx < filter_width; ++fx) {
            const int k = k_base + fx;
            if ((unsigned)(k - k0) >= (unsigned)(k1 - k0)) [[unlikely]] continue;

            cfu_op0(3, 0, pack_A_s2(packer, mg, in_channel, fx));
          }
        }
      }
    }

    for (int krnl_x = 0; krnl_x < N; krnl_x += TILE_N) {
      const int nn = std::min(TILE_N, N - krnl_x);
      const int kx = krnl_x + nn;

      // ---- compute ng range exactly (no break in loop)
      const int ng0 = (krnl_x >> 2);
      const int ng1 = std::min(N4, (kx + 3) >> 2);

      // ============================
      // Load B tile into CFU (per N-tile)
      // ============================
      cfu_op0(2, 0, 0);  // reset shared counter for B stream
      for (int ng = ng0; ng < ng1; ++ng) {
        const int base = ng * K;
        #pragma GCC unroll 4
        for (int k = krnl_y; k < ky; ++k) {
          cfu_op0(4, 0, packed_weights_ptr[base + k]); // [ng][k]
        }
      }

      // Start CFU matmul
      cfu_op0(1, input_offset, (mm << 20) | (kk << 10) | nn);
      while (cfu_op0(0, 0, 0)) { }

      // Reset address generator
      cfu_op0(6, (krnl_x << 16) | img_y, mm);

      // Read results
      for (int row = img_y; row < my; ++row) {
        #pragma GCC unroll 4
        for (int col = krnl_x; col < kx; col += 4) {
          mm_result[row][col + 0] += cfu_op0(5, 0, 0);
          mm_result[row][col + 1] += cfu_op0(5, 0, 1);
          mm_result[row][col + 2] += cfu_op0(5, 0, 2);
          mm_result[row][col + 3] += cfu_op0(5, 0, 3);
        }
      }
    }
  }

  // perf_disable_counter(2);

  // ----------------------------
  // Requantize + clamp
  // ----------------------------
  // perf_enable_counter(3);
  constexpr int CHUNK = 256;

  for (int base = 0; base < output_depth; base += CHUNK) {
    const int chunk_size = std::min(CHUNK, output_depth - base);
    const int padded_chunk_size = std::max(72, (chunk_size + 3) & ~3);

    // 1) Reset CFU and set offset
    cfu_op0(20, 1, output_offset);

    // 2) Load Params (branchless: two-phase)
    for (int j = 0; j < chunk_size; ++j) {
      const int ch = base + j;
      const uint32_t args_in0 =
          (uint32_t(uint16_t(bias_data[ch])) << 16) |
          uint32_t(uint16_t(output_shift[ch]));
      const uint32_t args_in1 = uint32_t(output_multiplier[ch]);
      cfu_op0(21, args_in0, args_in1);
    }
    for (int j = chunk_size; j < padded_chunk_size; ++j) {
      cfu_op0(21, 0, 0);
    }

    // 3) Process Pixels
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int row = out_x;

      // Push Accumulators (branchless: two-phase)
      const int32_t* accp = &mm_result[row][base];
      for (int j = 0; j < chunk_size; ++j) {
        cfu_op0(22, j, accp[j]);
      }
      for (int j = chunk_size; j < padded_chunk_size; ++j) {
        cfu_op0(22, j, 0);
      }

      // Read Results
      int8_t* outp = output_data + out_x * output_depth + base;  // NHWC fast offset

      cfu_op0(20, 0, 0);  // Reset read ptr

      // Process full groups of 4. It's safe to store 4 bytes because of padding, 
      // but we only care about valid data up to chunk_size.
      int j = 0;
      for (; j + 3 < chunk_size; j += 4) {
        const uint32_t packed = cfu_op0(23, 0, 0);
        outp[j + 0] = int8_t(uint8_t(packed >>  0));
        outp[j + 1] = int8_t(uint8_t(packed >>  8));
        outp[j + 2] = int8_t(uint8_t(packed >> 16));
        outp[j + 3] = int8_t(uint8_t(packed >> 24));
      }

      // tail of the elements
      if (j < chunk_size) {
        const uint32_t packed = cfu_op0(23, 0, 0);
        if (j + 0 < chunk_size) outp[j + 0] = int8_t(uint8_t(packed >>  0));
        if (j + 1 < chunk_size) outp[j + 1] = int8_t(uint8_t(packed >>  8));
        if (j + 2 < chunk_size) outp[j + 2] = int8_t(uint8_t(packed >> 16));
        // The case when (j+3 < chunk_size) happens is already handled by the main loop
      }
    }
  }
  // perf_disable_counter(3);
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
