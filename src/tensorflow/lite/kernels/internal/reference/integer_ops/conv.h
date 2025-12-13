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

#include "cfu.h"
#include "perf.h"
#define UNUSED(x) (void)(x)

namespace tflite {
namespace reference_integer_ops {

constexpr int kMaxIm2ColRows = 148;
constexpr int kMaxIm2ColCols = 8000;
constexpr int kMaxOutputDepth = 2000;

int8_t  m_kernel[kMaxOutputDepth][kMaxIm2ColCols];
int8_t  m_im2col[kMaxIm2ColRows][kMaxIm2ColCols];
int32_t mm_result[kMaxIm2ColRows][kMaxOutputDepth];

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  perf_enable_counter(6);

  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

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

  UNUSED(filters_per_group);
  UNUSED(batches);

  const int8_t neg_in_off = static_cast<int8_t>(-input_offset);

  const int img_off = filter_height * filter_width;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
      const int row = out_y * output_width + out_x;

      for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y =
              in_y_origin + filter_y * dilation_height_factor;
          const int off_y = filter_y * filter_width;

          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            const int in_x =
                in_x_origin + filter_x * dilation_width_factor;
            const int col = in_channel * img_off + off_y + filter_x;

            const bool is_inside =
                ((uint32_t)in_x < (uint32_t)input_width) &&
                ((uint32_t)in_y < (uint32_t)input_height);

            m_im2col[row][col] = is_inside
                ? input_data[Offset(input_shape, 0, in_y, in_x, in_channel)]
                : neg_in_off;
          }
        }
      }
    }
  }

  for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      const int off_y = filter_y * filter_width;
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        const int row = in_channel * img_off + off_y + filter_x;
        for (int col = 0; col < output_depth; ++col) {
          m_kernel[col][row] = filter_data[Offset(
                    filter_shape, col, filter_y, filter_x, in_channel)];
        }
      }
    }
  }

  // Shape of matrices:
  // m_im2col:   [output_height*output_width , filter_height*filter_width*filter_input_depth]
  // m_kernel:   [filter_height*filter_width*filter_input_depth , output_depth]
  // mm_result:  [output_height*output_width , output_depth]
  const int TILE_M = 148; // im2col_rows
  const int TILE_K = 256; // kernel_rows
  const int TILE_N = 256; // output_depth
  // const int T = 16;
  const int im2col_rows = output_height * output_width;
  const int kernel_rows = filter_height * filter_width * filter_input_depth;

  // // print the shape of the matrices
  // printf("im2col_rows: %d\n", im2col_rows);
  // printf("kernel_rows: %d\n", kernel_rows);
  // printf("output_depth: %d\n", output_depth);

  for (int row = 0; row < im2col_rows; ++row) {
    for (int col = 0; col < output_depth; ++col) {
      mm_result[row][col] = 0;
    }
  }

  for (int krnl_y = 0; krnl_y < kernel_rows; krnl_y += TILE_K) {
    const int kk = std::min(TILE_K, kernel_rows - krnl_y);
    const int ky = krnl_y + kk;

    for (int krnl_x = 0; krnl_x < output_depth; krnl_x += TILE_N) {
      const int nn = std::min(TILE_N, output_depth - krnl_x);
      const int kx = krnl_x + nn;

      int idx = 0;
      int8_t cfu_in[4];

      // Load matrix B (kernel tiles) into CFU
      for (int col = krnl_x; col < kx; col += 4) {
        for (int row = krnl_y; row < ky; ++row) {
          cfu_in[3] = (row < kernel_rows && col + 0 < output_depth)
              ? m_kernel[col + 0][row] : 0;
          cfu_in[2] = (row < kernel_rows && col + 1 < output_depth)
              ? m_kernel[col + 1][row] : 0;
          cfu_in[1] = (row < kernel_rows && col + 2 < output_depth)
              ? m_kernel[col + 2][row] : 0;
          cfu_in[0] = (row < kernel_rows && col + 3 < output_depth)
              ? m_kernel[col + 3][row] : 0;

          cfu_op0(4, idx, *(int32_t*)cfu_in);
          ++idx;
        }
      }

      for (int img_y = 0; img_y < im2col_rows; img_y += TILE_M) {
        const int mm = std::min(TILE_M, im2col_rows - img_y);
        const int my = img_y + mm;

        idx = 0;  
        // Load matrix A (im2col tiles) into CFU
        for (int row = img_y; row < my; row += 4) {
          for (int col = krnl_y; col < ky; ++col) {
            cfu_in[3] = (row + 0 < im2col_rows && col < kernel_rows)
                ? m_im2col[row + 0][col] : neg_in_off;
            cfu_in[2] = (row + 1 < im2col_rows && col < kernel_rows)
                ? m_im2col[row + 1][col] : neg_in_off;
            cfu_in[1] = (row + 2 < im2col_rows && col < kernel_rows)
                ? m_im2col[row + 2][col] : neg_in_off;
            cfu_in[0] = (row + 3 < im2col_rows && col < kernel_rows)
                ? m_im2col[row + 3][col] : neg_in_off;

            cfu_op0(3, idx, *(int32_t*)cfu_in);
            ++idx;
          }
        }

        // Configure and start CFU matmul
        cfu_op0(1, input_offset, mm << 20 | kk << 10 | nn);
        while (cfu_op0(0, 0, 0)) { }

        for (int row = img_y; row < my; ++row) {
          for (int col = krnl_x; col < kx; col += 4) {
            const int col_off = col - krnl_x;
            const int row_off = row - img_y;
            const int base_idx = row_off + (col_off >> 2) * mm;

            mm_result[row][col + 0] += cfu_op0(5, base_idx, 0);
            mm_result[row][col + 1] += cfu_op0(5, base_idx, 1);
            mm_result[row][col + 2] += cfu_op0(5, base_idx, 2);
            mm_result[row][col + 3] += cfu_op0(5, base_idx, 3);
          }
        }
      }
    }
  }

  for (int out_y = 0; out_y < output_height; ++out_y) {
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int row = out_y * output_width + out_x;
      for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
        int32_t acc = mm_result[row][out_channel] + bias_data[out_channel];

          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);

        output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
      }
    }
  }

  perf_disable_counter(6);
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
