/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "perf.h"
#include "cfu.h"
#include <cstdio>

namespace tflite {
namespace reference_ops {

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    // Note that alpha might be > 1 or < 0, so we don't use std::max here.
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32_t quantized_min = std::numeric_limits<T>::min();
  static const int32_t quantized_max = std::numeric_limits<T>::max();
  for (int i = 0; i < flat_size; ++i) {
    const int32_t input_value = input_data[i] - params.input_offset;
    int32_t unclamped_output;
    if (input_value >= 0) {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_identity,
                             params.output_shift_identity);
    } else {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_alpha,
                             params.output_shift_alpha);
    }
    const T clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

template <>
inline void QuantizeLeakyRelu<int8_t>(const LeakyReluParams& params,
                                      const RuntimeShape& input_shape,
                                      const int8_t* input_data,
                                      const RuntimeShape& output_shape,
                                      int8_t* output_data) {
  // perf_enable_counter(6);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // Configure CFU
  cfu_op0(10, params.input_offset, params.output_offset);
  cfu_op0(11, params.output_multiplier_alpha, params.output_shift_alpha);
  cfu_op0(12, params.output_multiplier_identity, params.output_shift_identity);

  const int chunk_size = 200;
  for (int i = 0; i < flat_size; i += chunk_size) {
    // Send inputs
    for (int j = 0; j < chunk_size; j += 8) {
      uint32_t w0 = *((uint32_t*)(input_data + i + j));
      uint32_t w1 = *((uint32_t*)(input_data + i + j + 4));
      cfu_op0(13, w0, w1);
    }

    // Retrieve outputs
    const int groups = chunk_size / 8;
    for (int g = 0; g < groups; ++g) {
      int addr = g;
      uint32_t in1_0 = ((uint32_t)addr << 1) | 0;
      uint32_t in1_1 = ((uint32_t)addr << 1) | 1;

      uint32_t out0 = cfu_op0(14, 0, in1_0);
      uint32_t out1 = cfu_op0(14, 0, in1_1);

      uint32_t* p = (uint32_t*)(output_data + i + g * 8);
      p[0] = out0;
      p[1] = out1;
    }
  }
  // perf_disable_counter(6);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
