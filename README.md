# AAML 2025 Final Project: Wav2Letter Acceleration

This project designs and implements an accelerator on CFU-Playground to offload 1D convolution, Leaky ReLU, and quantization operations, targeting efficient inference of the wav2letter model.

## Usage

### Setup
```bash
# Clone the project
cd ${CFU_ROOT}/proj
git clone https://github.com/czl0706/AAML-2025-Project.git
```

Different from the `Prepare the Model File` section of the [Lab website](https://nycu-caslab.github.io/AAML2025/project/final_project.html#prepare-the-model-file), this project requires transforming the model weights into a custom layout first.

The following steps first transform the model weights into the custom layout, and then convert the model into a header file that can be used by CFU-Playground.
```bash
wget https://github.com/ARM-software/ML-Zoo/raw/master/models/speech_recognition/wav2letter/tflite_pruned_int8/wav2letter_pruned_int8.tflite

# By PEP 723 (Inline script metadata), we can run the script without setting the environment
# uv run convert_model.py

# Or just install tflite and numpy manually and run the script
python3 convert_model.py

# Move the packed model to the model directory
mv wav2letter_pruned_int8_packed.tflite src/wav2letter/model/wav2letter_pruned_int8.tflite

# Convert the model to header file
cd src/wav2letter/model
chmod +x model_convert.sh
./model_convert.sh
```

### Performance Test
- Steps:
    1. make prog && make load
    2. Reboot LiteX.
    3. Close the litex-term terminal (Critical! To free up the UART port).
    4. Run the script:
        ```bash
        python eval.py --port /dev/ttyUSB1 (or any serial you are using)

        # Or use PEP 723 (Inline script metadata) to run the script
        # This script only adds the script metadata compared to the original ones
        uv run eval_uv.py --port /dev/ttyUSB1 (or any serial you are using)
        ```

## Overview

The accelerator is designed to offload computationally intensive operations using custom CFU operations.  
In hardware, it features a systolic array, a SIMD Leaky ReLU unit, and a hardware-accelerated quantization unit implemented in Verilog.  
In software, ...

## Key Features

### 1. Systolic Array for Matrix Multiplication
- **Tiled Computation**: Implements a tiled matrix multiplication architecture to handle matrix multiplication operations efficiently.
- **Systolic Array**: Utilizes a systolic array-like structure for parallel MAC (Multiply-Accumulate) operations.

### 2. SIMD Leaky ReLU
- **Parallel Processing**: Instantiates 8 parallel Leaky ReLU units to process 8 data points simultaneously.
- **Packed I/O**: Packs 8-bit input/output values into 64-bit words for limited bandwidth between the CPU and CFU.

### 3. Hardware Quantization
- **Per-Channel Quantization**: Offloads the complex per-channel quantization logic to the hardware.
- **Packed Output**: Returns quantized 8-bit results packed into 32-bit words, reducing the number of bus transactions.

### 4. Software Optimization
- **Offline weight packing**: Packs the 1D convolution weights offline to reduce runtime overhead.
- **Implicit Im2Col**: Packs the input activations at runtime without constructing the Im2Col matrix explicitly.
- **Efficient CFU Interaction**: Minimizes overhead by batching CFU commands and using efficient data packing.

## File Structure

- **`cfu.v`**: Top-level Verilog module for the Custom Function Unit. It orchestrates the TPU, Leaky ReLU, and Quantization units.
- **`TPU.v`**: Verilog implementation of the matrix multiplication unit.
- **`leaky_relu.v`**: Verilog implementation of the Leaky ReLU activation function.
- **`oc_quantize.v`**: Verilog implementation of the output channel quantization logic.
- **`src/tensorflow/lite/kernels/internal/reference/integer_ops/conv.h`**: Modified TensorFlow Lite kernel header integrating the CFU operations into the convolution reference implementation.
- **`src/tensorflow/lite/kernels/internal/reference/leaky_relu.h`**: Modified TensorFlow Lite kernel header integrating the CFU operations into the Leaky ReLU reference implementation.