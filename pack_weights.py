# /// script
# dependencies = [
#     "tflite",
#     "numpy",
# ]
# ///
import tflite
import numpy as np
import sys
import struct

def pack_weights(input_path, output_path):
    with open(input_path, 'rb') as f:
        buf = bytearray(f.read())

    model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    # Find CONV_2D opcode index
    conv_opcode_index = -1
    for i in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(i)
        builtin_code = op_code.BuiltinCode()
        if builtin_code == tflite.BuiltinOperator.CONV_2D:
            conv_opcode_index = i
            break
    
    if conv_opcode_index == -1:
        print("No CONV_2D operator found.")
        return

    print(f"Found CONV_2D opcode index: {conv_opcode_index}")

    # Collect all changes first
    changes = [] 

    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        if op.OpcodeIndex() == conv_opcode_index:
            print(f"Processing CONV_2D at operator index {i}")
            
            filter_tensor_index = op.Inputs(1)
            filter_tensor = subgraph.Tensors(filter_tensor_index)
            
            buffer_index = filter_tensor.Buffer()
            buffer_obj = model.Buffers(buffer_index)
            
            data = buffer_obj.DataAsNumpy()
            if data is None or data.size == 0:
                print("  Filter has no data (dynamic tensor?). Skipping.")
                continue
                
            shape = filter_tensor.ShapeAsNumpy()
            N, H, W, C_in = shape
            print(f"  Filter shape: {shape} (N, H, W, C_in)")
            
            original_weights = data.reshape((N, H, W, C_in))
            transposed_weights = original_weights.transpose((3, 1, 2, 0))
            
            K = C_in * H * W
            flat_weights = transposed_weights.reshape((K, N))
            
            if N % 4 != 0:
                pad_len = 4 - (N % 4)
                print(f"  Padding output depth {N} with {pad_len} zeros to {N + pad_len}")
                padding = np.zeros((K, pad_len), dtype=flat_weights.dtype)
                flat_weights = np.concatenate((flat_weights, padding), axis=1)
                N = N + pad_len
            
            # Reshape to [K, N/4, 4]
            packed_weights = flat_weights.reshape((K, N // 4, 4))
            
            # Reverse the last dimension (c0, c1, c2, c3 -> c3, c2, c1, c0)
            packed_weights = packed_weights[:, :, ::-1]
            
            # Transpose to [N/4, K, 4] to optimize memory access pattern
            packed_weights = packed_weights.transpose((1, 0, 2))
            
            # Flatten back to bytes
            new_data = packed_weights.flatten().tobytes()
            
            # Check if we can overwrite
            pos = buffer_obj._tab.Pos
            # Read vtable offset from the ORIGINAL buffer
            vtable_offset = struct.unpack_from('<i', buf, pos)[0]
            vtable_pos = pos - vtable_offset
            # Field 0 is 'data' in Buffer table
            field_0_offset = struct.unpack_from('<H', buf, vtable_pos + 4)[0]
            
            if field_0_offset == 0:
                print("  Error: Buffer table has no data field. Appending.")
                changes.append({
                    'type': 'append',
                    'buffer_pos': pos,
                    'data': new_data
                })
                continue

            field_pos = pos + field_0_offset
            relative_offset = struct.unpack_from('<I', buf, field_pos)[0]
            data_vector_offset = field_pos + relative_offset
            
            # Read existing data length
            existing_data_len = struct.unpack_from('<I', buf, data_vector_offset)[0]
            
            if len(new_data) <= existing_data_len:
                print(f"  Overwriting existing buffer at {data_vector_offset} (size {existing_data_len} -> {len(new_data)})")
                changes.append({
                    'type': 'overwrite',
                    'offset': data_vector_offset,
                    'data': new_data
                })
            else:
                print(f"  Appending new buffer (size {existing_data_len} -> {len(new_data)})")
                changes.append({
                    'type': 'append',
                    'buffer_pos': pos,
                    'data': new_data
                })

    # Release TFLite objects
    del data
    del buffer_obj
    del filter_tensor
    del op
    del subgraph
    del model
    
    # Create a new mutable buffer
    new_buf = bytearray(buf)
    
    for change in changes:
        if change['type'] == 'overwrite':
            offset = change['offset']
            new_data = change['data']
            # Update length
            struct.pack_into('<I', new_buf, offset, len(new_data))
            # Update data
            # We must be careful not to extend the buffer if we are just overwriting
            # bytearray slice assignment works
            new_buf[offset + 4 : offset + 4 + len(new_data)] = new_data
            
        elif change['type'] == 'append':
            new_data = change['data']
            buffer_pos = change['buffer_pos']
            
            # Append new data
            new_data_offset = len(new_buf)
            new_buf.extend(struct.pack('<I', len(new_data)))
            new_buf.extend(new_data)
            
            # Update Buffer table
            # We need to re-read offsets from new_buf because we might have modified it?
            # No, Buffer table offsets (vtable etc) are relative and inside the table, which we haven't moved.
            # But we need to find the field_pos again.
            
            pos = buffer_pos
            vtable_offset = struct.unpack_from('<i', new_buf, pos)[0]
            vtable_pos = pos - vtable_offset
            field_0_offset = struct.unpack_from('<H', new_buf, vtable_pos + 4)[0]
            
            if field_0_offset == 0:
                print(f"  Error: Buffer table at {pos} has no data field.")
                continue
                
            field_pos = pos + field_0_offset
            new_relative_offset = new_data_offset - field_pos
            struct.pack_into('<I', new_buf, field_pos, new_relative_offset)
            print(f"  Updated Buffer table at {pos} to point to new data at {new_data_offset}")

    with open(output_path, 'wb') as f:
        f.write(new_buf)
    print(f"Saved packed model to {output_path}")

if __name__ == "__main__":
    pack_weights("wav2letter_pruned_int8.tflite", "wav2letter_pruned_int8_packed.tflite")
