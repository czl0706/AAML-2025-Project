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
    changes = [] # List of (start_offset, new_data, buffer_obj_pos)

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
            # Old: [K, N/4, 4] -> Access [k][n]
            # New: [N/4, K, 4] -> Access [n][k]
            packed_weights = packed_weights.transpose((1, 0, 2))
            
            # Flatten back to bytes
            new_data = packed_weights.flatten().tobytes()
            
            # We need to find the offset of the OLD data to verify we are replacing the right thing
            # But wait, we are APPENDING new data, not replacing in place (because size changed)
            # So we don't need to find the old data offset for replacement, 
            # but we need the buffer_obj.Pos to update the table.
            
            changes.append({
                'buffer_pos': buffer_obj._tab.Pos,
                'new_data': new_data,
                'old_data_len': len(data)
            })

    # Now apply changes
    # We need to be careful about invalidating offsets if we insert data in the middle.
    # But we are appending to the end, so existing offsets are preserved.
    
    # However, we need to close the model access to resize buf?
    # tflite.Model doesn't hold a lock, but the memoryview 'data' does.
    # We need to make sure 'data' is released.
    del data
    del buffer_obj
    del filter_tensor
    del op
    del subgraph
    del model
    
    # Create a new mutable buffer to avoid BufferError
    new_buf = bytearray(buf)
    
    import struct
    
    for change in changes:
        new_data = change['new_data']
        buffer_pos = change['buffer_pos']
        
        # Append new data
        new_data_offset = len(new_buf)
        new_buf.extend(struct.pack('<I', len(new_data)))
        new_buf.extend(new_data)
        
        # Update Buffer table
        # We need to re-parse or just use the saved pos.
        # The pos is an offset in buf, which hasn't changed for the existing tables.
        
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
