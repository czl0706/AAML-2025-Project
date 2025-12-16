// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


`include "TPU.v"
`include "global_buffer_bram.v"
`include "leaky_relu.v"

module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output              rsp_valid,
    input               rsp_ready,
    output reg [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
);

// Combinational CFU
assign rsp_valid = cmd_valid;
assign cmd_ready = rsp_ready;

wire [6:0] funct7 = cmd_payload_function_id[9:3];
// Mode 0: Get TPU busy status
// Mode 1: Set K, M, N and start computation
// Mode 2: Not used
// Mode 3: Write A buffer
// Mode 4: Write B buffer
// Mode 5: Read C buffer

always @(*) begin
    rsp_payload_outputs_0 = 0;
    if (cmd_valid) begin
        case (funct7)
            0: rsp_payload_outputs_0 = busy_TPU;
            5: rsp_payload_outputs_0 = cmd_payload_inputs_1 == 3 ? C_data_out[ 31: 0] :
                                       cmd_payload_inputs_1 == 2 ? C_data_out[ 63:32] :
                                       cmd_payload_inputs_1 == 1 ? C_data_out[ 95:64] :
                                       cmd_payload_inputs_1 == 0 ? C_data_out[127:96] : 32'b0;
            14: rsp_payload_outputs_0 = cmd_payload_inputs_1[0] ? lrelu_bram_data_out[63:32] : lrelu_bram_data_out[31:0];
        endcase
    end 
end

reg [13:0] write_index;

wire        A_wr_en   = (cmd_valid && funct7 == 3);
wire [13:0] A_index   = A_wr_en ? write_index : A_index_TPU[13:0];
wire [31:0] A_data_in = cmd_payload_inputs_1;
wire [31:0] A_data_out;

wire        B_wr_en   = (cmd_valid && funct7 == 4);
wire [13:0] B_index   = B_wr_en ? write_index : B_index_TPU[13:0];
wire [31:0] B_data_in = cmd_payload_inputs_1;
wire [31:0] B_data_out;

wire         C_wr_en   = C_wr_en_TPU;
wire [ 13:0] C_index   = (cmd_valid && funct7 == 5) ? read_index : C_index_TPU[13:0];
wire [127:0] C_data_in = C_data_in_TPU;
wire [127:0] C_data_out;

reg [ 9:0] M;
reg [ 9:0] K;
reg [ 9:0] N;
reg [31:0] offset;
reg        in_valid;

reg [13:0] read_index;
reg [ 9:0] read_r;
reg [ 9:0] read_c;

// Leaky ReLU Registers
reg signed [31:0] lrelu_input_offset;
reg signed [31:0] lrelu_output_offset;
reg signed [31:0] lrelu_output_multiplier_alpha;
reg signed [31:0] lrelu_output_shift_alpha;
reg signed [31:0] lrelu_output_multiplier_identity;
reg signed [31:0] lrelu_output_shift_identity;

reg lrelu_in_valid;
reg [7:0] lrelu_input_val [0:7];

wire [7:0] lrelu_out_valid;
wire [7:0] lrelu_output_val [0:7];

reg [7:0] lrelu_seq_num;
wire [63:0] lrelu_bram_data_out;

wire [ 9:0] Nb_M = M[9:2] + |M[1:0];
wire [13:0] stride_col = {4'b0, Nb_M, 2'b00};

always @(posedge clk) begin
    if (cmd_valid && funct7 == 1) begin
        M <= cmd_payload_inputs_1[29:20];
        K <= cmd_payload_inputs_1[19:10];
        N <= cmd_payload_inputs_1[ 9: 0];

        offset   <= cmd_payload_inputs_0;
        in_valid <= 1'b1;
    end
    else begin
        in_valid <= 1'b0;
    end

    // LReLU Logic
    if (reset) begin
        lrelu_input_offset               <= 32'd0;
        lrelu_output_offset              <= 32'd0;
        lrelu_output_multiplier_alpha    <= 32'd0;
        lrelu_output_shift_alpha         <= 32'd0;
        lrelu_output_multiplier_identity <= 32'd0;
        lrelu_output_shift_identity      <= 32'd0;

        lrelu_input_val[0] <= 8'd0;
        lrelu_input_val[1] <= 8'd0;
        lrelu_input_val[2] <= 8'd0;
        lrelu_input_val[3] <= 8'd0;
        lrelu_input_val[4] <= 8'd0;
        lrelu_input_val[5] <= 8'd0;
        lrelu_input_val[6] <= 8'd0;
        lrelu_input_val[7] <= 8'd0;
        lrelu_in_valid     <= 1'b0;
        lrelu_seq_num      <= 0;
    end else begin
        lrelu_in_valid <= 0;
        if (cmd_valid) begin
            case (funct7)
                14: begin
                    lrelu_seq_num <= 0;
                end
                10: begin
                    lrelu_seq_num       <= 0;
                    lrelu_input_offset  <= cmd_payload_inputs_0;
                    lrelu_output_offset <= cmd_payload_inputs_1;                
                end
                11: begin
                    lrelu_output_multiplier_alpha <= cmd_payload_inputs_0;
                    lrelu_output_shift_alpha      <= cmd_payload_inputs_1;
                end
                12: begin
                    lrelu_output_multiplier_identity <= cmd_payload_inputs_0;
                    lrelu_output_shift_identity      <= cmd_payload_inputs_1;
                end
                13: begin
                    lrelu_in_valid <= 1;
                    {lrelu_input_val[3], lrelu_input_val[2], lrelu_input_val[1], lrelu_input_val[0]} <= cmd_payload_inputs_0;
                    {lrelu_input_val[7], lrelu_input_val[6], lrelu_input_val[5], lrelu_input_val[4]} <= cmd_payload_inputs_1;
                end
            endcase
        end

        if (lrelu_out_valid[0]) begin
            lrelu_seq_num <= lrelu_seq_num + 1;
        end
    end

    if (cmd_valid && funct7 == 2) begin
        write_index <= 0;
    end else if (A_wr_en || B_wr_en) begin
        write_index <= write_index + 1;
    end

    // Reset read_index (funct7=6)
    if (cmd_valid && funct7 == 6) begin
        read_index <= 0;
        read_r <= 0;
        read_c <= 0;
    end 
    // Auto-increment read_index when reading the last word (index 3)
    else if (cmd_valid && funct7 == 5 && cmd_payload_inputs_1 == 3) begin
        if (read_c + 4 >= N) begin
            // End of row, move to next row
            read_c <= 0;
            read_r <= read_r + 1;
            read_index <= {4'b0, read_r} + 1;
        end else begin
            // Next column block
            read_c <= read_c + 4;
            read_index <= read_index + stride_col;
        end
    end
end

wire         busy_TPU;

wire [31:0]  A_data_out_TPU = A_data_out;
wire [31:0]  B_data_out_TPU = B_data_out;
wire [127:0] C_data_out_TPU = C_data_out;

wire [15:0]  A_index_TPU; 
wire [15:0]  B_index_TPU; 
wire [15:0]  C_index_TPU; 

wire         C_wr_en_TPU;
wire [127:0] C_data_in_TPU; 

TPU u_TPU (
    .clk            (clk),     
    .rst_n          (~reset),   

    .in_valid       (in_valid),         
    .K              (K), 
    .M              (M), 
    .N              (N), 
    .busy           (busy_TPU),     

    .A_offset       (offset),

    .A_wr_en        (),         
    .A_index        (A_index_TPU),         
    .A_data_in      (),         
    .A_data_out     (A_data_out_TPU),         

    .B_wr_en        (),         
    .B_index        (B_index_TPU),         
    .B_data_in      (),         
    .B_data_out     (B_data_out_TPU),         

    .C_wr_en        (C_wr_en_TPU),         
    .C_index        (C_index_TPU),         
    .C_data_in      (C_data_in_TPU),         
    .C_data_out     (C_data_out_TPU)     
);

global_buffer_bram #( .ADDR_BITS(14), .DATA_BITS(32) ) 
gbuff_A (
    .clk     (clk       ),
    .rst_n   (1'b1      ),
    .ram_en  (1'b1      ),
    .wr_en   (A_wr_en   ),
    .index   (A_index   ),
    .data_in (A_data_in ),
    .data_out(A_data_out)
), 
gbuff_B (
    .clk     (clk       ),
    .rst_n   (1'b1      ),
    .ram_en  (1'b1      ),
    .wr_en   (B_wr_en   ),
    .index   (B_index   ),
    .data_in (B_data_in ),
    .data_out(B_data_out)
);

global_buffer_bram #( .ADDR_BITS(14), .DATA_BITS(128) ) 
gbuff_C (
    .clk     (clk       ),
    .rst_n   (1'b1      ),
    .ram_en  (1'b1      ),
    .wr_en   (C_wr_en   ),
    .index   (C_index   ),
    .data_in (C_data_in ),
    .data_out(C_data_out)
);

genvar i;
generate
    for (i = 0; i < 8; i = i + 1) begin : lrelu_gen
        leaky_relu LReLU (
            .clk(clk),
            .rst(reset),

            .in_valid(lrelu_in_valid),
            .input_val(lrelu_input_val[i]),

            .input_offset(lrelu_input_offset),
            .output_offset(lrelu_output_offset),
            .output_multiplier_alpha(lrelu_output_multiplier_alpha),
            .output_shift_alpha(lrelu_output_shift_alpha),
            .output_multiplier_identity(lrelu_output_multiplier_identity),
            .output_shift_identity(lrelu_output_shift_identity),

            .out_valid(lrelu_out_valid[i]),
            .output_val(lrelu_output_val[i])
        );
    end
endgenerate

global_buffer_bram #( .ADDR_BITS(5), .DATA_BITS(64) ) 
gbuff_LReLU (
    .clk     (clk       ),
    .rst_n   (1'b1      ),
    .ram_en  (1'b1      ),
    .wr_en   (lrelu_out_valid[0] ),
    .index   ((cmd_valid && funct7 == 14) ? cmd_payload_inputs_1[31:1] : lrelu_seq_num),
    .data_in ({lrelu_output_val[7], lrelu_output_val[6], lrelu_output_val[5], lrelu_output_val[4], lrelu_output_val[3], lrelu_output_val[2], lrelu_output_val[1], lrelu_output_val[0]}),
    .data_out(lrelu_bram_data_out)
);

endmodule