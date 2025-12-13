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
        endcase
    end 
end

wire        A_wr_en   = (cmd_valid && funct7 == 3);
wire [13:0] A_index   = A_wr_en ? cmd_payload_inputs_0[13:0] : A_index_TPU[13:0];
wire [31:0] A_data_in = cmd_payload_inputs_1;
wire [31:0] A_data_out;

wire        B_wr_en   = (cmd_valid && funct7 == 4);
wire [13:0] B_index   = B_wr_en ? cmd_payload_inputs_0[13:0] : B_index_TPU[13:0];
wire [31:0] B_data_in = cmd_payload_inputs_1;
wire [31:0] B_data_out;

wire         C_wr_en   = C_wr_en_TPU;
wire [ 13:0] C_index   = (cmd_valid && funct7 == 5) ? cmd_payload_inputs_0[13:0] : C_index_TPU[13:0];
wire [127:0] C_data_in = C_data_in_TPU;
wire [127:0] C_data_out;

reg [ 9:0] M;
reg [ 9:0] K;
reg [ 9:0] N;
reg [31:0] offset;
reg        in_valid;

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
end

wire         busy_TPU;

wire [31:0]  A_data_out_TPU = A_data_out;
wire [31:0]  B_data_out_TPU = B_data_out;
wire [127:0] C_data_out_TPU = C_data_out;

wire [15:0]  A_index_TPU; 
wire [15:0]  B_index_TPU; 
wire [15:0]  C_index_TPU; 

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

endmodule