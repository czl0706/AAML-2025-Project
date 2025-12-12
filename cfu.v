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

`include "global_buffer_bram.v"
`include "leaky_relu.v"
`include "mac_unit.v"
// `include "oc_quantize.v"

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

// Trivial handshaking for a combinational CFU
assign rsp_valid = cmd_valid;
assign cmd_ready = rsp_ready;

wire [6:0] funct7 = cmd_payload_function_id[9:3];

// Leaky ReLU
reg signed [31:0] lr_input_offset;
reg signed [31:0] lr_output_offset;
reg signed [31:0] lr_output_multiplier_alpha;
reg signed [31:0] lr_output_shift_alpha;
reg signed [31:0] lr_output_multiplier_identity;
reg signed [31:0] lr_output_shift_identity;

reg lr_in_valid;
reg [7:0] lr_in [0:7];

wire lr_out_valid;
wire [7:0] lr_out [0:7];
wire [7:0] lr_out_valid_arr;

assign lr_out_valid = lr_out_valid_arr[0];

reg [7:0] lr_seq_num;

wire [63:0] lr_data_out;

// Conv MAC Units
reg signed [31:0] conv_input_offset;
reg signed [7:0] conv_in_buf [0:7];
reg conv_en;
reg conv_clr;
reg signed [7:0] conv_weights [0:7];
wire signed [31:0] mac_acc [0:7][0:7];

// control block
always @(posedge clk) begin
    if (reset) begin
        lr_input_offset <= 32'd0; lr_output_offset <= 32'd0; lr_output_multiplier_alpha <= 32'd0; 
        lr_output_shift_alpha <= 32'd0; lr_output_multiplier_identity <= 32'd0; lr_output_shift_identity <= 32'd0; 
        lr_in[0] <= 8'd0; lr_in[1] <= 8'd0; lr_in[2] <= 8'd0; lr_in[3] <= 8'd0; 
        lr_in[4] <= 8'd0; lr_in[5] <= 8'd0; lr_in[6] <= 8'd0; lr_in[7] <= 8'd0;

        lr_in_valid <= 1'b0; lr_seq_num <= 0;

        conv_input_offset <= 32'd0;
        conv_en <= 1'b0; conv_clr <= 1'b0;
        conv_in_buf[0] <= 8'd0; conv_in_buf[1] <= 8'd0; conv_in_buf[2] <= 8'd0; conv_in_buf[3] <= 8'd0;
        conv_in_buf[4] <= 8'd0; conv_in_buf[5] <= 8'd0; conv_in_buf[6] <= 8'd0; conv_in_buf[7] <= 8'd0;
        
        conv_weights[0] <= 8'd0; conv_weights[1] <= 8'd0; conv_weights[2] <= 8'd0; conv_weights[3] <= 8'd0;
        conv_weights[4] <= 8'd0; conv_weights[5] <= 8'd0; conv_weights[6] <= 8'd0; conv_weights[7] <= 8'd0;
    end else begin
        lr_in_valid <= 0;
        conv_en     <= 0;
        conv_clr    <= 0;
        if (cmd_valid) begin
            case (funct7)
                0: begin
                    lr_seq_num                    <= 0;
                end
                1: begin
                    lr_seq_num                    <= 0;
                    lr_input_offset               <= cmd_payload_inputs_0;
                    lr_output_offset              <= cmd_payload_inputs_1;                
                end
                2: begin
                    lr_output_multiplier_alpha    <= cmd_payload_inputs_0;
                    lr_output_shift_alpha         <= cmd_payload_inputs_1;
                end
                3: begin
                    lr_output_multiplier_identity <= cmd_payload_inputs_0;
                    lr_output_shift_identity      <= cmd_payload_inputs_1;
                end
                4: begin
                    lr_in_valid <= 1;
                    {lr_in[3], lr_in[2], lr_in[1], lr_in[0]} <= cmd_payload_inputs_0;
                    {lr_in[7], lr_in[6], lr_in[5], lr_in[4]} <= cmd_payload_inputs_1;
                end

                10: begin // SET_INPUT_OFFSET
                    conv_input_offset <= cmd_payload_inputs_0;
                end
                11: begin // SET_INPUTS
                    {conv_in_buf[3], conv_in_buf[2], conv_in_buf[1], conv_in_buf[0]} <= cmd_payload_inputs_0;
                    {conv_in_buf[7], conv_in_buf[6], conv_in_buf[5], conv_in_buf[4]} <= cmd_payload_inputs_1;
                end
                12: begin // RUN_WEIGHTS
                    conv_en <= 1'b1;
                    {conv_weights[3], conv_weights[2], conv_weights[1], conv_weights[0]} <= cmd_payload_inputs_0;
                    {conv_weights[7], conv_weights[6], conv_weights[5], conv_weights[4]} <= cmd_payload_inputs_1;
                end
                13: begin // GET_ACC
                    // Handled in output block
                end
                14: begin // RESET_ACC
                    conv_clr <= 1'b1;
                end
            endcase
        end

        if (lr_out_valid) begin
            lr_seq_num <= lr_seq_num + 1;
        end
    end
end

// output block
always @(*) begin
    rsp_payload_outputs_0 = 0;
    if (cmd_valid) begin
        case (funct7)
            0: rsp_payload_outputs_0 = cmd_payload_inputs_1[0] ? lr_data_out[63:32] : lr_data_out[31:0];
            13: rsp_payload_outputs_0 = mac_acc[cmd_payload_inputs_0[5:3]][cmd_payload_inputs_0[2:0]];
        endcase
    end 
end







genvar x, oc;
generate
    for (x=0; x<8; x=x+1) begin : gen_x
        for (oc=0; oc<8; oc=oc+1) begin : gen_oc
            mac_unit u_mac (
                .clk(clk),
                .rst(reset),
                .clr(conv_clr),
                .en(conv_en),
                .weight(conv_weights[oc]),
                .act_in(conv_in_buf[x]),
                .act_off(conv_input_offset),
                .acc(mac_acc[x][oc])
            );
        end
    end
endgenerate















genvar i;
generate
    for (i=0; i<8; i=i+1) begin : gen_leaky_relu
        leaky_relu LReLU (
            .clk(clk),
            .rst(reset),

            .in_valid(lr_in_valid),
            .input_val(lr_in[i]),

            .input_offset(lr_input_offset),
            .output_offset(lr_output_offset),
            .output_multiplier_alpha(lr_output_multiplier_alpha),
            .output_shift_alpha(lr_output_shift_alpha),
            .output_multiplier_identity(lr_output_multiplier_identity),
            .output_shift_identity(lr_output_shift_identity),

            .out_valid(lr_out_valid_arr[i]),
            .output_val(lr_out[i])
        );
    end
endgenerate

// global_buffer_bram #( .ADDR_BITS(8), .DATA_BITS(64) ) 
global_buffer_bram #( .ADDR_BITS(5), .DATA_BITS(64) ) 
gbuff_A (
    .clk     (clk       ),
    .rst_n   (1'b1      ),
    .ram_en  (1'b1      ),
    .wr_en   (lr_out_valid),
    .index   ((cmd_valid && funct7 == 0) ? cmd_payload_inputs_1[31:1] : lr_seq_num),
    .data_in ({lr_out[7], lr_out[6], lr_out[5], lr_out[4], lr_out[3], lr_out[2], lr_out[1], lr_out[0]}),
    .data_out(lr_data_out)
);












endmodule
