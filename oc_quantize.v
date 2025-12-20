module oc_quantize (
    input clk,
    input rst,

    input in_valid,
    input signed [31:0] input_val,
    input [7:0] in_index,

    input signed [31:0] bias,
    input signed [31:0] output_offset,
    input signed [31:0] output_multiplier,
    input signed [31:0] output_shift,

    output reg out_valid,
    output reg signed [7:0] output_val,
    output reg [7:0] out_index
);

// Stage 1: Input Register (Align with BRAM latency)
reg signed [31:0] input_val_q0;
reg signed [31:0] bias_q0;
reg signed [31:0] output_offset_q0;
reg signed [31:0] output_multiplier_q0;
reg signed [31:0] output_shift_q0;
reg in_valid_q0;
reg [7:0] index_q0;

always @(posedge clk) begin
    if (rst) begin
        in_valid_q0 <= 1'b0;
        index_q0 <= 8'd0;
        input_val_q0 <= 32'd0;
        bias_q0 <= 32'd0;
        output_offset_q0 <= 32'd0;
        output_multiplier_q0 <= 32'd0;
        output_shift_q0 <= 32'd0;
    end else begin
        in_valid_q0 <= in_valid;
        index_q0 <= in_index;
        input_val_q0 <= input_val;
        bias_q0 <= bias;
        output_offset_q0 <= output_offset;
        output_multiplier_q0 <= output_multiplier;
        output_shift_q0 <= output_shift;
    end
end

// Stage 2: Add bias
reg signed [63:0] prod_q1;
reg valid_q1;
reg [7:0] index_q1;
reg signed [6:0] total_shift_q1;
reg signed [31:0] output_offset_q1;

always @(posedge clk) begin
    if (rst) begin
        valid_q1 <= 1'b0;
        prod_q1 <= 64'd0;
        index_q1 <= 8'd0;
        total_shift_q1 <= 7'd0;
        output_offset_q1 <= 32'd0;
    end else begin
        valid_q1 <= in_valid_q0;
        prod_q1 <= $signed(input_val_q0 + bias_q0) * $signed(output_multiplier_q0);
        index_q1 <= index_q0;
        total_shift_q1 <= 7'd31 - output_shift_q0;
        output_offset_q1 <= output_offset_q0;
    end
end

// Stage 3: MultiplyByQuantizedMultiplier
reg signed [63:0] round;
reg signed [63:0] shifted_prod;

always @(*) begin
    round = 64'd1 << (total_shift_q1 - 1);

    if (total_shift_q1 > 0) begin
        shifted_prod = (prod_q1 + round) >>> total_shift_q1;
    end else begin
        shifted_prod = prod_q1 << (-total_shift_q1);
    end
end

reg signed [63:0] shifted_prod_q2;
reg valid_q2;
reg [7:0] index_q2;
reg signed [31:0] output_offset_q2;

always @(posedge clk) begin
    if (rst) begin
        valid_q2 <= 1'b0;
        index_q2 <= 8'd0;
        shifted_prod_q2 <= 64'd0;
        output_offset_q2 <= 32'd0;
    end else begin
        valid_q2 <= valid_q1;
        index_q2 <= index_q1;
        shifted_prod_q2 <= shifted_prod;
        output_offset_q2 <= output_offset_q1;
    end
end

// Stage 4: Add offset and clamp
reg signed [31:0] unclamped_output;
reg signed [7:0] out;

always @(*) begin
    unclamped_output = $signed(shifted_prod_q2[31:0]) + output_offset_q2;
    if (unclamped_output > 127) begin
        out = 8'd127;
    end else if (unclamped_output < -128) begin
        out = -8'd128;
    end else begin
        out = unclamped_output[7:0];
    end
end

always @(posedge clk) begin
    if (rst) begin
        out_valid <= 1'b0;
        output_val <= 8'd0;
        out_index <= 8'd0;
    end else begin
        out_valid <= valid_q2;
        output_val <= out;
        out_index <= index_q2;
    end
end

endmodule
