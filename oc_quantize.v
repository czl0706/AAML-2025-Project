module oc_quantize (
    input clk,
    input rst,

    input in_valid,
    input signed [31:0] input_val,

    // Should be constants
    input signed [31:0] bias,
    input signed [31:0] output_offset,
    input signed [31:0] output_multiplier,
    input signed [31:0] output_shift,

    output reg out_valid,
    output reg signed [7:0] output_val
);

// Stage 1: Add bias
reg signed [31:0] acc;
always @(*) begin
    acc = input_val + bias;
end

reg signed [31:0] acc_q;
reg valid_q1;

always @(posedge clk) begin
    if (rst) begin
        valid_q1 <= 1'b0;
        acc_q <= 32'd0;
    end else begin
        valid_q1 <= in_valid;
        acc_q <= acc;
    end
end

// Stage 2: MultiplyByQuantizedMultiplier
reg signed [63:0] prod;
reg signed [31:0] total_shift;
reg signed [63:0] round;
reg signed [63:0] shifted_prod;

always @(*) begin
    prod = $signed(acc_q) * $signed(output_multiplier);
    total_shift = 31 - output_shift;
    round = 64'd1 << (total_shift - 1);

    if (total_shift > 0) begin
        shifted_prod = (prod + round) >>> total_shift;
    end else begin
        shifted_prod = prod << (-total_shift);
    end
end

reg signed [63:0] shifted_prod_q;
reg valid_q2;

always @(posedge clk) begin
    if (rst) begin
        valid_q2 <= 1'b0;
        shifted_prod_q <= 64'd0;
    end else begin
        valid_q2 <= valid_q1;
        shifted_prod_q <= shifted_prod;
    end
end

// Stage 3: Add offset and clamp
reg signed [31:0] unclamped_output;
reg signed [7:0] out;

always @(*) begin
    unclamped_output = $signed(shifted_prod_q[31:0]) + output_offset;
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
    end else begin
        out_valid <= valid_q2;
        output_val <= out;
    end
end

endmodule
