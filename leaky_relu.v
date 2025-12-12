module leaky_relu (
    input clk,
    input rst,

    input in_valid,
    input signed [7:0] input_val,

    // Should be constants
    input signed [31:0] input_offset,
    input signed [31:0] output_offset,
    input signed [31:0] output_multiplier_alpha,
    input signed [31:0] output_shift_alpha,
    input signed [31:0] output_multiplier_identity,
    input signed [31:0] output_shift_identity,

    output reg out_valid,
    output reg signed [7:0] output_val
);

// 1. Calculate input_value and select parameters based on input sign
reg signed [31:0] input_value;
reg signed [31:0] multiplier;
reg signed [31:0] shift;

always @(*) begin
    input_value = {{24{input_val[7]}}, input_val} - input_offset;
    if (input_value >= 0) begin
        multiplier = output_multiplier_identity;
        shift = output_shift_identity;
    end else begin
        multiplier = output_multiplier_alpha;
        shift = output_shift_alpha;
    end
end

reg signed [31:0] input_value_q;
reg signed [31:0] multiplier_q;
reg signed [31:0] shift_q;
reg valid_q1;

always @(posedge clk) begin
    if (rst) begin  
        valid_q1      <= 1'b0;
        input_value_q <= 32'd0;
        multiplier_q  <= 32'd0;
        shift_q       <= 32'd0;
    end else begin
        valid_q1      <= in_valid;
        input_value_q <= input_value;
        multiplier_q  <= multiplier;
        shift_q       <= shift;
    end
end

// 2. CustomMultiplyByQuantizedMultiplier Logic
// int total_shift = 31 - shift;
// int64_t prod = (int64_t)x * (int64_t)m;

reg signed [63:0] prod;
reg signed [31:0] total_shift;
reg signed [63:0] round;
reg signed [63:0] shifted_prod;

always @(*) begin
    prod = $signed(input_value_q) * $signed(multiplier_q);
    total_shift = 31 - shift_q;
    round = 64'd1 << (total_shift - 1);

    if (total_shift > 0) begin    
        shifted_prod = (prod + round) >>> total_shift;
    end else begin
        // total_shift <= 0 -> left shift
        shifted_prod = prod << (-total_shift);
    end
end

reg signed [63:0] shifted_prod_q;
reg valid_q2;
reg valid_q3;

always @(posedge clk) begin
    if (rst) begin
        valid_q2       <= 1'b0;
        valid_q3       <= 1'b0;
        shifted_prod_q <= 64'd0;
    end else begin
        valid_q2       <= valid_q1;
        valid_q3       <= valid_q2;
        shifted_prod_q <= shifted_prod;
    end
end

// 3. Add output_offset and clamp to [-128, 127]

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