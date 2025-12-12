module mac_unit (
    input clk,
    input rst,
    input clr,
    input en,
    input signed [7:0] weight,
    input signed [7:0] act_in,
    input signed [31:0] act_off,
    output reg signed [31:0] acc
);

wire signed [31:0] act_shift;
assign act_shift = act_in + act_off;

always @(posedge clk) begin
    if (rst) begin
        acc <= 32'd0;
    end else begin
        if (clr) begin
            acc <= 32'd0;
        end else if (en) begin
            acc <= acc + weight * act_shift;
        end
    end
end

endmodule