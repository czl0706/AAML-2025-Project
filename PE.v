module PE (
    input             clk,
    input             rst_n,
    input      signed [31:0] off,
    input      signed [ 7:0] U,
    input      signed [ 7:0] L,
    output reg signed [ 7:0] D,
    output reg signed [ 7:0] R,
    output reg signed [31:0] out
);

always @(posedge clk) begin
    if (!rst_n) begin
        D   <= 0;
        R   <= 0;
        out <= 0;
    end else begin
        D   <= U;
        R   <= L;
        out <= out + U * (L + off);
        // out <= out + U * L;
    end 
end

endmodule