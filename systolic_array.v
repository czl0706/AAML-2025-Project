`include "PE.v"

module systolic_array (
    input          clk,
    input          rst_n,
    input  [ 31:0] offset,
    input  [  7:0] in_u0,
    input  [  7:0] in_u1,
    input  [  7:0] in_u2,
    input  [  7:0] in_u3,
    input  [  7:0] in_l0,
    input  [  7:0] in_l1,
    input  [  7:0] in_l2,
    input  [  7:0] in_l3,
    output [127:0] out_r0,
    output [127:0] out_r1,
    output [127:0] out_r2,
    output [127:0] out_r3
);

wire [31:0] PE_out [0:3][0:3];
wire [ 7:0] PE_h   [0:3][0:3];
wire [ 7:0] PE_v   [0:3][0:3];

assign out_r0 = {PE_out[0][0], PE_out[0][1], PE_out[0][2], PE_out[0][3]};
assign out_r1 = {PE_out[1][0], PE_out[1][1], PE_out[1][2], PE_out[1][3]};
assign out_r2 = {PE_out[2][0], PE_out[2][1], PE_out[2][2], PE_out[2][3]};
assign out_r3 = {PE_out[3][0], PE_out[3][1], PE_out[3][2], PE_out[3][3]};

PE c00 ( .clk(clk), .rst_n(rst_n), .U(in_u0     ), .L(in_l0     ), .D(PE_v[0][0]), .R(PE_h[0][0]), .out(PE_out[0][0]), .off(offset) );
PE c01 ( .clk(clk), .rst_n(rst_n), .U(in_u1     ), .L(PE_h[0][0]), .D(PE_v[0][1]), .R(PE_h[0][1]), .out(PE_out[0][1]), .off(offset) );
PE c02 ( .clk(clk), .rst_n(rst_n), .U(in_u2     ), .L(PE_h[0][1]), .D(PE_v[0][2]), .R(PE_h[0][2]), .out(PE_out[0][2]), .off(offset) );
PE c03 ( .clk(clk), .rst_n(rst_n), .U(in_u3     ), .L(PE_h[0][2]), .D(PE_v[0][3]), .R(PE_h[0][3]), .out(PE_out[0][3]), .off(offset) );

PE c10 ( .clk(clk), .rst_n(rst_n), .U(PE_v[0][0]), .L(in_l1     ), .D(PE_v[1][0]), .R(PE_h[1][0]), .out(PE_out[1][0]), .off(offset) );
PE c11 ( .clk(clk), .rst_n(rst_n), .U(PE_v[0][1]), .L(PE_h[1][0]), .D(PE_v[1][1]), .R(PE_h[1][1]), .out(PE_out[1][1]), .off(offset) );
PE c12 ( .clk(clk), .rst_n(rst_n), .U(PE_v[0][2]), .L(PE_h[1][1]), .D(PE_v[1][2]), .R(PE_h[1][2]), .out(PE_out[1][2]), .off(offset) );
PE c13 ( .clk(clk), .rst_n(rst_n), .U(PE_v[0][3]), .L(PE_h[1][2]), .D(PE_v[1][3]), .R(PE_h[1][3]), .out(PE_out[1][3]), .off(offset) );

PE c20 ( .clk(clk), .rst_n(rst_n), .U(PE_v[1][0]), .L(in_l2     ), .D(PE_v[2][0]), .R(PE_h[2][0]), .out(PE_out[2][0]), .off(offset) );
PE c21 ( .clk(clk), .rst_n(rst_n), .U(PE_v[1][1]), .L(PE_h[2][0]), .D(PE_v[2][1]), .R(PE_h[2][1]), .out(PE_out[2][1]), .off(offset) );
PE c22 ( .clk(clk), .rst_n(rst_n), .U(PE_v[1][2]), .L(PE_h[2][1]), .D(PE_v[2][2]), .R(PE_h[2][2]), .out(PE_out[2][2]), .off(offset) );
PE c23 ( .clk(clk), .rst_n(rst_n), .U(PE_v[1][3]), .L(PE_h[2][2]), .D(PE_v[2][3]), .R(PE_h[2][3]), .out(PE_out[2][3]), .off(offset) );

PE c30 ( .clk(clk), .rst_n(rst_n), .U(PE_v[2][0]), .L(in_l3     ), .D(PE_v[3][0]), .R(PE_h[3][0]), .out(PE_out[3][0]), .off(offset) );
PE c31 ( .clk(clk), .rst_n(rst_n), .U(PE_v[2][1]), .L(PE_h[3][0]), .D(PE_v[3][1]), .R(PE_h[3][1]), .out(PE_out[3][1]), .off(offset) );
PE c32 ( .clk(clk), .rst_n(rst_n), .U(PE_v[2][2]), .L(PE_h[3][1]), .D(PE_v[3][2]), .R(PE_h[3][2]), .out(PE_out[3][2]), .off(offset) );
PE c33 ( .clk(clk), .rst_n(rst_n), .U(PE_v[2][3]), .L(PE_h[3][2]), .D(PE_v[3][3]), .R(PE_h[3][3]), .out(PE_out[3][3]), .off(offset) );

endmodule