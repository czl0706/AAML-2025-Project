`include "PE.v"

module systolic_array (
    input          clk,
    input          rst_n,
    input  [ 31:0] offset,
    input  [  7:0] in_u0,
    input  [  7:0] in_u1,
    input  [  7:0] in_u2,
    input  [  7:0] in_u3,
    input  [  7:0] in_u4,
    input  [  7:0] in_u5,
    input  [  7:0] in_u6,
    input  [  7:0] in_u7,
    input  [  7:0] in_l0,
    input  [  7:0] in_l1,
    input  [  7:0] in_l2,
    input  [  7:0] in_l3,
    input  [  7:0] in_l4,
    input  [  7:0] in_l5,
    input  [  7:0] in_l6,
    input  [  7:0] in_l7,
    output [255:0] out_r0,
    output [255:0] out_r1,
    output [255:0] out_r2,
    output [255:0] out_r3,
    output [255:0] out_r4,
    output [255:0] out_r5,
    output [255:0] out_r6,
    output [255:0] out_r7
);

wire [31:0] PE_out [0:7][0:7];
wire [ 7:0] PE_h   [0:7][0:7];
wire [ 7:0] PE_v   [0:7][0:7];

genvar r, c;
generate
    for (r = 0; r < 8; r = r + 1) begin : row_gen
        for (c = 0; c < 8; c = c + 1) begin : col_gen
            wire [7:0] u_in = (r == 0) ? (c == 0 ? in_u0 :
                                           c == 1 ? in_u1 :
                                           c == 2 ? in_u2 :
                                           c == 3 ? in_u3 :
                                           c == 4 ? in_u4 :
                                           c == 5 ? in_u5 :
                                           c == 6 ? in_u6 : in_u7)
                                       : PE_v[r-1][c];
            wire [7:0] l_in = (c == 0) ? (r == 0 ? in_l0 :
                                           r == 1 ? in_l1 :
                                           r == 2 ? in_l2 :
                                           r == 3 ? in_l3 :
                                           r == 4 ? in_l4 :
                                           r == 5 ? in_l5 :
                                           r == 6 ? in_l6 : in_l7)
                                       : PE_h[r][c-1];
            PE cxx (
                .clk(clk),
                .rst_n(rst_n),
                .U(u_in),
                .L(l_in),
                .D(PE_v[r][c]),
                .R(PE_h[r][c]),
                .out(PE_out[r][c]),
                .off(offset)
            );
        end
    end
endgenerate

wire [255:0] out_rows [0:7];

generate
    for (r = 0; r < 8; r = r + 1) begin : pack_rows
        assign out_rows[r] = {PE_out[r][0], PE_out[r][1], PE_out[r][2], PE_out[r][3],
                              PE_out[r][4], PE_out[r][5], PE_out[r][6], PE_out[r][7]};
    end
endgenerate

assign out_r0 = out_rows[0];
assign out_r1 = out_rows[1];
assign out_r2 = out_rows[2];
assign out_r3 = out_rows[3];
assign out_r4 = out_rows[4];
assign out_r5 = out_rows[5];
assign out_r6 = out_rows[6];
assign out_r7 = out_rows[7];

endmodule
