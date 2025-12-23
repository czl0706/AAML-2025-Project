`include "systolic_array.v"

module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    busy,

    A_offset,
    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);


input clk;
input rst_n;
input            in_valid;
input [9:0]      K;
input [9:0]      M;
input [9:0]      N;
output  reg      busy;

input  [31:0]     A_offset;

output           A_wr_en;
output [13:0]    A_index;
output [63:0]    A_data_in;
input  [63:0]    A_data_out;

output           B_wr_en;
output [13:0]    B_index;
output [63:0]    B_data_in;
input  [63:0]    B_data_out;

output           C_wr_en;
output [13:0]    C_index;
output [255:0]   C_data_in;
input  [255:0]   C_data_out;

assign A_wr_en = 1'b0;
assign B_wr_en = 1'b0;
assign A_data_in = 64'd0;
assign B_data_in = 64'd0;

parameter [2:0] IDLE = 3'd0, TILE = 3'd1, WAIT = 3'd2; // , DONE = 3'd3;
localparam integer ARRAY_SIZE = 8;

reg [2:0] cs;
reg [7:0] Md8_q, Nd8_q;
reg [2:0] Mm8_q;
reg [9:0] K_q;
reg       pe_rstn;

always @(*) busy = (cs != IDLE);

reg [13:0] A_idx, B_idx, C_idx;
reg [13:0] idx_mK;
reg [13:0] idx_i, idx_j; 

reg [13:0] B_idx_prev;

reg [ 3:0] wait_cnt;

assign A_index = A_idx;
assign B_index = B_idx;
assign C_index = C_idx;

wire [13:0] idx_mK_p1 = idx_mK + 1;
wire [13:0] A_idx_p1 = A_idx + 1;
wire [13:0] B_idx_p1 = B_idx + 1;
wire [13:0] C_idx_p1 = C_idx + 1;
wire [13:0] idx_i_p1 = idx_i + 1;
wire [13:0] idx_j_p1 = idx_j + 1;


wire [255:0] out_r0, out_r1, out_r2, out_r3, out_r4, out_r5, out_r6, out_r7;

assign C_wr_en   = (wait_cnt >= 8);
assign C_data_in = (wait_cnt == 8)  ? out_r0 :
                   (wait_cnt == 9)  ? out_r1 :
                   (wait_cnt == 10) ? out_r2 :
                   (wait_cnt == 11) ? out_r3 :
                   (wait_cnt == 12) ? out_r4 :
                   (wait_cnt == 13) ? out_r5 :
                   (wait_cnt == 14) ? out_r6 :
                   (wait_cnt == 15) ? out_r7 : 256'd0;

//* Implement your design here
always @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
        cs <= IDLE;

        K_q     <= 0;
        Md8_q   <= 0;
        Nd8_q   <= 0;
        Mm8_q   <= 0;

        pe_rstn <= 1'b0;

        A_idx <= 16'd0; B_idx <= 16'd0; C_idx <= 16'd0;
        idx_i <= 16'd0; idx_j <= 16'd0;
        idx_mK <= 16'd0;

        B_idx_prev <= 16'd0;

        wait_cnt <= 4'd0;
    end else begin
        case (cs)
            IDLE: begin
                A_idx <= 16'd0; B_idx <= 16'd0; C_idx <= 16'd0;
                idx_i <= 16'd0; idx_j <= 16'd0;
                idx_mK <= 16'd0;
                B_idx_prev <= 16'd0;

                if (in_valid) begin
                    // busy  <= 1'b1;
                    K_q   <= K;

                    Md8_q <= M[9:3] + |(M[2:0]);
                    Mm8_q <= M[2:0];
                    Nd8_q <= N[9:3] + |(N[2:0]);
                    
                    pe_rstn <= 1'b1;
                    cs <= TILE;
                end
            end
            TILE: begin
                pe_rstn<= 1'b1;
                A_idx  <= A_idx_p1;
                B_idx  <= B_idx_p1;
                idx_mK <= idx_mK_p1;

                if (idx_mK_p1 == K_q) begin
                    idx_mK   <= 0;
                    cs  <= WAIT;
                end else begin
                    idx_mK <= idx_mK_p1;
                end
            end
            WAIT: begin
                wait_cnt <= wait_cnt + 1;
                if (wait_cnt >= ARRAY_SIZE) begin
                    C_idx <= C_idx_p1;
                end

                if (wait_cnt == (ARRAY_SIZE * 2 - 2)) begin
                    pe_rstn  <= 1'b0;
                end
    
                if (wait_cnt == (ARRAY_SIZE * 2 - 1)) begin
                    wait_cnt <= 0;
                    pe_rstn  <= 1'b1;
                    
                    if (idx_j_p1 == Nd8_q) begin
                        if (idx_i_p1 == Md8_q) begin 
                            cs <= IDLE;
                        end else begin
                            B_idx <= B_idx_prev;
                            idx_i <= idx_i_p1;
                            cs <= TILE;
                        end
                    end else begin
                        if (idx_i_p1 == Md8_q) begin
                            A_idx <= 0;
                            B_idx_prev <= B_idx;
                            
                            idx_i <= 0;
                            idx_j <= idx_j_p1;
                            cs <= TILE;
                        end else begin
                            B_idx <= B_idx_prev;
                            idx_i <= idx_i_p1;
                            cs <= TILE;
                        end
                    end
                end
            end
        endcase
    end
end

// Delay pipeline
// reg [7:0] l0_q0, 
//           l1_q0, l1_q1,
//           l2_q0, l2_q1, l2_q2,
//           l3_q0, l3_q1, l3_q2, l3_q3;

// reg [7:0] u0_q0,
//           u1_q0, u1_q1,
//           u2_q0, u2_q1, u2_q2,
//           u3_q0, u3_q1, u3_q2, u3_q3;

// Delay pipeline for 8x8 array
wire signed [7:0] in_a [0:7];
assign in_a[0] = A_data_out[63:56];
assign in_a[1] = A_data_out[55:48];
assign in_a[2] = A_data_out[47:40];
assign in_a[3] = A_data_out[39:32];
assign in_a[4] = A_data_out[31:24];
assign in_a[5] = A_data_out[23:16];
assign in_a[6] = A_data_out[15: 8];
assign in_a[7] = A_data_out[ 7: 0];

wire signed [7:0] in_b [0:7];
assign in_b[0] = B_data_out[63:56];
assign in_b[1] = B_data_out[55:48];
assign in_b[2] = B_data_out[47:40];
assign in_b[3] = B_data_out[39:32];
assign in_b[4] = B_data_out[31:24];
assign in_b[5] = B_data_out[23:16];
assign in_b[6] = B_data_out[15: 8];
assign in_b[7] = B_data_out[ 7: 0];

reg signed [7:0] l_delay [0:7][0:7];
reg signed [7:0] u_delay [0:7][0:7];
integer rr, dd;

always @(posedge clk) begin
    if (!pe_rstn) begin
        for (rr = 0; rr < 8; rr = rr + 1) begin
            for (dd = 0; dd < 8; dd = dd + 1) begin
                l_delay[rr][dd] <= 0;
                u_delay[rr][dd] <= 0;
            end
        end
    end else begin
        for (rr = 0; rr < 8; rr = rr + 1) begin
            for (dd = 7; dd > 0; dd = dd - 1) begin
                if (dd <= rr) l_delay[rr][dd] <= l_delay[rr][dd-1];
            end
            l_delay[rr][0] <= cs == TILE ? in_a[rr] : 0;
        end

        for (rr = 0; rr < 8; rr = rr + 1) begin
            for (dd = 7; dd > 0; dd = dd - 1) begin
                if (dd <= rr) u_delay[rr][dd] <= u_delay[rr][dd-1];
            end
            u_delay[rr][0] <= cs == TILE ? in_b[rr] : 0;
        end
    end
end

wire signed [7:0] l_feed [0:7];
wire signed [7:0] u_feed [0:7];

genvar gi;
generate
    for (gi = 0; gi < 8; gi = gi + 1) begin : feed_gen
        assign l_feed[gi] = l_delay[gi][gi];
        assign u_feed[gi] = u_delay[gi][gi];
    end
endgenerate

systolic_array S_u0 (
    .clk(clk), 
    .rst_n(pe_rstn), 
    .offset(A_offset),
    .in_u0(u_feed[0]), 
    .in_u1(u_feed[1]), 
    .in_u2(u_feed[2]), 
    .in_u3(u_feed[3]),
    .in_u4(u_feed[4]),
    .in_u5(u_feed[5]),
    .in_u6(u_feed[6]),
    .in_u7(u_feed[7]),
    .in_l0(l_feed[0]), 
    .in_l1(l_feed[1]), 
    .in_l2(l_feed[2]), 
    .in_l3(l_feed[3]),
    .in_l4(l_feed[4]),
    .in_l5(l_feed[5]),
    .in_l6(l_feed[6]),
    .in_l7(l_feed[7]),
    .out_r0(out_r0), 
    .out_r1(out_r1), 
    .out_r2(out_r2), 
    .out_r3(out_r3),
    .out_r4(out_r4),
    .out_r5(out_r5),
    .out_r6(out_r6),
    .out_r7(out_r7)
);

endmodule
