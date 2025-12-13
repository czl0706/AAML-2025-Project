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
output [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output [15:0]    C_index;
output [127:0]   C_data_in;
input  [127:0]   C_data_out;

assign A_wr_en = 1'b0;
assign B_wr_en = 1'b0;
assign A_data_in = 32'd0;
assign B_data_in = 32'd0;

parameter [2:0] IDLE = 3'd0, TILE = 3'd1, WAIT = 3'd2; // , DONE = 3'd3;

reg [2:0] cs;
reg [7:0] Md4_q, Nd4_q;
reg [1:0] Mm4_q;
reg [9:0] K_q;
reg       pe_rstn;

always @(*) busy = (cs != IDLE);

reg [15:0] A_idx, B_idx, C_idx;
reg [15:0] idx_mK;
reg [15:0] idx_i, idx_j; 

reg [15:0] B_idx_prev;

reg [ 2:0] wait_cnt;

assign A_index = A_idx;
assign B_index = B_idx;
assign C_index = C_idx;

wire [15:0] idx_mK_p1 = idx_mK + 1;
wire [15:0] A_idx_p1 = A_idx + 1;
wire [15:0] B_idx_p1 = B_idx + 1;
wire [15:0] C_idx_p1 = C_idx + 1;
wire [15:0] idx_i_p1 = idx_i + 1;
wire [15:0] idx_j_p1 = idx_j + 1;


wire [127:0] out_r0, out_r1, out_r2, out_r3;

assign C_wr_en   = (wait_cnt >= 4);
assign C_data_in = (wait_cnt == 4) ? out_r0 :
                   (wait_cnt == 5) ? out_r1 :
                   (wait_cnt == 6) ? out_r2 :
                   (wait_cnt == 7) ? out_r3 : 128'd0;

//* Implement your design here
always @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
        cs <= IDLE;

        K_q     <= 0;
        Md4_q   <= 0;
        Nd4_q   <= 0;
        Mm4_q   <= 0;

        pe_rstn <= 1'b0;

        A_idx <= 16'd0; B_idx <= 16'd0; C_idx <= 16'd0;
        idx_i <= 16'd0; idx_j <= 16'd0;
        idx_mK <= 16'd0;

        B_idx_prev <= 16'd0;

        wait_cnt <= 2'd0;
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

                    // Md4_q <= M[7:2] + |(M[1:0]);
                    Md4_q <= M[9:2] + |(M[1:0]);
                    Mm4_q <= M[1:0];
                    Nd4_q <= N[9:2] + |(N[1:0]);
                    
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
                if (wait_cnt >= 4) begin
                    C_idx <= C_idx_p1;
                end

                if (wait_cnt == 6 || (Mm4_q != 0 && wait_cnt == Mm4_q + 2 && idx_i_p1 == Md4_q)) begin
                    pe_rstn  <= 1'b0;
                end
    
                if (wait_cnt == 7 || (Mm4_q != 0 && wait_cnt == Mm4_q + 3 && idx_i_p1 == Md4_q)) begin
                    wait_cnt <= 0;
                    pe_rstn  <= 1'b1;
                    
                    if (idx_j_p1 == Nd4_q) begin
                        if (idx_i_p1 == Md4_q) begin 
                            cs <= IDLE;
                        end else begin
                            B_idx <= B_idx_prev;
                            idx_i <= idx_i_p1;
                            cs <= TILE;
                        end
                    end else begin
                        if (idx_i_p1 == Md4_q) begin
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

reg [7:0] l0_q0, 
          l1_q0, l1_q1,
          l2_q0, l2_q1, l2_q2,
          l3_q0, l3_q1, l3_q2, l3_q3;

reg [7:0] u0_q0,
          u1_q0, u1_q1,
          u2_q0, u2_q1, u2_q2,
          u3_q0, u3_q1, u3_q2, u3_q3;

wire signed [7:0] in_a0 = A_data_out[31:24];
wire signed [7:0] in_a1 = A_data_out[23:16];
wire signed [7:0] in_a2 = A_data_out[15: 8];
wire signed [7:0] in_a3 = A_data_out[ 7: 0];

always @(posedge clk) begin
    if (!pe_rstn) begin
        l0_q0 <= 0;
        l1_q0 <= 0; l1_q1 <= 0;
        l2_q0 <= 0; l2_q1 <= 0; l2_q2 <= 0;
        l3_q0 <= 0; l3_q1 <= 0; l3_q2 <= 0; l3_q3 <= 0;

        u0_q0 <= 0;
        u1_q0 <= 0; u1_q1 <= 0;
        u2_q0 <= 0; u2_q1 <= 0; u2_q2 <= 0;
        u3_q0 <= 0; u3_q1 <= 0; u3_q2 <= 0; u3_q3 <= 0;
    end else begin
        // Horizontal delay line
        l3_q3 <= l3_q2;
        l3_q2 <= l3_q1;
        l3_q1 <= l3_q0;
        // l3_q0 <= cs == TILE ? A_data_out[ 7: 0] : 0;
        l3_q0 <= cs == TILE ? in_a3 : 0;

        l2_q2 <= l2_q1;
        l2_q1 <= l2_q0;
        l2_q0 <= cs == TILE ? in_a2 : 0;

        l1_q1 <= l1_q0;
        l1_q0 <= cs == TILE ? in_a1 : 0;

        l0_q0 <= cs == TILE ? in_a0 : 0;

        // Vertical delay line
        u3_q3 <= u3_q2;
        u3_q2 <= u3_q1;
        u3_q1 <= u3_q0;
        u3_q0 <= cs == TILE ? B_data_out[ 7: 0] : 0;

        u2_q2 <= u2_q1;
        u2_q1 <= u2_q0;
        u2_q0 <= cs == TILE ? B_data_out[15: 8] : 0;

        u1_q1 <= u1_q0;
        u1_q0 <= cs == TILE ? B_data_out[23:16] : 0;

        u0_q0 <= cs == TILE ? B_data_out[31:24] : 0;
    end
end

systolic_array S_u0 (
    .clk(clk), 
    .rst_n(pe_rstn), 
    .offset(A_offset),
    .in_u0(u0_q0), 
    .in_u1(u1_q1), 
    .in_u2(u2_q2), 
    .in_u3(u3_q3), 
    .in_l0(l0_q0), 
    .in_l1(l1_q1), 
    .in_l2(l2_q2), 
    .in_l3(l3_q3), 
    .out_r0(out_r0), 
    .out_r1(out_r1), 
    .out_r2(out_r2), 
    .out_r3(out_r3)
);

endmodule