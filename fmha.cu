__global__ //__launch_bounds__(128, 10)
void standardAttentionFusedL64D64F16Kernel(half* in_q, half* in_k, half* in_v, half* attn_out){
    //b = 128 = 4*32, B->64*64
    //G = batch_size * head_num
    //感觉太重，可能不好
    //有几种路线，短序列可能直接载入全算把比较好，长序列可能迭代方式比较好，比较基准为byteTrans，可以考虑屏蔽mask和bias，或者暂时加上两者
    int warp_id = threadIdx.x / 32;
    int warp_tx = threadIdx.x % 32;
    int group_id = blockIdx.x;
    // __shared__ half sh_soft_mat[64*64];
    __shared__ half sh_q[4*16*16*2]; //soft过程中可以兼职做一些中继
    __shared__ half sh_kv[4*16*16*2];

    half* in_q_p = in_q + group_id * 64 * 64;
    half* in_k_p = in_k + group_id * 64 * 64;
    half* in_v_p = in_v + group_id * 64 * 64;

    half reg_res[32];
    half reg_mat_a[8];
    half reg_mat_b[8];

    //轮次计算soft_mat
    uint32_t sh_q_add = __cvta_generic_to_shared(&sh_q);
    uint32_t sh_kv_add = __cvta_generic_to_shared(&sh_kv);
    // uint32_t sh_soft_mat_add = __cvta_generic_to_shared(&sh_soft_mat);
    uint32_t* mat_a = reinterpret_cast<uint32_t*>(reg_mat_a);
    uint32_t* mat_b = reinterpret_cast<uint32_t*>(reg_mat_b);
    uint32_t* mat_res = reinterpret_cast<uint32_t*>(reg_mat_res);
    #pragma unroll(16)
    for (int i = 0; i < 16; i++) reinterpret_cast<float*>(reg_res)[i] = 0.0f;

    asm volatile("cp.async.cg.shared.global"
                "[%0],[%1], 16;"
                :
                : "r"(sh_q_add + warp_id *16*32 + warp_tx*16),
                "l"(&in_q_p[warp_id * 16*64 + warp_tx % 16*64 + warp_tx/16*8]));
    asm volatile("cp.async.cg.shared.global"
                "[%0],[%1], 16;"
                :
                : "r"(sh_kv_add + warp_id *16*32 + warp_tx*16),
                "l"(&in_k_p[warp_id * 16*64 + warp_tx % 16*64 + warp_tx/16*8]));
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    #pragma unroll(3)
    for (int i = 1; i < 4; i++){
        int cal_i = i - 1;
        asm volatile("cp.async.cg.shared.global"
                    "[%0],[%1], 16;"
                    :
                    : "r"(sh_q_add + i%2*64*32 + warp_id *16*32 + warp_tx*16),
                    "l"(&in_q_p[warp_id * 16*64 + warp_tx % 16*64 + i*16 + warp_tx/16*8]));
        asm volatile("cp.async.cg.shared.global"
                    "[%0],[%1], 16;"
                    :
                    : "r"(sh_kv_add + i%2*64*32 + warp_id *16*32 + warp_tx*16),
                    "l"(&in_k_p[warp_id * 16*64 + warp_tx % 16*64 + i*16 + warp_tx/16*8]));
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                    "{%0,%1,%2,%3}, [%4];"
                    : "=r"(mat_a[0]), "=r"(mat_a[1]), "=r"(mat_a[2]), "=r"(mat_a[3])
                    : "r"(sh_q_add + cal_i%2*64*32 + warp_id * 16*32 + warp_tx*16));
        #pragma unroll(4)
        for (int j = 0; j < 4; j++){
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                        "{%0,%1,%2,%3}, [%4];"
                        : "=r"(mat_b[0]), "=r"(mat_b[1]), "=r"(mat_b[2]), "=r"(mat_b[3])
                        : "r"(sh_kv_add + cal_i%2*64*32 + j * 16*32 + warp_tx*16));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                        : "=r"(mat_res[j * 4]), "=r"(mat_res[j * 4 + 1])
                        : "r"(mat_a[0]), "r"(mat_a[1]), "r"(mat_a[2]), "r"(mat_a[3]),
                        "r"(mat_b[0]), "r"(mat_b[2]), "r"(mat_res[j*4]), "r"(mat_res[j*4 + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                        : "=r"(mat_res[j * 4 + 2]), "=r"(mat_res[j * 4 + 3])
                        : "r"(mat_a[0]), "r"(mat_a[1]), "r"(mat_a[2]), "r"(mat_a[3]),
                        "r"(mat_b[1]), "r"(mat_b[3]), "r"(mat_res[j*4 + 2]), "r"(mat_res[j*4 + 3]));
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                "{%0,%1,%2,%3}, [%4];"
                : "=r"(mat_a[0]), "=r"(mat_a[1]), "=r"(mat_a[2]), "=r"(mat_a[3])
                : "r"(sh_q_add + 64*32 + warp_id * 16*32 + warp_tx*16));
    #pragma unroll(4)
    for (int j = 0; j < 4; j++){
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                    "{%0,%1,%2,%3}, [%4];"
                    : "=r"(mat_b[0]), "=r"(mat_b[1]), "=r"(mat_b[2]), "=r"(mat_b[3])
                    : "r"(sh_kv_add + 64*32 + j * 16*32 + warp_tx*16));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                    : "=r"(mat_res[j * 4]), "=r"(mat_res[j * 4 + 1])
                    : "r"(mat_a[0]), "r"(mat_a[1]), "r"(mat_a[2]), "r"(mat_a[3]),
                    "r"(mat_b[0]), "r"(mat_b[2]), "r"(mat_res[j*4]), "r"(mat_res[j*4 + 1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                    : "=r"(mat_res[j * 4 + 2]), "=r"(mat_res[j * 4 + 3])
                    : "r"(mat_a[0]), "r"(mat_a[1]), "r"(mat_a[2]), "r"(mat_a[3]),
                    "r"(mat_b[1]), "r"(mat_b[3]), "r"(mat_res[j*4 + 2]), "r"(mat_res[j*4 + 3]));
    }

    // 进行soft_max
    // 首先考虑一次性完成，摒弃sh，直接reg操作（可以考虑作为迭代式模块内操作）
    float th_max_1 = -1e20f;
    float th_max_2 = -1e20f;
    float th_sum_1 = 0.0f;
    float th_sum_2 = 0.0f;
    #pragma unroll(8)
    for (int i = 0; i < 8; i++){
        th_max_1 = fmaxf(th_max_1, static_cast<float>(reg_res[i*4]));
        th_max_1 = fmaxf(th_max_1, static_cast<float>(reg_res[i*4 + 1]));
        th_max_2 = fmaxf(th_max_2, static_cast<float>(reg_res[i*4 + 2]));
        th_max_2 = fmaxf(th_max_2, static_cast<float>(reg_res[i*4 + 3]));
    }
    th_max_1 = fmaxf(th_max_1, __shfl_xor_sync(0xffffffff, th_max_1, 0x1, 32));
    th_max_1 = fmaxf(th_max_1, __shfl_xor_sync(0xffffffff, th_max_1, 0x2, 32));
    th_max_2 = fmaxf(th_max_2, __shfl_xor_sync(0xffffffff, th_max_2, 0x1, 32));
    th_max_2 = fmaxf(th_max_2, __shfl_xor_sync(0xffffffff, th_max_2, 0x2, 32));
    #pragma unroll(8)
    for (int i = 0; i < 8; i++){
        float tmp;
        tmp = __expf(static_cast<float>(reg_res[i*4]) - th_max_1);
        th_sum_1 += tmp;
        reg_res[i*4] = static_cast<half>(tmp);
        tmp = __expf(static_cast<float>(reg_res[i*4 + 1]) - th_max_1);
        th_sum_1 += tmp;
        reg_res[i*4 + 1] = static_cast<half>(tmp);
        tmp = __expf(static_cast<float>(reg_res[i*4 + 2]) - th_max_2);
        th_sum_2 += tmp;
        reg_res[i*4 + 2] = static_cast<half>(tmp);
        tmp = __expf(static_cast<float>(reg_res[i*4 + 3]) - th_max_2);
        th_sum_2 += tmp;
        reg_res[i*4 + 3] = static_cast<half>(tmp);
    }
    th_sum_1 += __shfl_xor_sync(0xffffffff, th_sum_1, 0x1, 32);
    th_sum_1 += __shfl_xor_sync(0xffffffff, th_sum_1, 0x2, 32);
    th_sum_2 += __shfl_xor_sync(0xffffffff, th_sum_2, 0x1, 32);
    th_sum_2 += __shfl_xor_sync(0xffffffff, th_sum_2, 0x2, 32);
    #pragma unroll(8)
    for (int i = 0; i < 8; i++){
        reg_res[i*4] = static_cast<half>(static_cast<float>(reg_res[i*4]) / th_sum_1);
        reg_res[i*4 + 1] = static_cast<half>(static_cast<float>(reg_res[i*4 + 1]) / th_sum_1);
        reg_res[i*4 + 2] = static_cast<half>(static_cast<float>(reg_res[i*4 + 2]) / th_sum_2);
        reg_res[i*4 + 3] = static_cast<half>(static_cast<float>(reg_res[i*4 + 3]) / th_sum_2);
    }

    // *v
    half reg_final[32];
    mat_res = reinterpret_cast<uint32_t*>(reg_final);
    mat_a = reinterpret_cast<uint32_t*>(reg_res);
    #pragma unroll(16)
    for (int i = 0; i < 16; i++) reinterpret_cast<float*>(reg_final)[i] = 0.0f;

    asm volatile("cp.async.ca.shared.global"
                "[%0], [%1], 16;"
                :
                : "r"(sh_kv_add + ((threadIdx.x%8/2*32 + threadIdx.x%2*16 + threadIdx.x/8) ^ (threadIdx.x%8)) * 16),
                "l"(&in_v_p[threadIdx.x * 8]));
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    #pragma unroll(3)
    for (int i = 1; i < 4; i++){
        asm volatile("cp.async.ca.shared.global"
                    "[%0], [%1], 16;"
                    :
                    : "r"(sh_kv_add + i%2*64*32 + ((threadIdx.x%8/2*32 + threadIdx.x%2*16 + threadIdx.x/8) ^ (threadIdx.x%8)) * 16),
                    "l"(&in_v_p[threadIdx.x * 8 + i * 16*64]));
        asm volatile("cp.async.commit_group;\n" ::);
        #pragma unroll(4)
        for (int j = 0; j < 4; j++){
            int cal_i = i - 1;
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                        "{%0,%1,%2,%3}, [%4];"
                        : "=r"(mat_b[0]), "=r"(mat_b[1]), "=r"(mat_b[2]), "=r"(mat_b[3])
                        : "r"(sh_kv_add + cal_i%2*64*32 + ((j*32 + warp_tx)^(j*2 + warp_tx/16))*16));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                        : "=r"(mat_res[cal_i * 4]), "=r"(mat_res[cal_i * 4 + 1])
                        : "r"(mat_a[j*4]), "r"(mat_a[j*4 + 1]), "r"(mat_a[j*4 + 2]), "r"(mat_a[j*4 + 3]),
                        "r"(mat_b[0]), "r"(mat_b[2]), "r"(mat_res[cal_i*4]), "r"(mat_res[cal_i*4 + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                        : "=r"(mat_res[cal_i * 4 + 2]), "=r"(mat_res[cal_i * 4 + 3])
                        : "r"(mat_a[j*4]), "r"(mat_a[j*4 + 1]), "r"(mat_a[j*4 + 2]), "r"(mat_a[j*4 + 3]),
                        "r"(mat_b[1]), "r"(mat_b[3]), "r"(mat_res[cal_i*4 + 2]), "r"(mat_res[cal_i*4 + 3]));
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    #pragma unroll(4)
    for (int j = 0; j < 4; j++){
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                    "{%0,%1,%2,%3}, [%4];"
                    : "=r"(mat_b[0]), "=r"(mat_b[1]), "=r"(mat_b[2]), "=r"(mat_b[3])
                    : "r"(sh_kv_add + 64*32 + ((j*32 + warp_tx)^(j*2 + warp_tx/16))*16));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                    : "=r"(mat_res[12]), "=r"(mat_res[13])
                    : "r"(mat_a[j*4]), "r"(mat_a[j*4 + 1]), "r"(mat_a[j*4 + 2]), "r"(mat_a[j*4 + 3]),
                    "r"(mat_b[0]), "r"(mat_b[2]), "r"(mat_res[12]), "r"(mat_res[13]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                    : "=r"(mat_res[14]), "=r"(mat_res[15])
                    : "r"(mat_a[j*4]), "r"(mat_a[j*4 + 1]), "r"(mat_a[j*4 + 2]), "r"(mat_a[j*4 + 3]),
                    "r"(mat_b[1]), "r"(mat_b[3]), "r"(mat_res[14]), "r"(mat_res[15]));
    }

    // warp内合并 然后写出
    half* attn_out_p = attn_out + group_id * 64 * 64;
    mat_a = reinterpret_cast<uint32_t*>(reg_res);
    mat_b = reinterpret_cast<uint32_t*>(reg_final);
    #pragma unroll(4)
    for (int i = 0; i < 4; i++){
        // mat_a[i*4] = __shfl_sync(0xffffffff, mat_b[i*4 + warp_tx % 4], warp_tx / 4 * 4, 32);
        // mat_a[i*4 + 1] = __shfl_sync(0xffffffff, mat_b[i*4 + warp_tx % 4], warp_tx / 4 * 4 + 1, 32);
        // mat_a[i*4 + 2] = __shfl_sync(0xffffffff, mat_b[i*4 + warp_tx % 4], warp_tx / 4 * 4 + 2, 32);
        // mat_a[i*4 + 3] = __shfl_sync(0xffffffff, mat_b[i*4 + warp_tx % 4], warp_tx / 4 * 4 + 3, 32);
        // reinterpret_cast<float4*>(attn_out_p + warp_id * 16*64 + (warp_tx / 4 + warp_tx % 2 * 8) * 64 + i * 16)[warp_tx % 4 / 2] = 
        //         reinterpret_cast<float4*>(mat_a)[i];
        #pragma unroll(4)
        for (int j = 0; j < 4; j++) reinterpret_cast<float*>(sh_q + warp_id*16*16 + j*8*8)[warp_tx] = reinterpret_cast<float*>(reg_final)[i*4 + j];
        reinterpret_cast<float4*>(attn_out_p + warp_id * 16*64 + warp_tx % 16 * 64 + i * 16)[warp_tx / 16] = 
                reinterpret_cast<float4*>(sh_q + warp_id*16*16)[warp_tx];
    }
}