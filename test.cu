//暂存点，superpoint nms 和 softmax

#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <vector>

#include "cuda_fp16.h"

#include <cstdlib>
#include <cstring>

__global__ void transForNMSByte2Kernel(int16_t* src_p, int16_t* dst_p, int h, int w){
    //c1, c2, h, w -> h, c1, w, c2
    //warp->1h,32w  1c1,8c2, B=256=8warp->1h,32w
    //w%32==0, c1==c2==8, data=int8
    //compute head, for c1, trans c2 , w
    __shared__ int16_t sh_data[8*32*8];
    int warp_id = threadIdx.x / 32;
    int warp_tx = threadIdx.x % 32;
    int w_group_id = blockIdx.x;
    int h_id = blockIdx.y;
    int c1_id = warp_id;
    int16_t* src_head_p = src_p + c1_id*8*h*w + h_id*w + w_group_id*32;
    int16_t* dst_head_p = dst_p + h_id*8*w*8 + c1_id*w*8 + w_group_id*32*8;
    int16_t reg_data[8];
    // #pragma unroll(8)
    // for (int i = 0; i < 8; i++){
    //     reg_data[i] = src_head_p[i*h*w + warp_tx];
    // }
    reinterpret_cast<float4*>(sh_data + warp_id*32*8)[warp_tx] = reinterpret_cast<float4*>(src_head_p + warp_tx/4*h*w)[warp_tx%4];
    #pragma unroll(8)
    for (int i = 0; i < 8; i++) reg_data[i] = sh_data[warp_id*32*8 + i*32 + warp_tx];
    reinterpret_cast<float4*>(dst_head_p)[warp_tx] = reinterpret_cast<float4*>(reg_data)[0];
}

void transForNMSByte2(int16_t* src_p, int16_t* dst_p, int h, int w, cudaStream_t stream){
    dim3 dimGrid(w/32, h);
    dim3 dimBlock(256);
    transForNMSByte2Kernel<<<dimGrid, dimBlock, 0, stream>>>(src_p, dst_p, h, w);
}

void transCpuTest(int16_t* src_p, int16_t* dst_p, int h, int w){
    //c1, c2, h, w -> h, c1, w, c2; c1=c2=8
    for (int c1 = 0; c1 < 8; c1++){
        for (int c2 = 0; c2 < 8; c2++){
            for (int h_id = 0; h_id < h; h_id++){
                for (int w_id = 0; w_id < w; w_id++){
                    dst_p[h_id*8*w*8+c1*w*8+w_id*8+c2] = src_p[c1*8*h*w+c2*h*w+h_id*w+w_id];
                }
            }
        }
    }
}

__global__ void findWindowMaxInRowKernel(half* score_map_p, half* pixel_row_max_score_p, int* pixel_row_max_id_p, int w, int range){
    //G=h,B=128, B->row t->? 可考虑 warp->row
    //w<1024! w%32==0 format h,w
    //需检查读sh时是否有冲突存在，若有可考虑bank不占满的形式
    __shared__ half sh_score_row[1024];
    int sh_load_round = (w + 127) / 128;
    int h_id = blockIdx.x;
    for (int i = 0; i < sh_load_round; i++){
        int w_id = i*128 + threadIdx.x;
        if (w_id < w) sh_score_row[w_id] = score_map_p[h_id*w + w_id];
    }
    __syncthreads();
    for (int i = 0; i < sh_load_round; i++){
        int w_id = i*128 + threadIdx.x;
        if (w_id < w){
            half max_score = sh_score_row[w_id];
            int max_id = w_id;
            for (int j = -range; j < 0; j++){
                int check_id = max(0, w_id + j);
                if (max_score < sh_score_row[check_id]){
                    max_score = sh_score_row[check_id];
                    max_id = check_id;
                }
            }
            for (int j = 1; j <= range; j++){
                int check_id = min(w - 1, w_id + j);
                if (max_score < sh_score_row[check_id]){
                    max_score = sh_score_row[check_id];
                    max_id = check_id;
                }
            }
            max_id = h_id*w + max_id;
            pixel_row_max_score_p[h_id*w + w_id] = max_score;
            pixel_row_max_id_p[h_id*w + w_id] = max_id;
        }
    }
}
__global__ void transMaxScoreAndIdHW32Byte2Kernel(half* src_max_score_p, int* src_max_id_p, half* dst_max_score_p, int* dst_max_id_p, int h, int w){
    //G=h/32 * w/32, B=128, warp->16*16, B->32*32
    //可考虑sh转接
    // __shared__ half sh_max_score[4*16*16];
    // __shared__ int sh_max_id[4*16*16];
    int warp_id = threadIdx.x / 32;
    int warp_tx = threadIdx.x % 32;
    int h_warp_id = blockIdx.x*2 + warp_id/2;
    int w_warp_id = blockIdx.y*2 + warp_id%2;
    int src_warp_idx = h_warp_id*16*w + w_warp_id*16;
    int dst_warp_idx = w_warp_id*16*h + h_warp_id*16;
    half reg_score[8];
    int reg_id[8];
    reinterpret_cast<float4*>(reg_score)[0] = reinterpret_cast<float4*>(src_max_score_p + src_warp_idx + warp_tx%16*w + warp_tx/16*8)[0];
    reinterpret_cast<float4*>(reg_id)[0] = reinterpret_cast<float4*>(src_max_id_p + src_warp_idx + warp_tx%16*w + warp_tx/16*4)[0];
    reinterpret_cast<float4*>(reg_id)[1] = reinterpret_cast<float4*>(src_max_id_p + src_warp_idx + warp_tx%16*w + warp_tx/16*4 + 8)[0];
    #pragma unroll(8)
    for (int i = 0; i < 8; i++) dst_max_score_p[dst_warp_idx + warp_tx/16*8*h + i*h + warp_tx%16] = reg_score[i];
    #pragma unroll(4)
    for (int i = 0; i < 4; i++) dst_max_id_p[dst_warp_idx + warp_tx/16*4*h + i*h + warp_tx%16] = reg_id[i];
    #pragma unroll(4)
    for (int i = 0; i < 4; i++) dst_max_id_p[dst_warp_idx + 8*h + warp_tx/16*4*h + i*h + warp_tx%16] = reg_id[i + 4];

    // half* warp_sh_max_score_p = sh_max_score + warp_id*16*16;
    // int* warp_sh_max_id_p = sh_max_id + warp_id*16*16;
    // reinterpret_cast<float4*>(warp_sh_max_score_p)[warp_tx^((warp_tx/16)*2)] = reinterpret_cast<float4*>(src_max_score_p + src_warp_idx + warp_tx/2*w)[warp_tx%2];
    // reinterpret_cast<float4*>(warp_sh_max_id_p)[warp_tx^((warp_tx/16)*4)] = reinterpret_cast<float4*>(src_max_id_p + src_warp_idx + warp_tx/4*w)[warp_tx%4];
    // reinterpret_cast<float4*>(warp_sh_max_id_p + 8*16)[warp_tx^((warp_tx/16)*4)] = reinterpret_cast<float4*>(src_max_id_p + src_warp_idx + warp_tx/4*w + 8*w)[warp_tx%4];
    // #pragma unroll(8)
    // for (int i = 0; i < 8; i++) reg_score[i] = warp_sh_max_score_p[warp_tx/16*8*16 + (i^(warp_tx/16))*16 + warp_tx%16];
    // #pragma unroll(4)
    // for (int i = 0; i < 4; i++){
    //     reg_id[i] = warp_sh_max_id_p[warp_tx/16*4*16 + (i^(warp_tx/16))*16 + warp_tx%16];
    //     reg_id[i + 4] = warp_sh_max_id_p[warp_tx/16*4*16 + (i^(warp_tx/16))*16 + 8*16 + warp_tx%16];
    // } 
    // reinterpret_cast<float4*>(dst_max_score_p + dst_warp_idx + warp_tx%16*h)[warp_tx/16] = reinterpret_cast<float4*>(reg_score)[0];
    // reinterpret_cast<float4*>(dst_max_id_p + dst_warp_idx + warp_tx%16*h)[warp_tx/16] = reinterpret_cast<float4*>(reg_id)[0];
    // reinterpret_cast<float4*>(dst_max_id_p + dst_warp_idx + warp_tx%16*h + 8)[warp_tx/16] = reinterpret_cast<float4*>(reg_id)[1];
}
__global__ void findWindowMaxInColKernel(half* pixel_row_max_score_p, int* pixel_row_max_id_p, half* pixel_max_score_p, int* pixel_max_id_p, int h, int range){
    //G=w，B=128 B->col(row)
    //h<1024! h%32==0, format w,h
    //与row处理一致，考虑一致
    __shared__ half sh_score_col[1024];
    __shared__ int sh_id_col[1024];
    int sh_load_round = (h + 127) / 128;
    int w_id = blockIdx.x;
    for (int i = 0; i < sh_load_round; i++){
        int h_id = i*128 + threadIdx.x;
        if (h_id < h){
            sh_score_col[h_id] = pixel_row_max_score_p[w_id*h + h_id];
            sh_id_col[h_id] = pixel_row_max_id_p[w_id*h + h_id];
        }
    }
    __syncthreads();
    for (int i = 0; i < sh_load_round; i++){
        int h_id = i*128 + threadIdx.x;
        if (h_id < h){
            half max_score = sh_score_col[h_id];
            int max_id = sh_id_col[h_id];
            for (int j = -range; j < 0; j++){
                int check_id = max(0, h_id + j);
                if (max_score < sh_score_col[check_id]){
                    max_score = sh_score_col[check_id];
                    max_id = sh_id_col[check_id];
                }
            }
            for (int j = 1; j <= range; j++){
                int check_id = min(h - 1, h_id + j);
                if (max_score < sh_score_col[check_id]){
                    max_score = sh_score_col[check_id];
                    max_id = sh_id_col[check_id];
                }
            }
            pixel_max_score_p[w_id*h + h_id] = max_score;
            pixel_max_id_p[w_id*h + h_id] = max_id;
        }
    }
}
__global__ void extractAndSuppressKernel(half* score_map_p, half* extracted_score_map_p, half* pixel_max_score_p, int* pixel_max_id_p, int h, int w, int range){
    //map format h,w; pixel format w,h(可考虑倒转,则全部需倒转考虑), id from format h,w
    //h,w%32==0; h,w < 1024! range<16!
    //extracted map need set zero pre
    //G=w, B=256, B->1w*pixel->extracted / warp->1*score-set-zero
    __shared__ int sh_extracted_h[8 * 128];
    __shared__ int sh_warp_extracted_num[8];
    int warp_id = threadIdx.x / 32;
    int warp_tx = threadIdx.x % 32;
    int load_round = (h + 255) / 256;
    int w_id = blockIdx.x;
    if (warp_tx == 0) sh_warp_extracted_num[warp_id] = 0;
    for (int i = 0; i < load_round; i++){
        int h_id = i*256 + threadIdx.x;
        if (h_id < h){
            int pixel_id = w_id * h + h_id;
            int max_map_id = pixel_max_id_p[pixel_id];
            half max_map_score = pixel_max_score_p[pixel_id];
            if (max_map_id == h_id * w + w_id && max_map_score > static_cast<half>(0.0f)){
                int warp_extracted_id = atomicAdd(sh_warp_extracted_num + warp_id, 1);
                sh_extracted_h[warp_id*128 + warp_extracted_id] = h_id;
                extracted_score_map_p[max_map_id] = max_map_score;
            }
        }
    }
    int reg_warp_extracted_num = sh_warp_extracted_num[warp_id];
    int th_w_id = w_id + min(warp_tx, 2*range) - range;
    th_w_id = max(0, min(th_w_id, w - 1));
    for (int i = 0; i < reg_warp_extracted_num; i++){
        int h_id = sh_extracted_h[warp_id*128 + i];
        int start_row = max(0, h_id - range);
        int end_row = min(h - 1, h_id + range);
        for (int row = start_row; row <= end_row; row++) score_map_p[row * w + th_w_id] = static_cast<half>(0.0f);
    }
}

__global__ void fusedColFindingAndSuppress(half* pixel_row_max_score_p, int* pixel_row_max_id_p, half* score_map_p, half* extracted_score_map_p,
                                            int h, int w, int range){
    //G=w，B=256 B->col(row)
    //h<1024! h%32==0, format w,h
    //与row处理一致，考虑一致
    __shared__ half sh_score_col[1024];
    __shared__ int sh_id_col[1024];
    __shared__ int sh_extracted_h[8 * 128];
    __shared__ int sh_warp_extracted_num[8];
    int sh_load_round = (h + 255) / 256;
    int w_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int warp_tx = threadIdx.x % 32;
    if (warp_tx == 0) sh_warp_extracted_num[warp_id] = 0;
    for (int i = 0; i < sh_load_round; i++){
        int h_id = i*256 + threadIdx.x;
        if (h_id < h){
            sh_score_col[h_id] = pixel_row_max_score_p[w_id*h + h_id];
            sh_id_col[h_id] = pixel_row_max_id_p[w_id*h + h_id];
        }
    }
    __syncthreads();
    for (int i = 0; i < sh_load_round; i++){
        int h_id = i*256 + threadIdx.x;
        if (h_id < h){
            half max_score = sh_score_col[h_id];
            int max_id = sh_id_col[h_id];
            for (int j = -range; j < 0; j++){
                int check_id = max(0, h_id + j);
                if (max_score < sh_score_col[check_id]){
                    max_score = sh_score_col[check_id];
                    max_id = sh_id_col[check_id];
                }
            }
            for (int j = 1; j <= range; j++){
                int check_id = min(h - 1, h_id + j);
                if (max_score < sh_score_col[check_id]){
                    max_score = sh_score_col[check_id];
                    max_id = sh_id_col[check_id];
                }
            }
            if (max_id == h_id * w + w_id && max_score > static_cast<half>(0.0f)){
                int warp_extracted_id = atomicAdd(sh_warp_extracted_num + warp_id, 1);
                sh_extracted_h[warp_id*128 + warp_extracted_id] = h_id;
                extracted_score_map_p[max_id] = max_score;
            }
        }
    }
    int reg_warp_extracted_num = sh_warp_extracted_num[warp_id];
    int th_w_id = w_id + min(warp_tx, 2*range) - range;
    th_w_id = max(0, min(th_w_id, w - 1));
    for (int i = 0; i < reg_warp_extracted_num; i++){
        int h_id = sh_extracted_h[warp_id*128 + i];
        int start_row = max(0, h_id - range);
        int end_row = min(h - 1, h_id + range);
        for (int row = start_row; row <= end_row; row++) score_map_p[row * w + th_w_id] = static_cast<half>(0.0f);
    }
}

void rangeNMSExtract(half* score_map_p, half* extracted_score_map_p,
                    half* pixel_row_max_score_p, half* pixel_max_score_p, int* pixel_row_max_id_p, int* pixel_max_id_p,
                    half* transposed_pixel_row_max_score_p, int* transposed_pixel_row_max_id_p,
                    int h, int w, int range, int nms_round, cudaStream_t stream){
    cudaMemsetAsync(extracted_score_map_p, 0, sizeof(half)*w*h, stream);
    dim3 dimGrid(h);
    dim3 dimBlock(128);
    for (int round = 0; round < nms_round; round++){
        dimGrid.x = h;
        dimBlock.x = 128;
        findWindowMaxInRowKernel<<<dimGrid, dimBlock, 0, stream>>>(score_map_p, pixel_row_max_score_p, pixel_row_max_id_p, w, range);
        dimGrid.x = h / 32;
        dimGrid.y = w / 32;
        transMaxScoreAndIdHW32Byte2Kernel<<<dimGrid, dimBlock, 0, stream>>>(pixel_row_max_score_p, pixel_row_max_id_p,
                                                                    transposed_pixel_row_max_score_p, transposed_pixel_row_max_id_p, h, w);
        // dimGrid = dim3(w);
        // findWindowMaxInColKernel<<<dimGrid, dimBlock, 0, stream>>>(transposed_pixel_row_max_score_p, transposed_pixel_row_max_id_p,
        //                                                     pixel_max_score_p, pixel_max_id_p, h, range);
        // dimGrid.x = w;
        // dimBlock.x = 256;
        // extractAndSuppressKernel<<<dimGrid, dimBlock,0, stream>>>(score_map_p, extracted_score_map_p, pixel_max_score_p, pixel_max_id_p, h, w, range);
        dimGrid = dim3(w);
        dimBlock.x = 256;
        fusedColFindingAndSuppress<<<dimGrid, dimBlock, 0, stream>>>(transposed_pixel_row_max_score_p, transposed_pixel_row_max_id_p,
                                                                    score_map_p, extracted_score_map_p, h, w, range);
    }   
}

void rangeNMSCpuTest(half* score_map_p, half* extracted_score_map_p, int h, int w, int range, int nms_round){
    std::vector<int> matched_score_id; //h,w
    std::vector<half> src_map_bp(h*w);
    std::memcpy(src_map_bp.data(), score_map_p, sizeof(half)*h*w);
    int matched_num = 0;
    std::memset(extracted_score_map_p, 0, sizeof(half)*h*w);
    for (int round = 0; round < nms_round; round++){
        matched_score_id.clear();
        matched_num = 0;
        int zero_num = 0;
        for (int h_id = 0; h_id < h; h_id++){
            for (int w_id = 0; w_id < w; w_id++){
                half max_score = score_map_p[h_id*w + w_id];
                int h_max = h_id;
                int w_max = w_id;
                for (int i = -range; i <= range; i++){
                    for (int j = -range; j <= range; j++){
                        int h_cmp = max(0, min(h_id + i, h - 1));
                        int w_cmp = max(0, min(w_id + j, w - 1));
                        if (max_score < score_map_p[h_cmp*w + w_cmp]){ //存在范围内出现等值可能
                            h_max = h_cmp;
                            w_max = w_cmp;
                            max_score = score_map_p[h_cmp*w + w_cmp];
                        }
                    }
                }
                if (h_max == h_id && w_max == w_id){
                    if (max_score <= static_cast<half>(0.0f)){continue;zero_num++;} 
                    matched_score_id.emplace_back(h_max);
                    matched_score_id.emplace_back(w_max);
                    matched_num++;
                    extracted_score_map_p[h_max*w + w_max] = max_score;
                }
            }
        }
        
        std::cout<<"cpu1: "<<matched_num<<' ' <<zero_num<<std::endl;
        for (int max_id = 0; max_id < matched_num; max_id++){
            int h_max = matched_score_id[max_id*2];
            int w_max = matched_score_id[max_id*2 + 1];
            // extracted_score_map_p[h_max*w + w_max] = score_map_p[h_max*w + w_max];
            for (int i = -range; i <= range; i++){
                for (int j = -range; j <= range; j++){
                    int h_id = max(0, min(h_max + i, h - 1));
                    int w_id = max(0, min(w_max + j, w - 1));
                    // if (h_id == h_max && w_id == w_max) continue;
                    score_map_p[h_id*w + w_id] = static_cast<half>(0.0f);
                }
            }
        }
    }
}

__global__ void softmaxHalfForSuperpointKernel(half* input_p, half* output_p, int scope){
    //scope = h*w, cin=65, cout=64,drop last cin
    //shape = c, scope
    //G=h*w/64, B=128, warp->16*scope， t->8*scope for max/sum(4c), 32c for res
    //h,w%32==0! h,w<1024!
    __shared__ half sh_data[4*64*16];
    half reg_data[8];
    half reg_max[8];
    float reg_sum[8];
    int warp_id = threadIdx.x / 32;
    int warp_tx = threadIdx.x % 32;
    half* warp_input_p = input_p + blockIdx.x*64 + warp_id*16;
    half* warp_output_p = output_p + blockIdx.x*64 + warp_id*16;
    #pragma unroll(8)
    for (int i = 0; i < 8; i++){
        reg_max[i] = CUDART_MIN_DENORM_FP16;
        reg_sum[i] = 0.0f;
    }
    #pragma unroll(4)
    for (int i = 0; i < 4; i++){
        reinterpret_cast<float4*>(reg_data)[0] = reinterpret_cast<float4*>(warp_input_p + i*16*scope + warp_tx/2*scope)[warp_tx%2];
        reinterpret_cast<float4*>(sh_data + warp_id*64*16 + i*16*16)[warp_tx] = reinterpret_cast<float4*>(reg_data)[0];
        #pragma unroll(4)
        for (int j = 0; j < 4; j++){
            half2 last_max = reinterpret_cast<half2*>(reg_max)[j];
            half2 cur_data = reinterpret_cast<half2*>(reg_data)[j];
            half2 cur_max = __hmax2(last_max, cur_data);
            reinterpret_cast<half2*>(reg_max)[j] = cur_max;
            reg_sum[j*2] = reg_sum[j*2] * __expf(static_cast<float>(last_max.x) - static_cast<float>(cur_max.x))
                            + __expf(static_cast<float>(cur_data.x) - static_cast<float>(cur_max.x));
            reg_sum[j*2 + 1] = reg_sum[j*2 + 1] * __expf(static_cast<float>(last_max.y) - static_cast<float>(cur_max.y))
                            + __expf(static_cast<float>(cur_data.y) - static_cast<float>(cur_max.y));
        }
    }

    #pragma unroll(4)
    for (int i = 0; i < 4; i++){
        #pragma unroll(4)
        for (int j = 0; j < 4; j++){
            half2 last_max = reinterpret_cast<half2*>(reg_max)[j];
            half2 cur_data = __shfl_xor_sync(0xffffffff, reinterpret_cast<half2*>(reg_max)[j], 0x2<<i);
            half2 cur_max = __hmax2(last_max, cur_data);
            reinterpret_cast<half2*>(reg_max)[j] = cur_max;
            float tmp_sum = __shfl_xor_sync(0xffffffff, reg_sum[j*2], 0x2<<i);
            reg_sum[j*2] = reg_sum[j*2] * __expf(static_cast<float>(last_max.x) - static_cast<float>(cur_max.x))
                            + tmp_sum * __expf(static_cast<float>(cur_data.x) - static_cast<float>(cur_max.x));
            tmp_sum = __shfl_xor_sync(0xffffffff, reg_sum[j*2 + 1], 0x2<<i);
            reg_sum[j*2 + 1] = reg_sum[j*2 + 1] * __expf(static_cast<float>(last_max.y) - static_cast<float>(cur_max.y))
                            + tmp_sum * __expf(static_cast<float>(cur_data.y) - static_cast<float>(cur_max.y));
        }
    }

    reinterpret_cast<float4*>(reg_data)[0] = reinterpret_cast<float4*>(warp_input_p + 64*scope)[warp_tx%2];
    #pragma unroll(4)
    for (int i = 0; i < 4; i++){
        half2 last_max = reinterpret_cast<half2*>(reg_max)[i];
        half2 cur_data = reinterpret_cast<half2*>(reg_data)[i];
        half2 cur_max = __hmax2(last_max, cur_data);
        reinterpret_cast<half2*>(reg_max)[i] = cur_max;
        reg_sum[i*2] = reg_sum[i*2] * __expf(static_cast<float>(last_max.x) - static_cast<float>(cur_max.x))
                        + __expf(static_cast<float>(cur_data.x) - static_cast<float>(cur_max.x));
        reg_sum[i*2 + 1] = reg_sum[i*2 + 1] * __expf(static_cast<float>(last_max.y) - static_cast<float>(cur_max.y))
                        + __expf(static_cast<float>(cur_data.y) - static_cast<float>(cur_max.y));
    }

    #pragma unroll(4)
    for (int i = 0; i < 4; i++){
        reinterpret_cast<float4*>(reg_data)[0] = reinterpret_cast<float4*>(sh_data + warp_id*64*16 + i*16*16)[warp_tx];
        #pragma unroll(4)
        for (int j = 0; j < 4; j++){
            half2 cur_logit = __hsub2(reinterpret_cast<half2*>(reg_data)[j], reinterpret_cast<half2*>(reg_max)[j]);
            reg_data[j*2] = static_cast<half>(__expf(static_cast<float>(cur_logit.x)) / reg_sum[j*2]);
            reg_data[j*2 + 1] = static_cast<half>(__expf(static_cast<float>(cur_logit.y)) / reg_sum[j*2 + 1]);
        }
        reinterpret_cast<float4*>(warp_output_p + i*16*scope + warp_tx/2*scope)[warp_tx%2] = reinterpret_cast<float4*>(reg_data)[0];
    }
}

void softmaxHalfForSuperpoint(half* input_p, half* output_p, int h, int w, cudaStream_t stream){
    dim3 dimGrid(h*w / 64);
    dim3 dimBlock(128);
    softmaxHalfForSuperpointKernel<<<dimGrid, dimBlock, 0, stream>>>(input_p, output_p, h*w);
}

void softmaxHalfForSuperpointCpuTest(half* input_p, half* output_p, int h, int w){
    int c_in = 65;
    int c_out = 64;
    for (int n = 0; n < h*w; n++){
        float c_max = static_cast<float>(input_p[64*h*w + n]);
        for (int c = 0; c < c_out; c++){
            c_max = std::max(c_max, static_cast<float>(input_p[c*h*w + n]));
        }
        float c_sum = std::exp(static_cast<float>(input_p[64*h*w + n]) - c_max);
        for (int c = 0; c < c_out; c++){
            c_sum = c_sum + std::exp(static_cast<float>(input_p[c*h*w + n]) - c_max);
        }
        for (int c = 0; c < c_out; c++){
            output_p[c*h*w + n] = static_cast<half>(std::exp(static_cast<float>(input_p[c*h*w + n]) - c_max) / c_sum);
        }
    }
}

int main(){
    int h = 64;
    int w = 64;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<half> soft_input(65*h*w);
    std::vector<half> soft_output(64*h*w, static_cast<half>(0.0f));
    std::vector<half> soft_output_gpu(64*h*w, static_cast<half>(0.0f));
    for (int i = 0; i < soft_input.size(); i++){
        half tmp = static_cast<half>(static_cast<float>(std::abs(rand())) / static_cast<float>(RAND_MAX));
        soft_input[i] = tmp;
    }
    half* soft_input_p = nullptr;
    half* soft_output_p = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&soft_input_p), 65*h*w*sizeof(half));
    cudaMalloc(reinterpret_cast<void**>(&soft_output_p), 64*h*w*sizeof(half));
    cudaMemset(soft_output_p, 64*h*w*sizeof(half), 0);
    cudaMemcpy(soft_input_p, soft_input.data(), 65*h*w*sizeof(half), cudaMemcpyHostToDevice);
    softmaxHalfForSuperpoint(soft_input_p, soft_output_p, h, w, stream);
    softmaxHalfForSuperpointCpuTest(soft_input.data(), soft_output.data(), h, w);
    cudaMemcpy(soft_output_gpu.data(), soft_output_p, 64*h*w*sizeof(half), cudaMemcpyDeviceToHost);
    for (int c = 0; c < 64; c++){
        float soft_max_err = 0.0f;
        float soft_sum_err = 0.0f;
        float abs_sum = 0.0f;
        float rec_ref = 0.0f;
        float rec_res = 0.0f;
        for (int n = 0; n < h*w; n++){
            float ref = static_cast<float>(soft_output[c*h*w + n]);
            float res = static_cast<float>(soft_output_gpu[c*h*w + n]);
            float err = std::abs(ref - res);
            if (err > soft_max_err){
                soft_max_err = err;
                rec_ref = ref;
                rec_res = res;
            }
            soft_sum_err += err;
            abs_sum += std::abs(ref);
        }
        std::cout<<c<<": max err: "<<soft_max_err<<", ref: "<<rec_ref<<", res: "<<rec_res<<", err sum: "<<soft_sum_err<<", ref sum: "<<abs_sum<<std::endl;
    }

    h = 64;
    w = 64;
    std::vector<int16_t> src(8*8*h*w);
    std::vector<int16_t> dst_cpu(8*8*h*w, 0);
    std::vector<int16_t> dst_gpu(8*8*h*w, 0);
    int16_t* src_p = nullptr;
    int16_t* dst_p = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&src_p), 8*8*h*w*sizeof(int16_t));
    cudaMalloc(reinterpret_cast<void**>(&dst_p), 8*8*h*w*sizeof(int16_t));
    cudaMemset(dst_p, 8*8*h*w*sizeof(int16_t), 0);
    for (int i = 0; i < src.size(); i++){
        int16_t tmp = static_cast<int16_t>(rand() % 256);
        src[i] = tmp;
    }
    cudaMemcpy(src_p, src.data(), 8*8*h*w*sizeof(int16_t), cudaMemcpyHostToDevice);
    transCpuTest(reinterpret_cast<int16_t*>(src.data()), reinterpret_cast<int16_t*>(dst_cpu.data()), h, w);
    transForNMSByte2(src_p, dst_p, h, w, stream);
    cudaMemcpy(dst_gpu.data(), dst_p, 8*8*h*w*sizeof(int16_t), cudaMemcpyDeviceToHost);
    int err = 0;
    for (int i = 0; i < src.size(); i++){
        err += std::abs(static_cast<int>(dst_cpu[i]) - static_cast<int>(dst_gpu[i]));
    }
    std::cout<<"trans for NMS err: "<<err<<std::endl;

    h = 512;
    w = 512;
    int nms_round = 3;
    int range = 4;
    std::vector<half> src_map(h*w);
    for (int i = 0; i < src_map.size(); i++){
        half tmp = static_cast<half>(static_cast<float>(std::abs(rand())) / static_cast<float>(RAND_MAX));
        src_map[i] = tmp;
    }
    std::vector<half> extracted_score_map_cpu(h*w);
    std::vector<half> extracted_score_map_gpu(h*w);
    
    half* score_map_p = nullptr;
    half* extracted_score_map_p = nullptr;
    half* pixel_row_max_score_p = nullptr;
    half* pixel_max_score_p = nullptr;
    half* transposed_pixel_row_max_score_p = nullptr;
    int* pixel_row_max_id_p = nullptr;
    int* pixel_max_id_p = nullptr;
    int* transposed_pixel_row_max_id_p = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&score_map_p), sizeof(half)*h*w);
    cudaMalloc(reinterpret_cast<void**>(&extracted_score_map_p), sizeof(half)*h*w);
    cudaMalloc(reinterpret_cast<void**>(&pixel_row_max_score_p), sizeof(half)*h*w);
    cudaMalloc(reinterpret_cast<void**>(&pixel_max_score_p), sizeof(half)*h*w);
    cudaMalloc(reinterpret_cast<void**>(&transposed_pixel_row_max_score_p), sizeof(half)*h*w);
    cudaMalloc(reinterpret_cast<void**>(&pixel_row_max_id_p), sizeof(int)*h*w);
    cudaMalloc(reinterpret_cast<void**>(&pixel_max_id_p), sizeof(int)*h*w);
    cudaMalloc(reinterpret_cast<void**>(&transposed_pixel_row_max_id_p), sizeof(int)*h*w);

    cudaMemcpyAsync(score_map_p, src_map.data(), sizeof(half)*h*w, cudaMemcpyHostToDevice, stream);
    rangeNMSExtract(score_map_p, extracted_score_map_p,
                    pixel_row_max_score_p, pixel_max_score_p, pixel_row_max_id_p, pixel_max_id_p,
                    transposed_pixel_row_max_score_p, transposed_pixel_row_max_id_p,
                    h, w, range, nms_round, stream);
    cudaMemcpyAsync(extracted_score_map_gpu.data(), extracted_score_map_p, sizeof(half)*h*w, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;

    std::memset(extracted_score_map_cpu.data(), 0, sizeof(half)*extracted_score_map_cpu.size());
    rangeNMSCpuTest(src_map.data(), extracted_score_map_cpu.data(), h, w, range, nms_round);

    float sum_err = 0.0f;
    float max_err = 0.0f;
    float rec_cpu = 0.0f;
    float rec_gpu = 0.0f;
    float abs_sum_cpu = 0.0f;
    float abs_sum_gpu = 0.0f;
    int gpu_extract_num = 0;
    int cpu_extract_num = 0;
    for (int i = 0; i < h*w; i++){
        float tmp_err = std::abs(static_cast<float>(extracted_score_map_cpu[i]) - static_cast<float>(extracted_score_map_gpu[i]));
        abs_sum_cpu += std::abs(static_cast<float>(extracted_score_map_cpu[i]));
        abs_sum_gpu += std::abs(static_cast<float>(extracted_score_map_gpu[i]));
        sum_err += tmp_err;
        if (tmp_err > max_err){
            max_err = tmp_err;
            rec_cpu = static_cast<float>(extracted_score_map_cpu[i]);
            rec_gpu = static_cast<float>(extracted_score_map_gpu[i]);
        }
        if (std::abs(static_cast<float>(extracted_score_map_cpu[i])) > 0.0f) cpu_extract_num++;
        if (std::abs(static_cast<float>(extracted_score_map_gpu[i])) > 0.0f) gpu_extract_num++;
    }
    std::cout<<cpu_extract_num<<" : "<<gpu_extract_num<<std::endl;
    std::cout<<"NMS err"<<std::endl;
    std::cout<<"sum err: "<<sum_err<<", max err: "<<max_err<<", rec cpu: "<<rec_cpu<<", rec gpu: "<<rec_gpu<<std::endl;
    std::cout<<"abs sum cpu: "<<abs_sum_cpu<<", abs sum gpu: "<<abs_sum_gpu<<std::endl;

    std::ofstream outfile;
    outfile.open("../cpu_extract.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(extracted_score_map_cpu.data()), sizeof(half)*h*w);
    outfile.close();
    outfile.open("../gpu_extract.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(extracted_score_map_gpu.data()), sizeof(half)*h*w);
    outfile.close();
}
