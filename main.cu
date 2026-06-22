#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------
// Macros & Hyperparameters
// ---------------------------------------------------------
#define TILE_DIM 16
#define FILTER_RADIUS 3
#define HALO_DIM (TILE_DIM + 2 * FILTER_RADIUS)

// Robust CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Fatal Error at %s:%d - code=%d(%s)\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Constant memory for O(1) spatial weight broadcasts
__constant__ float c_spatial_kernel[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];

// ---------------------------------------------------------
// Device: CUDA Bilateral Filter with Shared Memory Tiling
// ---------------------------------------------------------
__global__ void gpu_bilateral_shared(const float* input, float* output, int width, int height, float sigma_r) {
    // Allocate the L1 cache tile
    __shared__ float s_tile[HALO_DIM][HALO_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int x = bx + tx;
    int y = by + ty;

    // Phase 1: Collaborative 1D Strided Halo Loading
    int shared_size = HALO_DIM * HALO_DIM;
    int tid = ty * TILE_DIM + tx; 
    int block_size = TILE_DIM * TILE_DIM;

    for (int i = tid; i < shared_size; i += block_size) {
        int s_row = i / HALO_DIM;
        int s_col = i % HALO_DIM;
        
        int g_row = by - FILTER_RADIUS + s_row;
        int g_col = bx - FILTER_RADIUS + s_col;

        // Boundary Clamping to prevent segfaults
        g_row = max(0, min(g_row, height - 1));
        g_col = max(0, min(g_col, width - 1));

        s_tile[s_row][s_col] = input[g_row * width + g_col];
    }

    // Hardware barrier: Guarantee all L1 reads are complete
    __syncthreads();

    // Phase 2: Non-Linear Convolution
    if (x < width && y < height) {
        float i_center = s_tile[ty + FILTER_RADIUS][tx + FILTER_RADIUS];
        float filtered_pixel = 0.0f;
        float w_p = 0.0f;
        int spatial_idx = 0;

        // Hoist the constant division out of the O(r^2) loops
        float sigma_r_coeff = -1.0f / (2.0f * sigma_r * sigma_r);

        for (int row = -FILTER_RADIUS; row <= FILTER_RADIUS; ++row) {
            for (int col = -FILTER_RADIUS; col <= FILTER_RADIUS; ++col) {
                float i_neighbor = s_tile[ty + FILTER_RADIUS + row][tx + FILTER_RADIUS + col];
                
                // Photometric range weight calculation (Optimized)
                float diff = i_center - i_neighbor;
                float range_weight = expf((diff * diff) * sigma_r_coeff);
                
                // Fetch O(1) spatial weight
                float spatial_weight = c_spatial_kernel[spatial_idx++];
                
                float combined_weight = spatial_weight * range_weight;
                filtered_pixel += i_neighbor * combined_weight;
                w_p += combined_weight;
            }
        }
        output[y * width + x] = filtered_pixel / w_p;
    }
}

// ---------------------------------------------------------
// Host: Precompute Spatial Kernel
// ---------------------------------------------------------
void precompute_spatial_kernel(float sigma_s) {
    int kernel_size = 2 * FILTER_RADIUS + 1;
    float* h_kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));
    
    if (!h_kernel) {
        fprintf(stderr, "Fatal: Failed to allocate host memory for spatial kernel.\n");
        exit(EXIT_FAILURE);
    }

    int idx = 0;
    for (int row = -FILTER_RADIUS; row <= FILTER_RADIUS; ++row) {
        for (int col = -FILTER_RADIUS; col <= FILTER_RADIUS; ++col) {
            h_kernel[idx++] = expf(-(float)(row * row + col * col) / (2.0f * sigma_s * sigma_s));
        }
    }
    
    // Transfer directly to symbol in constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_spatial_kernel, h_kernel, kernel_size * kernel_size * sizeof(float)));
    free(h_kernel);
}

// ---------------------------------------------------------
// Application Logic
// ---------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <input_image.png>\n", argv[0]);
        return -1;
    }

    int width, height, channels;
    // Force 1-channel grayscale load
    unsigned char* raw_img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!raw_img) {
        fprintf(stderr, "Fatal: Error decoding image from disk.\n");
        return -1;
    }

    // Cast dimensions to size_t to prevent 32-bit integer overflow on massive arrays
    size_t num_pixels = (size_t)width * (size_t)height;
    size_t bytes = num_pixels * sizeof(float);

    // Host allocations with safety checks
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "Fatal: Host memory allocation failed.\n");
        stbi_image_free(raw_img);
        return -1;
    }

    // Normalize uint8_t to 32-bit floats [0.0f, 1.0f]
    for (size_t i = 0; i < num_pixels; i++) {
        h_input[i] = (float)raw_img[i] / 255.0f;
    }

    float sigma_s = 50.0f;
    float sigma_r = 0.1f;
    precompute_spatial_kernel(sigma_s);

    // Device allocations and payload transfer
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    // Launch Kernel and check for execution failure
    gpu_bilateral_shared<<<gridDim, blockDim>>>(d_input, d_output, width, height, sigma_r);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Retrieve payload
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Allocate output image buffer
    unsigned char* out_img = (unsigned char*)malloc(num_pixels);
    if (!out_img) {
        fprintf(stderr, "Fatal: Failed to allocate host memory for output image.\n");
        // Teardown before exit
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        stbi_image_free(raw_img);
        return -1;
    }

    // Denormalize safely: Clamp and round to prevent truncation artifacts/overflows
    for (size_t i = 0; i < num_pixels; i++) {
        float scaled = h_output[i] * 255.0f;
        scaled = fmaxf(0.0f, fminf(scaled, 255.0f)); 
        out_img[i] = (unsigned char)(scaled + 0.5f);
    }

    // Write to disk
    if (!stbi_write_png("filtered_output.png", width, height, 1, out_img, width)) {
        fprintf(stderr, "Fatal: Failed to write output PNG to disk.\n");
    } else {
        printf("Filtering complete. Output successfully saved to filtered_output.png\n");
    }

    // Teardown
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    free(out_img);
    stbi_image_free(raw_img); // STB specific free to prevent allocator mismatch

    return 0;
}
