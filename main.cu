#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Include the STB image libraries for reading/writing PNG files
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------
// GPU Function: Basic Bilateral Filter
// ---------------------------------------------------------
// This function runs on the graphics card. Each thread calculates exactly ONE pixel.
__global__ void gpu_bilateral_basic(const float* input, float* output, int width, int height, float sigma_s, float sigma_r) {
    
    // 1. Find out which pixel this specific thread is supposed to process
    int x = blockIdx.x * blockDim.x + threadIdx.x; // The column (X coordinate)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // The row (Y coordinate)

    // 2. Stop the thread if it falls outside the actual image boundaries
    // (This happens because we launch threads in blocks of 16x16)
    if (x >= width || y >= height) {
        return; 
    }

    // 3. Read the center pixel's color value from the global GPU memory
    float center_color = input[y * width + x];
    
    float final_pixel_value = 0.0f; // This will hold our blurred result
    float total_weight = 0.0f;      // We use this to divide at the end so it doesn't get too bright

    // 4. Look at the surrounding 7x7 grid (3 pixels in every direction)
    int radius = 3;
    
    for (int row_offset = -radius; row_offset <= radius; row_offset++) {
        for (int col_offset = -radius; col_offset <= radius; col_offset++) {
            
            // Calculate the exact X and Y coordinate of the neighbor we are looking at
            int neighbor_x = x + col_offset;
            int neighbor_y = y + row_offset;

            // 5. Make sure the neighbor is actually inside the picture!
            // If we are at the edge of the image, we don't want to read outside of it.
            if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                
                // Read the neighbor's color from global GPU memory
                float neighbor_color = input[neighbor_y * width + neighbor_x];

                // 6. Calculate Spatial Weight (How far away is it physically?)
                // The further away it is, the lower the weight.
                float distance_squared = (row_offset * row_offset) + (col_offset * col_offset);
                float spatial_weight = expf(-distance_squared / (2.0f * sigma_s * sigma_s));

                // 7. Calculate Range Weight (How different is the color?)
                // If the color is very different (like an edge), the weight becomes almost zero.
                float color_diff = center_color - neighbor_color;
                float color_diff_squared = color_diff * color_diff;
                float range_weight = expf(-color_diff_squared / (2.0f * sigma_r * sigma_r));

                // 8. Multiply the two weights together to get the final importance of this neighbor
                float combined_weight = spatial_weight * range_weight;

                // Add the neighbor's color (scaled by its importance) to our running total
                final_pixel_value += neighbor_color * combined_weight;
                total_weight += combined_weight; // Keep track of the total weight used
            }
        }
    }

    // 9. Normalize the final pixel (divide by total weight) and save it to the output array
    output[y * width + x] = final_pixel_value / total_weight;
}

// ---------------------------------------------------------
// Main Program (Runs on the CPU)
// ---------------------------------------------------------
int main(int argc, char** argv) {
    
    // Make sure the user provided an image file
    if (argc < 2) {
        printf("Usage: ./bilateral_filter <image.png>\n");
        return -1;
    }

    // 1. Load the image from the hard drive
    int width, height, channels;
    unsigned char* raw_img = stbi_load(argv[1], &width, &height, &channels, 1);
    
    if (raw_img == NULL) {
        printf("Failed to load the image.\n");
        return -1;
    }

    // Calculate how much memory we need
    int total_pixels = width * height;
    int memory_size = total_pixels * sizeof(float);

    // Allocate memory on the CPU
    float* cpu_input = (float*)malloc(memory_size);
    float* cpu_output = (float*)malloc(memory_size);

    // Convert the image pixels from integers (0-255) to floats (0.0 to 1.0)
    for (int i = 0; i < total_pixels; i++) {
        cpu_input[i] = (float)raw_img[i] / 255.0f;
    }

    // Setup our filter strengths
    float sigma_s = 50.0f; // Spatial blur strength
    float sigma_r = 0.1f;  // Color edge-preservation strength

    // 2. Allocate memory on the GPU
    float *gpu_input, *gpu_output;
    cudaMalloc((void**)&gpu_input, memory_size);
    cudaMalloc((void**)&gpu_output, memory_size);

    // Copy the image from the CPU to the GPU
    cudaMemcpy(gpu_input, cpu_input, memory_size, cudaMemcpyHostToDevice);

    // 3. Setup the GPU execution grid (Divide the image into 16x16 blocks)
    int threads_per_block = 16;
    dim3 blockDim(threads_per_block, threads_per_block);
    
    // Calculate how many blocks we need to cover the entire width and height
    dim3 gridDim((width + threads_per_block - 1) / threads_per_block, 
                 (height + threads_per_block - 1) / threads_per_block);

    // 4. Run the function on the GPU
    gpu_bilateral_basic<<<gridDim, blockDim>>>(gpu_input, gpu_output, width, height, sigma_s, sigma_r);
    
    // Wait for the GPU to finish all its work
    cudaDeviceSynchronize();

    // Copy the finished image back from the GPU to the CPU
    cudaMemcpy(cpu_output, gpu_output, memory_size, cudaMemcpyDeviceToHost);

    // 5. Convert the pixels back from floats (0.0 to 1.0) to integers (0-255)
    unsigned char* final_image = (unsigned char*)malloc(total_pixels);
    
    for (int i = 0; i < total_pixels; i++) {
        float pixel_val = cpu_output[i] * 255.0f;
        
        // Make sure the values stay between 0 and 255
        if (pixel_val > 255.0f) pixel_val = 255.0f;
        if (pixel_val < 0.0f) pixel_val = 0.0f;
        
        final_image[i] = (unsigned char)pixel_val;
    }

    // Save the new image to the hard drive
    stbi_write_png("filtered_output.png", width, height, 1, final_image, width);
    printf("Successfully filtered the image!\n");

    // 6. Clean up memory to prevent leaks
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    free(cpu_input);
    free(cpu_output);
    free(final_image);
    stbi_image_free(raw_img);

    return 0;
}
