__kernel void applyKernel(
    __global uchar *original_img,
    __global uchar *processed_img,
    __global float *flat_kernel,
    int width,
    int height,
    int channels,
    int kernel_size)
{
    int posx = get_global_id(1);
    int posy = get_global_id(0);

    int w = width;
    float result_red = 0;
    float result_green = 0;
    float result_blue = 0;

    int kernel_size_offset = (kernel_size - 1) / 2;

    if (posx >= kernel_size_offset && posx < width - kernel_size_offset &&
        posy >= kernel_size_offset && posy < height - kernel_size_offset) {
        
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int x = posx + i - kernel_size_offset;
                int y = posy + j - kernel_size_offset;

                int pixel_index = (y * w + x) * channels;
                int kernel_value = flat_kernel[i * kernel_size + j];

                result_red += (float)original_img[pixel_index] * kernel_value;
                result_green += (float)original_img[pixel_index + 1] * kernel_value;
                result_blue += (float)original_img[pixel_index + 2] * kernel_value;
            }
        }

        // Update the result in the processed_img
        int processed_pixel_index = (posy * w + posx) * channels;
        processed_img[processed_pixel_index] = (char)result_red;
        processed_img[processed_pixel_index + 1] = (char)result_green;
        processed_img[processed_pixel_index + 2] = (char)result_blue;
    }
    else {
        // Copy the pixel from the original image to the processed image
        int pixel_index = (posy * w + posx) * channels;
        processed_img[pixel_index] = original_img[pixel_index];
        processed_img[pixel_index + 1] = original_img[pixel_index + 1];
        processed_img[pixel_index + 2] = original_img[pixel_index + 2];
    }
}
