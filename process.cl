uchar _clamp(float value, uchar min_val, uchar max_val) {
    if (value < min_val) {
        return min_val;
    } else if (value > max_val) {
        return max_val;
    } else {
        return (uchar)(value);
    }
}

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

    int h = height;
    int w = width;
    float result_red = 0;
    float result_green = 0;
    float result_blue = 0;

    int kernel_size_offset = (kernel_size - 1) / 2;

    if (posx >= kernel_size_offset && posx < w - kernel_size_offset &&
        posy >= kernel_size_offset && posy < h - kernel_size_offset) {
        
        float result_red = 0;
        float result_green = 0;
        float result_blue = 0;

        for (int i = 0; i < kernel_size; i++) {
            
            int x = posx + i - kernel_size_offset;              
            // show value of x and i
            for (int j = 0; j < kernel_size; j++) {  

                int y = posy + j - kernel_size_offset;

                int pixel_index = (y * w + x) * channels;
                float kernel_value = flat_kernel[i * kernel_size + j];

                result_red += (float)original_img[pixel_index] * kernel_value;
                result_green += (float)original_img[pixel_index + 1] * kernel_value;
                result_blue += (float)original_img[pixel_index + 2] * kernel_value;
            }
        }

        // Clamp and update the result in the processed_img
        int processed_pixel_index = (posy * w + posx) * channels;
        processed_img[processed_pixel_index] = _clamp((char)result_red, 0, 255);
        processed_img[processed_pixel_index + 1] = _clamp((char)result_green, 0, 255);
        processed_img[processed_pixel_index + 2] = _clamp((char)result_blue, 0, 255);
    }
    else {
        // Copy the pixel from the original image to the processed image
        int pixel_index = (posy * w + posx) * channels;
        processed_img[pixel_index] = original_img[pixel_index];
        processed_img[pixel_index + 1] = original_img[pixel_index + 1];
        processed_img[pixel_index + 2] = original_img[pixel_index + 2];
    }
}

