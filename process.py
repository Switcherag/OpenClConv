import numpy as np
from matplotlib.pyplot import imread, imsave
import pyopencl as cl
import time

# Read in the original image
original_img = imread("input.jpg").astype(np.uint8)

print(original_img.shape)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags
# Create the kernel matrix for edge detection
kernel = np.array([[-1, -1, -1, -1, -1],
                     [-1,  2,  2,  2, -1],
                     [-1,  2,  8,  2, -1],
                     [-1,  2,  2,  2, -1],
                     [-1, -1, -1, -1, -1]])

kernel = kernel.astype(np.float32).flatten()
kernel_size = np.int32(kernel.shape[0])

# Read the OpenCL kernel code from an external file
with open("process.cl", "r") as kernel_file:
    src = kernel_file.read()

# Kernel function instantiation
prg = cl.Program(ctx, src).build()

# Image dimensions
width, height, channels = original_img.shape[1], original_img.shape[0], original_img.shape[2]

# Allocate memory for variables on the device
original_img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=original_img)
processed_img_g = cl.Buffer(ctx, mf.WRITE_ONLY, original_img.nbytes)
kernel_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)

# Call Kernel. Automatically takes care of block/grid distribution
global_work_size = (height, width)
local_work_size = None  # Let OpenCL choose the local work size

# Start the timer
start = time.time()

# Use the correct kernel function name "applyKernel" instead of "invertImage"
evt = prg.applyKernel(queue, global_work_size, local_work_size, 
                original_img_g, processed_img_g, kernel_g,
                np.int32(width), np.int32(height), np.int32(channels), np.int32(kernel_size))

#Show the time in ms using sprinf for format
evt.wait()
second_elapsed = 1e-9 * (evt.profile.end - evt.profile.start)
print("Kernel execution time in milliseconds: %0.3f" % (second_elapsed * 1e3))
# Show TFLOPS
print("TFLOPS: %0.3f" % (1e-12 * (height * width * kernel_size * channels * 2)/second_elapsed))
processed_img = np.empty_like(original_img)

# Copy the result from the device to the host
cl.enqueue_copy(queue, processed_img, processed_img_g)

# Show the processed image
imsave("output.jpg", processed_img)
