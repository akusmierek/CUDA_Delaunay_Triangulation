
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "delaunay.h"

#include <stdio.h>
#include "delaunayCuda.cuh"

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t triangulateWithCuda();

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}


    ///////////////////
    // TRIANGULATION //
    ///////////////////
    std::vector<dt::Vector2<float>> points;
    points.push_back(dt::Vector2<float>{0, 2});
    points.push_back(dt::Vector2<float>{1, 0});
    points.push_back(dt::Vector2<float>{0, -2});
    points.push_back(dt::Vector2<float>{-10, 0});
    points.push_back(dt::Vector2<float>{2, 2});

    dt::Delaunay<float> triangulation;
    const std::vector<dt::Triangle<float>> triangles = triangulation.triangulate(points);
    const std::vector<dt::Edge<float>> edges = triangulation.getEdges();
    for (const auto& e : edges)
    {
        printf("edge from (%f, %f) to (%f, %f)\n", e.v->x, e.v->y, e.w->x, e.w->y);
    }

    printf("\n\n");

    std::vector<float2> cudaPoints;
    cudaPoints.push_back(make_float2(0, 2));
    cudaPoints.push_back(make_float2(1, 0));
    cudaPoints.push_back(make_float2(0, -2));
    cudaPoints.push_back(make_float2(-10, 0));
    cudaPoints.push_back(make_float2(2, 2));

    dtc::DelaunayCuda triangulationCuda;
    const std::vector<dtc::Triangle> trianglesCuda = triangulationCuda.triangulate(cudaPoints);
    const std::vector<dtc::Edge> edgesCuda = triangulationCuda.getEdges();
    for (const auto& e : edgesCuda)
    {
        printf("edge from (%f, %f) to (%f, %f)\n", e.v->x, e.v->y, e.w->x, e.w->y);
    }

    cudaError_t triangulationCudaStatus = triangulateWithCuda();
    if (triangulationCudaStatus != cudaSuccess)
    {
        fprintf(stderr, "triangulationWithCuda failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel KERNEL_ARGS2(1, size) (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t triangulateWithCuda()
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    return cudaStatus;
}
