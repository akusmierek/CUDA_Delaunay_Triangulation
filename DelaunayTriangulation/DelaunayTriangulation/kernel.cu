
#include <chrono>
#include <stdio.h>

#include "cuda_runtime.h"
#include "delaunay.h"
#include "delaunayCuda.cuh"
#include "device_launch_parameters.h"

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

int main()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\nDevice: %s\n", prop.name);

    std::vector<dt::Vector2<float>> points;
    std::vector<float2> cudaPoints;

    srand(static_cast<unsigned>(time(0)));

    for (size_t i = 0; i < 1000; i++)
    {
        float x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 100);
        float y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 100);
        points.push_back(dt::Vector2<float>{x, y});
        cudaPoints.push_back(make_float2(x, y));
    }

    dt::Delaunay<float> triangulation;
    
    auto t1_host = std::chrono::high_resolution_clock::now();
    const std::vector<dt::Triangle<float>> triangles = triangulation.triangulate(points);
    auto t2_host = std::chrono::high_resolution_clock::now();

    const std::vector<dt::Edge<float>> edges = triangulation.getEdges();
    /*for (const auto& e : edges)
    {
        printf("edge from (%f, %f) to (%f, %f)\n", e.v->x, e.v->y, e.w->x, e.w->y);
    }*/

    bool isTriangulationOk = true;
    for (const auto& triangle : triangles)
    {
        for (size_t i = 0; i < points.size(); i++)
        {
            auto a = make_float2(triangle.a->x, triangle.a->y);
            auto b = make_float2(triangle.b->x, triangle.b->y);
            auto c = make_float2(triangle.c->x, triangle.c->y);
            auto v = make_float2(points[i].x, points[i].y);
            if (!dtc::almost_equal(a, v) && !dtc::almost_equal(b, v) && !dtc::almost_equal(c, v))
            {
                if (circumCircleContains(a, b, c, v))
                {
                    isTriangulationOk = false;
                    break;
                }
            }
        }
    }

    std::cout << "Triangulation: " << isTriangulationOk << std::endl;

    printf("\n\n");

    dtc::DelaunayCuda triangulationCuda;
    
    auto t1_device = std::chrono::high_resolution_clock::now();
    const std::vector<dtc::Triangle> trianglesCuda = triangulationCuda.triangulate(cudaPoints);
    auto t2_device = std::chrono::high_resolution_clock::now();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    bool isTriangulationCudaOk = true;
    for (const auto& triangle : triangles)
    {
        for (size_t i = 0; i < cudaPoints.size(); i++)
        {
            auto a = make_float2(triangle.a->x, triangle.a->y);
            auto b = make_float2(triangle.b->x, triangle.b->y);
            auto c = make_float2(triangle.c->x, triangle.c->y);
            auto v = cudaPoints[i];
            if (!dtc::almost_equal(a, v) && !dtc::almost_equal(b, v) && !dtc::almost_equal(c, v))
            {
                if (circumCircleContains(a, b, c, v))
                {
                    isTriangulationCudaOk = false;
                    break;
                }
            }
        }
    }

    std::cout << "Triangulation: " << isTriangulationCudaOk << std::endl;

    const std::vector<dtc::Edge> edgesCuda = triangulationCuda.getEdges();
    /*for (const auto& e : edgesCuda)
    {
        printf("edge from (%f, %f) to (%f, %f)\n", e.v->x, e.v->y, e.w->x, e.w->y);
    }*/

    std::cout << "Host execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_host - t1_host).count() << " ms" << std::endl;
    std::cout << "Device execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_device - t1_device).count() << " ms" << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}