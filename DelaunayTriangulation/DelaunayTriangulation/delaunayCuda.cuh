#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
//#include "edge.cuh"
#include "triangle.cuh"

#include <vector>
#include <algorithm>

__global__ void reserveVerticesDevice(void);

void reserveVertices(float2* verticesToAdd, int verticesToAddNum, int* triangles, int trianglesNum, float2* allVertices, int* verticesReservations, int allVerticesNum);

namespace dtc
{
	class DelaunayCuda
	{
		using Type = float;
		using VertexType = float2;
		using EdgeType = Edge;
		using TriangleType = Triangle;

		std::vector<TriangleType> _triangles;
		std::vector<EdgeType> _edges;
		std::vector<VertexType> _vertices;

	public:

		DelaunayCuda() = default;
		DelaunayCuda(const DelaunayCuda&) = delete;
		DelaunayCuda(DelaunayCuda&&) = delete;

		const std::vector<TriangleType>& triangulate(std::vector<VertexType>& vertices);
		const std::vector<TriangleType>& getTriangles() const;
		const std::vector<EdgeType>& getEdges() const;
		const std::vector<VertexType>& getVertices() const;

		DelaunayCuda& operator=(const DelaunayCuda&) = delete;
		DelaunayCuda& operator=(DelaunayCuda&&) = delete;
	};
} // namespace dtc