#include "delaunayCuda.cuh"
#include <thrust\device_vector.h>

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

__host__ __device__ bool circumCircleContains(const float2 a, const float2 b, const float2 c, const float2 v)
{
	const float ab = a.x * a.x + a.y * a.y;
	const float cd = b.x * b.x + b.y * b.y;
	const float ef = c.x * c.x + c.y * c.y;

	const float ax = a.x;
	const float ay = a.y;
	const float bx = b.x;
	const float by = b.y;
	const float cx = c.x;
	const float cy = c.y;

	const float circum_x = (ab * (cy - by) + cd * (ay - cy) + ef * (by - ay)) / (ax * (cy - by) + bx * (ay - cy) + cx * (by - ay));
	const float circum_y = (ab * (cx - bx) + cd * (ax - cx) + ef * (bx - ax)) / (ay * (cx - bx) + by * (ax - cx) + cy * (bx - ax));

	const float2 circum = make_float2(circum_x / 2, circum_y / 2);

	const float dx = a.x - circum.x;
	const float dy = a.y - circum.y;
	const float circum_radius = dx * dx + dy * dy;

	const float dx2 = v.x - circum.x;
	const float dy2 = v.y - circum.y;
	const float dist = dx2 * dx2 + dy2 * dy2;
	return dist <= circum_radius;
}

__global__ void reserveVerticesDevice(
	int* verticesToAdd, bool* canAdd, int verticesToAddNum,
	int* trianglesVertices, int* trianglesReservations, int trianglesVerticesNum,
	float2* allVertices, int allVerticesNum)
{
	int i = threadIdx.x;
	int trianglesNum = trianglesVerticesNum / 3;

	bool* reservations = (bool*)malloc(trianglesNum * sizeof(bool));
	memset(reservations, 0, trianglesNum * sizeof(bool));

	int vertexToAddId = verticesToAdd[i];
	float2 vertexToAdd = allVertices[vertexToAddId];
	for (int j = 0; j < trianglesVerticesNum; j += 3)
	{
		int a_id = trianglesVertices[j];
		int b_id = trianglesVertices[j + 1];
		int c_id = trianglesVertices[j + 2];
		float2 a = allVertices[a_id];
		float2 b = allVertices[b_id];
		float2 c = allVertices[c_id];

		if (circumCircleContains(a, b, c, vertexToAdd))
		{
			int triangleNum = j / 3;
			atomicMax(&trianglesReservations[triangleNum], vertexToAddId);
			reservations[triangleNum] = true;
		}
	}

	__syncthreads();

	for (int j = 0; j < trianglesNum; j++)
	{
		if (reservations[j] && trianglesReservations[j] != vertexToAddId)
		{
			canAdd[i] = false;
			break;
		}
	}

	free(reservations);
}

void reserveVertices(
	int* verticesToAdd, bool* canAdd, size_t verticesToAddNum,
	int* trianglesVertices, int* trianglesReservations, size_t trianglesVerticesNum,
	float2* d_allVertices, size_t allVerticesNum)
{
	int* d_verticesToAdd;
	bool* d_canAdd;
	int* d_trianglesVertices;
	int* d_trianglesReservations;

	size_t trianglesNum = trianglesVerticesNum / 3;

	size_t verticesToAddSize = sizeof(int) * verticesToAddNum;
	size_t canAddSize = sizeof(bool) * verticesToAddNum;
	size_t trianglesVerticesSize = sizeof(int) * trianglesVerticesNum;
	size_t trianglesSize = sizeof(int) * trianglesNum;

	cudaMalloc(&d_verticesToAdd, verticesToAddSize);

	cudaMalloc(&d_canAdd, canAddSize);
	cudaMemset(d_canAdd, 1, canAddSize);

	cudaMalloc(&d_trianglesVertices, trianglesVerticesSize);

	cudaMalloc(&d_trianglesReservations, trianglesSize);
	cudaMemset(d_trianglesReservations, -1, trianglesSize);

	cudaMemcpy(d_verticesToAdd, verticesToAdd, verticesToAddSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_trianglesVertices, trianglesVertices, trianglesVerticesSize, cudaMemcpyHostToDevice);

	reserveVerticesDevice KERNEL_ARGS2(1, static_cast<int>(verticesToAddNum))(
		d_verticesToAdd, d_canAdd, static_cast<int>(verticesToAddNum),
		d_trianglesVertices, d_trianglesReservations, static_cast<int>(trianglesVerticesNum),
		d_allVertices, static_cast<int>(allVerticesNum));


	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaDeviceSynchronize();
	cudaEventRecord(startEvent, 0);
	cudaMemcpy(trianglesReservations, d_trianglesReservations, trianglesSize, cudaMemcpyDeviceToHost);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	float time;
	cudaEventElapsedTime(&time, startEvent, stopEvent);
	std::cout << "Device to Host bandwidth (GB/s): " << trianglesSize * 1e-6 / time << " (" << trianglesSize * 1e-6 << "/" << time << ")" << std::endl;

	cudaMemcpy(canAdd, d_canAdd, canAddSize, cudaMemcpyDeviceToHost);

	cudaFree(d_verticesToAdd);
	cudaFree(d_canAdd);
	cudaFree(d_trianglesVertices);
	cudaFree(d_trianglesReservations);
}

namespace dtc
{
	const std::vector<Triangle>&
		DelaunayCuda::triangulate(std::vector<VertexType>& vertices)
	{
		// Store the vertices locally
		_vertices = vertices;

		// Determinate the super triangle
		float minX = vertices[0].x;
		float minY = vertices[0].y;
		float maxX = minX;
		float maxY = minY;

		for (int i = 0; i < static_cast<int>(vertices.size()); ++i)
		{
			if (vertices[i].x < minX) minX = vertices[i].x;
			if (vertices[i].y < minY) minY = vertices[i].y;
			if (vertices[i].x > maxX) maxX = vertices[i].x;
			if (vertices[i].y > maxY) maxY = vertices[i].y;
		}

		const float dx = maxX - minX;
		const float dy = maxY - minY;
		const float deltaMax = std::max(dx, dy);
		const float midx = (minX + maxX) / 2;
		const float midy = (minY + maxY) / 2;

		VertexType p1 = make_float2(midx - 20 * deltaMax, midy - deltaMax);
		VertexType p2 = make_float2(midx, midy + 20 * deltaMax);
		VertexType p3 = make_float2(midx + 20 * deltaMax, midy - deltaMax);

		// 3 last vertices are super triangle
		int p1_id = static_cast<int>(vertices.size());
		int p2_id = static_cast<int>(vertices.size() + 1);
		int p3_id = static_cast<int>(vertices.size() + 2);

		// verticesToAdd contains ids of the vertices - not actual float2 position
		std::vector<int> verticesToAdd(vertices.size());
		for (int i = 0; i < static_cast<int>(vertices.size()); i++)
			verticesToAdd[i] = i;

		_vertices.push_back(p1);
		_vertices.push_back(p2);
		_vertices.push_back(p3);

		// Create a list of triangles, and add the supertriangle in it
		_triangles.push_back(TriangleType(p1, p2, p3, p1_id, p2_id, p3_id));

		std::vector<int> trianglesVertices;
		trianglesVertices.reserve(3 * _triangles.size());
		for (auto& triangle : _triangles)
		{
			trianglesVertices.push_back(triangle.a_id);
			trianglesVertices.push_back(triangle.b_id);
			trianglesVertices.push_back(triangle.c_id);
		}

		int* trianglesReservations;
		bool* canAdd;
		float2* d_allVertices;
		size_t allVerticesSize = sizeof(float2) * _vertices.size();
		cudaMalloc(&d_allVertices, allVerticesSize);
		cudaMemcpy(d_allVertices, _vertices.data(), allVerticesSize, cudaMemcpyHostToDevice);

		while (!verticesToAdd.empty())
		{
			canAdd = new bool[verticesToAdd.size()];
			size_t trianglesReservationsSize = trianglesVertices.size() / 3;
			//trianglesReservations = (int*)malloc(trianglesReservationsSize * sizeof(int));
			cudaMallocHost((void**)&trianglesReservations, trianglesReservationsSize * sizeof(int));

			reserveVertices(
				verticesToAdd.data(), canAdd, verticesToAdd.size(),
				trianglesVertices.data(), trianglesReservations, trianglesVertices.size(),
				d_allVertices, _vertices.size());

			std::vector<int> idsToRemove;
			for (int i = 0; i < static_cast<int>(verticesToAdd.size()); i++)
			{
				// has vertex access to all affected triangles?
				if (!canAdd[i])
					continue;

				idsToRemove.push_back(i);

				int vertexToAdd = verticesToAdd[i];
				std::vector<Edge> polygon;
				std::vector<int> trisToRemove;

				for (int t = 0; t < trianglesVertices.size() && (t / 3) < trianglesReservationsSize; t += 3)
				{
					int a = trianglesVertices[t];
					int b = trianglesVertices[t + 1];
					int c = trianglesVertices[t + 2];
					int reservation = trianglesReservations[t / 3];

					if (reservation == vertexToAdd)
					{
						_triangles[t / 3].isBad = true;

						trisToRemove.push_back(t);
						trisToRemove.push_back(t + 1);
						trisToRemove.push_back(t + 2);
						polygon.push_back(Edge{_vertices[a], _vertices[b], a, b});
						polygon.push_back(Edge{_vertices[b], _vertices[c], b, c});
						polygon.push_back(Edge{_vertices[c], _vertices[a], c, a});
					}
				}

				for (int r = static_cast<int>(trisToRemove.size()) - 1; r >= 0; r--)
					trianglesVertices.erase(trianglesVertices.begin() + trisToRemove[r]);

				_triangles.erase(
					std::remove_if(_triangles.begin(), _triangles.end(),
								   [](TriangleType& t) {
									   return t.isBad;
								   }), _triangles.end());

				for (auto e1 = polygon.begin(); e1 != polygon.end(); ++e1)
				{
					for (auto e2 = e1 + 1; e2 != polygon.end(); ++e2)
					{
						if (almost_equal(*e1, *e2))
						{
							e1->isBad = true;
							e2->isBad = true;
						}
					}
				}

				polygon.erase(
					std::remove_if(polygon.begin(), polygon.end(),
								   [](Edge& e) {
									   return e.isBad;
								   }), end(polygon));

				for (const auto edge : polygon)
				{
					_triangles.push_back(Triangle(*edge.v, *edge.w, _vertices[vertexToAdd], edge.v_id, edge.w_id, vertexToAdd));

					trianglesVertices.push_back(edge.v_id);
					trianglesVertices.push_back(edge.w_id);
					trianglesVertices.push_back(vertexToAdd);
				}
			}

			for (int i = static_cast<int>(idsToRemove.size()) - 1; i >= 0; i--)
				verticesToAdd.erase(verticesToAdd.begin() + idsToRemove[i]);

			delete[] canAdd;
			cudaFreeHost(trianglesReservations);
			//delete[] trianglesReservations;
		}

		_triangles.erase(
			std::remove_if(_triangles.begin(), _triangles.end(),
							[p1, p2, p3](Triangle& t) {
								return t.containsVertex(p1) || t.containsVertex(p2) || t.containsVertex(p3);
							}), _triangles.end());

		for (const auto triangle : _triangles)
		{
			_edges.push_back(Edge(*triangle.a, * triangle.b, triangle.a_id, triangle.b_id));
			_edges.push_back(Edge(*triangle.b, * triangle.c, triangle.b_id, triangle.c_id));
			_edges.push_back(Edge(*triangle.c, * triangle.a, triangle.c_id, triangle.a_id));
		}

		cudaFree(d_allVertices);

		return _triangles;
	}

	const std::vector<Triangle>&
		DelaunayCuda::getTriangles() const
	{
		return _triangles;
	}

	const std::vector<Edge>&
		DelaunayCuda::getEdges() const
	{
		return _edges;
	}

	const std::vector<float2>&
		DelaunayCuda::getVertices() const
	{
		return _vertices;
	}
} // namespace dtc
