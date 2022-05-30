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
	int* triangles, int* trianglesReservations, int trianglesNum,
	float2* allVertices, int* verticesReservations, int allVerticesNum)
{
	for (int i = 0; i < verticesToAddNum; i++)
	{
		int vertexToAddId = verticesToAdd[i];
		float2 vertexToAdd = allVertices[vertexToAddId];
		for (int j = 0; j < trianglesNum; j += 3)
		{
			int a_id = triangles[j];
			int b_id = triangles[j + 1];
			int c_id = triangles[j + 2];
			float2 a = allVertices[a_id];
			float2 b = allVertices[b_id];
			float2 c = allVertices[c_id];

			if (circumCircleContains(a, b, c, vertexToAdd))
			{
				atomicMax(&verticesReservations[a_id], vertexToAddId);
				atomicMax(&verticesReservations[b_id], vertexToAddId);
				atomicMax(&verticesReservations[c_id], vertexToAddId);
			}
		}
	}

	// I hope it can be done better than calculating the same thing 2 times...
	__syncthreads();

	for (int i = 0; i < verticesToAddNum; i++)
	{
		int vertexToAddId = verticesToAdd[i];
		float2 vertexToAdd = allVertices[vertexToAddId];
		for (int j = 0; j < trianglesNum; j += 3)
		{
			int a_id = triangles[j];
			int b_id = triangles[j + 1];
			int c_id = triangles[j + 2];
			float2 a = allVertices[a_id];
			float2 b = allVertices[b_id];
			float2 c = allVertices[c_id];

			if (circumCircleContains(a, b, c, vertexToAdd))
			{
				if (verticesReservations[a_id] != vertexToAddId ||
					verticesReservations[b_id] != vertexToAddId ||
					verticesReservations[c_id] != vertexToAddId)
				{
					canAdd[i] = false;
					break;
				}

				canAdd[i] = true;
				trianglesReservations[j] = vertexToAddId;
				trianglesReservations[j + 1] = vertexToAddId;
				trianglesReservations[j + 2] = vertexToAddId;
			}
		}
	}
}

void reserveVertices(
	int* verticesToAdd, bool* canAdd, int verticesToAddNum,
	int* triangles, int* trianglesReservations, int trianglesNum,
	float2* allVertices, int* verticesReservations, int allVerticesNum)
{
	int* d_verticesToAdd;
	bool* d_canAdd;
	int* d_triangles;
	int* d_trianglesReservations;
	float2* d_allVertices;
	int* d_verticesReservations;

	size_t verticesToAddSize = sizeof(int) * verticesToAddNum;
	size_t canAddSize = sizeof(bool) * verticesToAddNum;
	size_t trianglesSize = sizeof(int) * trianglesNum;
	size_t allVerticesSize = sizeof(float2) * allVerticesNum;
	size_t verticesReservationsSize = sizeof(int) * allVerticesNum;

	cudaMalloc(&d_verticesToAdd, verticesToAddSize);
	cudaMalloc(&d_canAdd, canAddSize);
	cudaMalloc(&d_triangles, trianglesSize);
	cudaMalloc(&d_trianglesReservations, trianglesSize);
	cudaMalloc(&d_allVertices, allVerticesSize);
	cudaMalloc(&d_verticesReservations, verticesReservationsSize);

	cudaMemcpy(d_verticesToAdd, verticesToAdd, verticesToAddSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangles, triangles, trianglesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_trianglesReservations, trianglesReservations, trianglesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_allVertices, allVertices, allVerticesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_verticesReservations, verticesReservations, verticesReservationsSize, cudaMemcpyHostToDevice);

	reserveVerticesDevice KERNEL_ARGS2(1, 1)(
		d_verticesToAdd, d_canAdd, verticesToAddNum,
		d_triangles, d_trianglesReservations, trianglesNum,
		d_allVertices, d_verticesReservations, allVerticesNum);

	cudaMemcpy(verticesReservations, d_verticesReservations, verticesReservationsSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(trianglesReservations, d_trianglesReservations, trianglesSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(canAdd, d_canAdd, canAddSize, cudaMemcpyDeviceToHost);

	cudaFree(d_verticesToAdd);
	cudaFree(d_canAdd);
	cudaFree(d_triangles);
	cudaFree(d_trianglesReservations);
	cudaFree(d_allVertices);
	cudaFree(d_verticesReservations);
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

		for (std::size_t i = 0; i < vertices.size(); ++i)
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
		int p1_id = vertices.size();
		int p2_id = vertices.size() + 1;
		int p3_id = vertices.size() + 2;

		// verticesToAdd contains ids of the vertices - not actual float2 position
		std::vector<int> verticesToAdd(vertices.size());
		for (int i = 0; i < vertices.size(); i++)
			verticesToAdd[i] = i;

		_vertices.push_back(p1);
		_vertices.push_back(p2);
		_vertices.push_back(p3);

		// Create a list of triangles, and add the supertriangle in it
		_triangles.push_back(TriangleType(p1, p2, p3, p1_id, p2_id, p3_id));

		std::vector<int> triangles;
		triangles.reserve(3 * _triangles.size());
		for (auto& triangle : _triangles)
		{
			triangles.push_back(triangle.a_id);
			triangles.push_back(triangle.b_id);
			triangles.push_back(triangle.c_id);
		}

		int* verticesReservations = new int[_vertices.size()];
		int* trianglesReservations = new int[triangles.size()];
		bool* canAdd;

		while (!verticesToAdd.empty())
		{
			canAdd = new bool[verticesToAdd.size()];
			int* trianglesReservations = new int[triangles.size()];

			std::fill_n(verticesReservations, _vertices.size(), -1);
			std::fill_n(trianglesReservations, triangles.size(), -1);

			reserveVertices(
				&verticesToAdd[0], canAdd, verticesToAdd.size(),
				&triangles[0], trianglesReservations, triangles.size(),
				&_vertices[0], verticesReservations, _vertices.size());

			std::vector<int> idsToRemove;
			for (int i = 0; i < verticesToAdd.size(); i++)
			{
				// has vertex access to all affected triangles?
				if (!canAdd[i])
					continue;

				idsToRemove.push_back(i);

				int vertexToAdd = verticesToAdd[i];
				std::vector<Edge> polygon;
				std::vector<int> trisToRemove;

				for (int t = 0; t < triangles.size(); t += 3)
				{
					int a = triangles[t];
					int b = triangles[t + 1];
					int c = triangles[t + 2];
					int aReservation = trianglesReservations[t];
					int bReservation = trianglesReservations[t + 1];
					int cReservation = trianglesReservations[t + 2];

					if (aReservation == bReservation && bReservation == cReservation && cReservation == vertexToAdd)
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

				for (int r = trisToRemove.size() - 1; r >= 0; r--)
					triangles.erase(triangles.begin() + trisToRemove[r]);

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

					triangles.push_back(edge.v_id);
					triangles.push_back(edge.w_id);
					triangles.push_back(vertexToAdd);
				}
			}

			for (int i = idsToRemove.size() - 1; i >= 0; i--)
				verticesToAdd.erase(verticesToAdd.begin() + idsToRemove[i]);

			delete[] canAdd;
			delete[] trianglesReservations;
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
		
		delete[] verticesReservations;

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
