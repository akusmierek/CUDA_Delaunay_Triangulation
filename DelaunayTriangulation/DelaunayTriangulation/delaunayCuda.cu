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

__device__ bool circumCircleContains(const float2 a, const float2 b, const float2 c, const float2 v)
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

__global__ void reserveVerticesDevice(float2* verticesToAdd, int verticesToAddNum, int* triangles, int trianglesNum, float2* allVertices, int* verticesReservations, int allVerticesNum)
{
	for (int i = 0; i < verticesToAddNum; i++)
	{
		for (int j = 0; j < trianglesNum; j += 3)
		{
			int a_id = triangles[j];
			int b_id = triangles[j + 1];
			int c_id = triangles[j + 2];
			float2 a = allVertices[a_id];
			float2 b = allVertices[b_id];
			float2 c = allVertices[c_id];

			if (circumCircleContains(a, b, c, verticesToAdd[i]))
			{
				atomicMax(&verticesReservations[a_id], i);
				atomicMax(&verticesReservations[b_id], i);
				atomicMax(&verticesReservations[c_id], i);
			}
		}
	}
}

void reserveVertices(float2* verticesToAdd, int verticesToAddNum, int* triangles, int trianglesNum, float2* allVertices, int* verticesReservations, int allVerticesNum)
{
	// Print data for tests

	/*printf("Vertices to add:\n");
	for (int i = 0; i < verticesToAddNum; i++)
		printf("(%f, %f)\n", verticesToAdd[i].x, verticesToAdd[i].y);

	printf("Triangles:\n");
	for (int i = 0; i < trianglesNum; i += 3)
		printf("(%d, %d, %d)\n", triangles[i], triangles[i + 1], triangles[i + 2]);

	printf("All vertices:\n");
	for (int i = 0; i < allVerticesNum; i++)
		printf("(%f, %f)\n", allVertices[i].x, allVertices[i].y);*/

	float2* d_verticesToAdd;
	int* d_triangles;
	float2* d_allVertices;
	int* d_verticesReservations;

	size_t verticesToAddSize = sizeof(float2) * verticesToAddNum;
	size_t trianglesSize = sizeof(int) * trianglesNum;
	size_t allVerticesSize = sizeof(float2) * allVerticesNum;
	size_t verticesReservationsSize = sizeof(int) * allVerticesNum;

	cudaMalloc(&d_verticesToAdd, verticesToAddSize);
	cudaMalloc(&d_triangles, trianglesSize);
	cudaMalloc(&d_allVertices, allVerticesSize);
	cudaMalloc(&d_verticesReservations, verticesReservationsSize);

	cudaMemcpy(d_verticesToAdd, verticesToAdd, verticesToAddSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangles, triangles, trianglesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_allVertices, allVertices, allVerticesSize, cudaMemcpyHostToDevice);

	reserveVerticesDevice KERNEL_ARGS2(1, 1)(d_verticesToAdd, verticesToAddNum, d_triangles, trianglesNum, d_allVertices, d_verticesReservations, allVerticesNum);

	cudaMemcpy(verticesReservations, d_verticesReservations, verticesReservationsSize, cudaMemcpyDeviceToHost);

	printf("Vertices reservations:\n");
	for (int i = 0; i < allVerticesNum; i++)
		printf("%d: %d\n", i, verticesReservations[i]);
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

		const VertexType p1 = make_float2(midx - 20 * deltaMax, midy - deltaMax);
		const VertexType p2 = make_float2(midx, midy + 20 * deltaMax);
		const VertexType p3 = make_float2(midx + 20 * deltaMax, midy - deltaMax);

		// 3 last vertices are super triangle
		int p1_id = vertices.size();
		int p2_id = vertices.size() + 1;
		int p3_id = vertices.size() + 2;

		auto verticesToAdd = vertices;

		_vertices.push_back(p1);
		_vertices.push_back(p2);
		_vertices.push_back(p3);

		// Create a list of triangles, and add the supertriangle in it
		_triangles.push_back(TriangleType(p1, p2, p3, p1_id, p2_id, p3_id));

		std::vector<int> triangles;
		for (auto& triangle : _triangles)
		{
			triangles.push_back(triangle.a_id);
			triangles.push_back(triangle.b_id);
			triangles.push_back(triangle.c_id);
		}

		int* verticesReservations = new int[_vertices.size()];

		reserveVertices(&verticesToAdd[0], verticesToAdd.size(), &triangles[0], triangles.size(), &_vertices[0], verticesReservations, _vertices.size());

		/*for (auto vertex = vertices.begin(); vertex != vertices.end(); vertex++)
		{
			std::vector<EdgeType> polygon;

			for (auto& triangle : _triangles)
			{
				if (triangle.circumCircleContains(*vertex))
				{
					triangle.isBad = true;
					polygon.push_back(Edge<float>{*triangle.a, * triangle.b});
					polygon.push_back(Edge<float>{*triangle.b, * triangle.c});
					polygon.push_back(Edge<float>{*triangle.c, * triangle.a});
				}
			}

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
							   [](EdgeType& e) {
								   return e.isBad;
							   }), end(polygon));

			for (const auto edge : polygon)
				_triangles.push_back(TriangleType(*edge.v, *edge.w, *vertex));
		}

		_triangles.erase(
			std::remove_if(_triangles.begin(), _triangles.end(),
						   [p1, p2, p3](TriangleType& t) {
							   return t.containsVertex(p1) || t.containsVertex(p2) || t.containsVertex(p3);
						   }), _triangles.end());

		for (const auto triangle : _triangles)
		{
			_edges.push_back(Edge<float>{*triangle.a, * triangle.b});
			_edges.push_back(Edge<float>{*triangle.b, * triangle.c});
			_edges.push_back(Edge<float>{*triangle.c, * triangle.a});
		}*/

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
