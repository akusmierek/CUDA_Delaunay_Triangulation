#include "delaunay.h"

namespace dt
{
	template<typename T>
	const std::vector<typename Delaunay<T>::TriangleType>&
		Delaunay<T>::triangulate(std::vector<VertexType>& vertices)
	{
		// Store the vertices locally
		_vertices = vertices;

		// Determinate the super triangle
		T minX = vertices[0].x;
		T minY = vertices[0].y;
		T maxX = minX;
		T maxY = minY;

		for (std::size_t i = 0; i < vertices.size(); ++i)
		{
			if (vertices[i].x < minX) minX = vertices[i].x;
			if (vertices[i].y < minY) minY = vertices[i].y;
			if (vertices[i].x > maxX) maxX = vertices[i].x;
			if (vertices[i].y > maxY) maxY = vertices[i].y;
		}

		const T dx = maxX - minX;
		const T dy = maxY - minY;
		const T deltaMax = std::max(dx, dy);
		const T midx = (minX + maxX) / 2;
		const T midy = (minY + maxY) / 2;

		const VertexType p1(midx - 20 * deltaMax, midy - deltaMax);
		const VertexType p2(midx, midy + 20 * deltaMax);
		const VertexType p3(midx + 20 * deltaMax, midy - deltaMax);

		// Create a list of triangles, and add the supertriangle in it
		_triangles.push_back(TriangleType(p1, p2, p3));

		for (auto vertex = vertices.begin(); vertex != vertices.end(); vertex++)
		{
			std::vector<EdgeType> polygon;

			for (auto& triangle : _triangles)
			{
				if (triangle.circumCircleContains(*vertex))
				{
					triangle.isBad = true;
					polygon.push_back(Edge<T>{*triangle.a, * triangle.b});
					polygon.push_back(Edge<T>{*triangle.b, * triangle.c});
					polygon.push_back(Edge<T>{*triangle.c, * triangle.a});
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

			for (const auto& edge : polygon)
				_triangles.push_back(TriangleType(*edge.v, *edge.w, *vertex));
		}

		_triangles.erase(
			std::remove_if(_triangles.begin(), _triangles.end(),
						   [p1, p2, p3](TriangleType& t) {
							   return t.containsVertex(p1) || t.containsVertex(p2) || t.containsVertex(p3);
						   }), _triangles.end());

		for (const auto& triangle : _triangles)
		{
			_edges.push_back(Edge<T>{*triangle.a, * triangle.b});
			_edges.push_back(Edge<T>{*triangle.b, * triangle.c});
			_edges.push_back(Edge<T>{*triangle.c, * triangle.a});
		}

		return _triangles;
	}

	template<typename T>
	const std::vector<typename Delaunay<T>::TriangleType>&
		Delaunay<T>::getTriangles() const
	{
		return _triangles;
	}

	template<typename T>
	const std::vector<typename Delaunay<T>::EdgeType>&
		Delaunay<T>::getEdges() const
	{
		return _edges;
	}

	template<typename T>
	const std::vector<typename Delaunay<T>::VertexType>&
		Delaunay<T>::getVertices() const
	{
		return _vertices;
	}

	template class Delaunay<float>;
	template class Delaunay<double>;

} // namespace dt
