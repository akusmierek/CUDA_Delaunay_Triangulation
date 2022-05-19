#pragma once

#include "numeric.h"
#include "edge.cuh"

//struct Edge;

namespace dtc
{
	struct Triangle
	{
		using Type = float;
		using VertexType = float2;
		using EdgeType = Edge;

		Triangle() = default;
		Triangle(const Triangle&) = default;
		Triangle(Triangle&&) = default;
		Triangle(const VertexType& v1, const VertexType& v2, const VertexType& v3, const int v1_id, const int v2_id, const int v3_id);

		bool containsVertex(const VertexType& v) const;
		bool circumCircleContains(const VertexType& v) const;

		Triangle& operator=(const Triangle&) = default;
		Triangle& operator=(Triangle&&) = default;
		bool operator ==(const Triangle& t) const;

		friend std::ostream& operator <<(std::ostream& str, const Triangle& t);

		const VertexType* a;
		const VertexType* b;
		const VertexType* c;
		const int a_id;
		const int b_id;
		const int c_id;
		bool isBad = false;
	};

	bool almost_equal(const Triangle& t1, const Triangle& t2);
} // namespace dtc