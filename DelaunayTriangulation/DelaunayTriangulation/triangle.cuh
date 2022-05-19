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

		/*Triangle(){}
		Triangle(const Triangle&) = default;
		Triangle(Triangle&&) = default;*/
		Triangle(VertexType& v1, VertexType& v2, VertexType& v3, int v1_id, int v2_id, int v3_id);

		bool containsVertex(const VertexType& v) const;
		bool circumCircleContains(const VertexType& v) const;

		//Triangle& operator=(const Triangle&) = default;
		//Triangle& operator=(Triangle&&) = default;
		bool operator ==(const Triangle& t) const;

		friend std::ostream& operator <<(std::ostream& str, const Triangle& t);

		VertexType* a = nullptr;
		VertexType* b = nullptr;
		VertexType* c = nullptr;
		int a_id = 0;
		int b_id = 0;
		int c_id = 0;
		bool isBad = false;
	};

	bool almost_equal(const Triangle& t1, const Triangle& t2);
} // namespace dtc