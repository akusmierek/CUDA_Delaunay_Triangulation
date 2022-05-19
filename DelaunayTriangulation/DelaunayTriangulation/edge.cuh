#pragma once

#include "cuda_runtime.h"
#include <iostream>

namespace dtc
{
	struct Edge
	{
		using Type = float;
		using VertexType = float2;

		/*Edge(){}
		Edge(const Edge&) = default;
		Edge(Edge&&) = default;*/
		Edge(VertexType& v1, VertexType& v2, int v1_id, int v2_id);

		/*Edge& operator=(const Edge&){};
		Edge& operator=(Edge&&) = default;*/
		bool operator ==(const Edge& e) const;

		friend bool operator ==(const float2& lhs, const float2& rhs);
		friend std::ostream& operator <<(std::ostream& str, const float2& v);
		friend std::ostream& operator <<(std::ostream& str, const Edge& e);

		VertexType* v = nullptr;
		VertexType* w = nullptr;
		int v_id = 0;
		int w_id = 0;
		bool isBad = false;
	};

	bool almost_equal(const float x, const float y, int ulp = 2);
	bool almost_equal(const float2& v1, const float2& v2);
	bool almost_equal(const Edge& e1, const Edge& e2);
} // namespace dtc