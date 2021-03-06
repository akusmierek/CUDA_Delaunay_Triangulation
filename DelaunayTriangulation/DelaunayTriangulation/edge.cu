#include "edge.cuh"

namespace dtc
{
	Edge::Edge(VertexType& v1, VertexType& v2, int v1_id, int v2_id) :
		v(&v1), w(&v2), v_id(v1_id), w_id(v2_id)
	{
	}

	bool
		operator ==(const float2& lhs, const float2& rhs)
	{
		return (lhs.x == rhs.x) && (lhs.y == rhs.y);
	}

	std::ostream&
		operator <<(std::ostream& str, const float2& v)
	{
		return str << "Point x: " << v.x << " y: " << v.y;
	}

	bool
		Edge::operator ==(const Edge& e) const
	{
		return this->v_id == e.v_id && this->w_id == e.w_id ||
			this->v_id == e.w_id && this->w_id == e.v_id;
		/*return (*(this->v) == *e.v && *(this->w) == *e.w) ||
			(*(this->v) == *e.w && *(this->w) == *e.v);*/
	}

	std::ostream&
		operator <<(std::ostream& str, const Edge& e)
	{
		return str << "Edge " << *e.v << ", " << *e.w;
	}

	bool almost_equal(const float x, const float y, int ulp)
	{
		return fabsf(x - y) <= std::numeric_limits<float>::epsilon() * fabsf(x + y) * static_cast<float>(ulp)
			|| fabsf(x - y) < std::numeric_limits<float>::min();
	}

	bool almost_equal(const float2& v1, const float2& v2)
	{
		return almost_equal(v1.x, v2.x) && almost_equal(v1.y, v2.y);
	}

	bool
		almost_equal(const Edge& e1, const Edge& e2)
	{
		return	(almost_equal(*e1.v, *e2.v) && almost_equal(*e1.w, *e2.w)) ||
			(almost_equal(*e1.v, *e2.w) && almost_equal(*e1.w, *e2.v));
	}
} // namespace dtc
