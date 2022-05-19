#include "triangle.cuh"

namespace dtc
{
	Triangle::Triangle(const VertexType& v1, const VertexType& v2, const VertexType& v3, const int v1_id, const int v2_id, const int v3_id) :
		a(&v1), b(&v2), c(&v3), a_id(v1_id), b_id(v2_id), c_id(v3_id), isBad(false)
	{
	}

	bool
		Triangle::containsVertex(const VertexType& v) const
	{
		// return p1 == v || p2 == v || p3 == v;
		return almost_equal(*a, v) || almost_equal(*b, v) || almost_equal(*c, v);
	}

	bool
		Triangle::circumCircleContains(const VertexType& v) const
	{
		const float ab = a->x * a->x + a->y * a->y;
		const float cd = b->x * b->x + b->y * b->y;
		const float ef = c->x * c->x + c->y * c->y;

		const float ax = a->x;
		const float ay = a->y;
		const float bx = b->x;
		const float by = b->y;
		const float cx = c->x;
		const float cy = c->y;

		const float circum_x = (ab * (cy - by) + cd * (ay - cy) + ef * (by - ay)) / (ax * (cy - by) + bx * (ay - cy) + cx * (by - ay));
		const float circum_y = (ab * (cx - bx) + cd * (ax - cx) + ef * (bx - ax)) / (ay * (cx - bx) + by * (ax - cx) + cy * (bx - ax));

		const VertexType circum = make_float2(circum_x / 2, circum_y / 2);

		const float dx = a->x - circum.x;
		const float dy = a->y - circum.y;
		const float circum_radius = dx * dx + dy * dy;

		const float dx2 = v.x - circum.x;
		const float dy2 = v.y - circum.y;
		const float dist = dx2 * dx2 + dy2 * dy2;
		return dist <= circum_radius;
	}

	bool
		Triangle::operator ==(const Triangle& t) const
	{
		return	(*this->a == *t.a || *this->a == *t.b || *this->a == *t.c) &&
			(*this->b == *t.a || *this->b == *t.b || *this->b == *t.c) &&
			(*this->c == *t.a || *this->c == *t.b || *this->c == *t.c);
	}

	std::ostream&
		operator <<(std::ostream& str, const Triangle& t)
	{
		return str << "Triangle:" << "\n\t" <<
			*t.a << "\n\t" <<
			*t.b << "\n\t" <<
			*t.c << '\n';
	}

	bool almost_equal(const Triangle& t1, const Triangle& t2)
	{
		return	(almost_equal(*t1.a, *t2.a) || almost_equal(*t1.a, *t2.b) || almost_equal(*t1.a, *t2.c)) &&
			(almost_equal(*t1.b, *t2.a) || almost_equal(*t1.b, *t2.b) || almost_equal(*t1.b, *t2.c)) &&
			(almost_equal(*t1.c, *t2.a) || almost_equal(*t1.c, *t2.b) || almost_equal(*t1.c, *t2.c));
	}
} // namespace dtc
