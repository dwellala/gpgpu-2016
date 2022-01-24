#pragma once
#include <iostream>
#include <cassert>

typedef short _t;
const _t fix = 9000;

template<_t _S>
class set
{
	_t m_size;
	_t m_curr;
	_t m_data[_S];
public:
	set()
		:
		m_curr(0),
		m_size(_S)
	{
		memset(m_data, -1, sizeof(_t)* m_size);
	}

	void insert(_t v)
	{
		//cout << "add " << v << endl;
		assert(m_curr >= 0 && m_curr < m_size);
		assert(v < 32767);

		for (_t i = 0; i < m_curr; ++i)
		{
			if (m_data[i] == v)
				return;
		}
		m_data[m_curr] = v;
		m_curr++;
	}

	void remove(_t v)
	{
		//cout << "remove " << v << endl;
		assert(m_curr >= 0);

		for (_t i = 0; i < m_curr; ++i)
		{
			if (m_data[i] == v)
			{
				memcpy(&m_data[i], &m_data[i + 1], sizeof(_t)* (m_size - (i + 1)));
				m_curr--;
				m_data[m_curr] = -1;
				break;
			}
		}
	}

	_t operator [] (_t index)
	{
		assert(index < m_size);
		return m_data[index];
	}

	set<_S> U(_t v)
	{
		set<_S> ret = *this;
		ret.insert(v);
		return ret;
	}

	set<_S> U(const set<_S>& other)
	{
		//assert(other.m_curr + m_curr < m_size);
		set<_S> ret = *this;
		for(_t j = 0; j < other.m_curr; ++j)
		{
			ret.insert(other.m_data[j]);
		}
		/*memcpy(&ret.m_data[ret.m_curr], other.m_data, sizeof(_t)*(other.m_curr));
		ret.m_curr += other.m_curr;
		*/
		return ret;
	};

	set<_S> not_U(const set<_S>& other)
	{
		set<_S> ret;

		for (_t i = 0; i < m_curr; ++i)
		{
			for (_t j = 0; j < other.m_curr; ++j)
			{
				if (m_data[i] == other.m_data[j])
				{
					ret.insert(m_data[i]);
				}
			}
		}
		return ret;
	}

	bool empty() const
	{
		return m_curr == 0;
	}

	_t size() const
	{
		return m_curr;
	}

		
	template <_t _St>
	friend std::ostream& operator << (std::ostream& os, const set<_St>& other);
};


template <_t _S>
std::ostream& operator << (std::ostream& os, const set<_S>& other)
{
	os << "{";
	for (_t i = 0; i < other.m_size; ++i)
	{
		if (other.m_data[i] == -1)
			continue;

		os << other.m_data[i] << ",";
	}
	os << "}";

	return os;
}

struct sets_t
{
	set<fix> P;
	set<fix> R;
	set<fix> X;

	sets_t()
	{}

	sets_t(const set<fix>& _p, const set<fix>& _r, const set<fix>& _x)
		:
		P(_p),
		R(_r),
		X(_x)
	{}
};

