#pragma once
#include <iostream>
#include <cassert>

template <typename _T, typename sizeType>
class stack
{
	struct node
	{
		_T data;
		node* prev;
		explicit node(const _T& tmp)
			:
			data(tmp),
			prev(0)
		{}
	};

	node* m_curr;
	sizeType m_size;

	stack(const stack&);

	stack& operator = (const stack&);

	_T m_glob;

public:
	stack()
		:
		m_curr(0),
		m_size(0),
		m_glob(_T())
	{}

	~stack()
	{
		node* tmp = 0;
		while (m_curr)
		{
			//cout << "delete " << m_curr->data << endl;
			tmp = m_curr;
			m_curr = m_curr->prev;
			delete tmp;
		}
	}

	void push(const _T& data)
	{
		++m_size;
		if (!m_curr)
		{
			m_curr = new node(data);
			return;
		}
		node* tmp = new node(data);
		tmp->prev = m_curr;
		m_curr = tmp;
	}

	void pop()
	{
		--m_size;
		assert(m_size >= 0);

		node* tmp = m_curr;
		m_curr = m_curr->prev;
		delete tmp;
	}

	_T& top()
	{
		if (!m_curr)
		{
			m_glob = _T();
			return m_glob;
		}

		return m_curr->data;
	}

	_T& before_top()
	{
		if (!m_curr)
		{
			m_glob = _T();
			return m_glob;
		}

		node* tmp = m_curr;
		tmp = tmp->prev;
		if (!tmp)
		{
			m_glob = _T();
			return m_glob;
		}

		return tmp->data;
	}

	sizeType size()  const
	{
		return m_size;
	}

	bool empty() const
	{
		return m_size == 0;
	}
};
