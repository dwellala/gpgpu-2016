#include <sys/time.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <string.h>
#include "defs.h"
#include "stack.h"

using namespace std;

typedef std::vector<size_t> Neighbor;
typedef std::vector<Neighbor> Neighbors;	

void p(const Neighbors& neighbors)
{
	for(size_t i = 0; i < neighbors.size(); ++i)
	{
		for(size_t j = 0; j < neighbors.at(i).size(); ++j)
		{
			cout << neighbors.at(i).at(j) << " ";	
		}
		cout << endl;
	}
}

void p(_t* offset, _t offset_len, _t* detail)
{
	for(_t i = 0, index = 0; i < offset_len; ++i)
	{
		for(_t j = 0; j < offset[i]; ++j, index++)
		{
			cout << detail[index]<< " ";
		}
		cout << endl;
	}
}

void get_neighbors(_t** offset, _t* offset_len, _t** detail, size_t* detail_len, _t* N)
{
	ifstream inf("graph.txt");
	inf >> *N;
	size_t edge;
	Neighbors neighbors;
	Neighbor neighbor;
	for(_t i = 0; i < *N; ++i)
	{
		neighbor.clear();
		for(_t j = 0; j < *N; ++j)
		{
			inf >> edge;
			if(edge == 1)
			{
				neighbor.push_back(j);	
				if(i == j)
				{
					fprintf(stderr, "self loop found at node %d \n", i);
					return;
				}
			}

		}
		neighbors.push_back(neighbor);
	}
	
//	p(neighbors);
	*offset_len = neighbors.size();
	*detail_len = 0;
	(*offset) = new _t[*offset_len];

	for(size_t i = 0; i < neighbors.size(); ++i)
	{
		*detail_len += neighbors[i].size();
		(*offset)[i] = neighbors[i].size();
	}
	
	(*detail) = new _t[*detail_len];
	for(size_t i = 0, index = 0; i < neighbors.size(); ++i)
	{
		for(size_t j = 0; j < neighbors[i].size(); ++j, index++)	
		{
			(*detail)[index] = neighbors[i][j];
		}
	}
}

template <_t _S>
set<_S> nbrs(_t v, _t* offset, _t* detail)
{
	set<_S> ret;

	size_t tmp_i = 0;
	for (_t x = 0; x < v; ++x)
	{
		tmp_i += offset[x];
	}

	for (_t i = 0; i < offset[v]; ++i)
	{
		ret.insert(detail[i + tmp_i]);
	}
	return ret;
}


void bron_kerbosch_rec(set<fix> P, set<fix> R, set<fix> X, _t* offset, _t offset_len, _t* detail, _t N)
{
	if ((P.U(X)).empty())
	{
		//cout << "maximal " << R << endl;
		return;
	}

	for (_t i = 0; i < P.size(); ++i)
	{
		_t v = P[i];
		if (v == -1)
			break;
		assert(i < offset_len);

		set<fix> tmp_nbrs = nbrs<fix>(v, offset, detail);
		bron_kerbosch_rec(P.not_U(tmp_nbrs), R.U(v), X.not_U(tmp_nbrs), offset, offset_len, detail, N);
		i--;
		P.remove(v);
		X.insert(v);
	}
}

void bron_kerbosch_ite(set<fix> P, set<fix> R, set<fix> X, _t* offset, _t offset_len, _t* detail, _t N)
{
	stack<sets_t, _t> _stack;
	_stack.push(sets_t(P, R, X));
	while (!_stack.empty())
	{
		sets_t& tmp_top = _stack.top();
		if (tmp_top.P.empty())
		{
			_stack.pop();
			continue;
		}

		_t v = tmp_top.P[0];
		const set<fix>& tmp_nbrs = nbrs<fix>(v, offset, detail);

		const set<fix>& _P = tmp_top.P.not_U(tmp_nbrs);
		const set<fix>& _R = tmp_top.R.U(v);
		const set<fix>& _X = tmp_top.X.not_U(tmp_nbrs);

		//push clild set
		sets_t tmptmp(_P, _R, _X);
		_stack.push(tmptmp);

		//update 
		tmp_top.P.remove(v);
		tmp_top.X.insert(v);

		//check for cliques
		sets_t& supect_clique = _stack.top();
		if (supect_clique.P.U(supect_clique.X).empty())
		{
			cout << supect_clique.R << endl; 
			_stack.pop();
		}

		while (!_stack.empty() && _stack.top().P.empty())
		{
			_stack.pop();
		}
	}
}

void parallel(_t N, _t* offset, _t offset_len, _t* detail)
{
	for (_t v = 0; v < N; ++v)
	{
		set<fix> P, R, X;
		for (_t i = v; i < N; ++i){ P.insert(i); }
		for (_t i = 0; i < v; ++i){ X.insert(i); }

		const set<fix>& tmp_nbrs = nbrs<fix>(v, offset, detail);
		bron_kerbosch_ite(P.not_U(tmp_nbrs), R.U(v), X.not_U(tmp_nbrs), offset, offset_len, detail, N);
		//cout << "\t"<< v ;
	}
	fprintf(stderr, "done serial\n");
}

int main()
{
	timeval start, stop;

	_t *offset = 0, *detail = 0, offset_len = 0, N = 0;
	size_t detail_len = 0;

	gettimeofday(&start, 0);	
	get_neighbors(&offset, &offset_len, &detail, &detail_len, &N);

	//p(offset, offset_len, detail);

	parallel(N, offset, offset_len, detail);
	gettimeofday(&stop, 0);
	fprintf(stderr,"[serial] time = %.6f sec\n\n", (stop.tv_sec+stop.tv_usec*1e-6)-(start.tv_sec+start.tv_usec*1e-6));
	
	delete[] detail;
	delete[] offset;

	return 0;
}
