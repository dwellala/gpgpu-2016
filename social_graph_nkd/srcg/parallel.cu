#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "defs.h"
#include "print.h"
#include <unistd.h>
#include <iomanip>

using namespace std;

class set_t
{
	_t m_size;
	_t m_curr;
	_t* m_data;
	typedef _t value_type;
public:
	__device__ 
	explicit set_t(_t N)
		:
		m_curr(0),
		m_size(N),
		m_data(new _t[m_size])
	{
		memset(m_data, -1, sizeof(_t)* m_size);
	}

	__device__ 
	set_t(const set_t& other)
		:
		m_size(other.m_size),
		m_curr(other.m_curr),
		m_data(new _t[m_size])
	{
		memcpy(m_data, other.m_data, sizeof(_t) * m_size);
	}

	__device__ 
	set_t& operator= (const set_t& other)
	{
		if(m_data == other.m_data)
			return *this;
		m_curr = other.m_curr;
		m_size = other.m_size;
		memcpy(m_data, other.m_data, sizeof(_t) * m_size);
		return *this;
	}

	__device__ 
	~set_t()
	{
		delete[] m_data;
	}

	__device__ 
	void insert(_t v)
	{
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

	__device__ 
	void remove(_t v)
	{
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

	__device__ 
	_t operator [] (_t index)
	{
		assert(index < m_size);
		return m_data[index];
	}

	__device__ 
	set_t U(_t v)
	{
		set_t ret = *this;
		ret.insert(v);
		return ret;
	}

	__device__ 
	set_t U(const set_t& other)
	{
		set_t ret = *this;
		for(_t j = 0; j < other.m_curr; ++j)
		{
			ret.insert(other.m_data[j]);
		}
		return ret;
	}

	__device__ 
	set_t not_U(const set_t& other)
	{
		set_t ret(m_size);
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

	__device__ 
	bool empty() const
	{
		return m_curr == 0;
	}

	__device__ 
	_t size() const
	{
		return m_curr;
	}

	__device__
	const _t* data() const
	{
		return m_data;
	}
	friend std::ostream& operator << (std::ostream& os, const set_t& other);
};

struct sets_t
{
	set_t* P;
	set_t* R;
	set_t* X;
	typedef set_t value_type;
	__device__
	sets_t(const set_t& P, const set_t& R, const set_t& X)
		:
		P(new set_t(P)),
		R(new set_t(R)),
		X(new set_t(X))
	{}

	__device__
	sets_t(const sets_t& other)
		:
		P(new set_t(*other.P)),
		R(new set_t(*other.R)),
		X(new set_t(*other.X))
	{}

	__device__
	~sets_t()
	{
		delete P;
		delete R;
		delete X;
	}
private:
	__device__
	sets_t& operator= (const sets_t&);
	
};

template <typename _T, typename sizeType>
class stack_t
{
	struct node
	{
		_T data;
		node* prev;
		__device__
		explicit node(const _T& tmp)
			:
			data(tmp),
			prev(0)
		{}
	};

	node* m_curr;
	sizeType m_size;

	__device__
	stack_t(const stack_t&);

	__device__
	stack_t& operator = (const stack_t&);

	_T m_glob;
public:
	__device__
	explicit stack_t(size_t N)
		:
		m_curr(0),
		m_size(0),
		m_glob(_T(typename _T::value_type(N), typename _T::value_type(N), typename _T::value_type(N)))
	{}

	__device__
	~stack_t()
	{
		node* tmp = 0;
		while (m_curr)
		{
			tmp = m_curr;
			m_curr = m_curr->prev;
			delete tmp;
		}
	}

	__device__
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

	__device__
	void pop()
	{
		--m_size;
		assert(m_size >= 0);

		node* tmp = m_curr;
		m_curr = m_curr->prev;
		delete tmp;
	}

	__device__
	_T& top()
	{
		if (!m_curr)
		{
			return m_glob;
		}
		return m_curr->data;
	}

	__device__
	_T& before_top()
	{
		if (!m_curr)
		{
			return m_glob;
		}

		node* tmp = m_curr;
		tmp = tmp->prev;
		if (!tmp)
		{
			return m_glob;
		}
		return tmp->data;
	}

	__device__
	sizeType size()  const
	{
		return m_size;
	}

	__device__
	bool empty() const
	{
		return m_size == 0;
	}
};

__device__ 
set_t nbrs(_t v, _t* offset, _t* detail, _t N)
{
	set_t ret(N);

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

__device__
void bron_kerbosch_ite(const set_t& P, const set_t& R, const set_t& X, _t* offset, _t offset_len, _t* detail, _t N)
{
	stack_t<sets_t, _t> _stack(N);
	_stack.push(sets_t(P, R, X));
	while (!_stack.empty())
	{	
		//printf("stack size %d\n", _stack.size());
		sets_t& tmp_top = _stack.top();
		if (tmp_top.P->empty())
		{
			_stack.pop();
			continue;
		}

		_t v = (*(tmp_top.P))[0];
		const set_t& tmp_nbrs = nbrs(v, offset, detail, N);

		const set_t& _P = tmp_top.P->not_U(tmp_nbrs);
		const set_t& _R = tmp_top.R->U(v);
		const set_t& _X = tmp_top.X->not_U(tmp_nbrs);

		//push clild set
		sets_t tmptmp(_P, _R, _X);
		_stack.push(tmptmp);

		//update 
		tmp_top.P->remove(v);
		tmp_top.X->insert(v);

		//check for cliques
		sets_t& suspect_clique = _stack.top();
		if (suspect_clique.P->U(*suspect_clique.X).empty())
		{
			print(suspect_clique.R->data(), suspect_clique.R->size());
			_stack.pop();
		}

		while (!_stack.empty() && _stack.top().P->empty())
		{
			_stack.pop();
		}
	}
}

__global__
void parallel(_t N, _t* offset, _t offset_len, _t* detail, _t* tmp_ptr, int gpu, int num_tasks)
{
	const size_t v = blockIdx.x * blockDim.x + threadIdx.x;

	if(v >= N) return; 
	
	tmp_ptr[v] = -1;

	if((v % num_tasks) != gpu)
		return;

	set_t P(N), R(N), X(N);
	for (_t i = v; i < N; ++i){ P.insert(i); }
	for (_t i = 0; i < v; ++i){ X.insert(i); }
	
	const set_t& tmp_nbrs = nbrs(v, offset, detail, N);	
	/*
	if(tmp_nbrs.empty())
	{
		tmp_ptr[v] = v;
		return;
	}
	*/
	bron_kerbosch_ite(P.not_U(tmp_nbrs), R.U(v), X.not_U(tmp_nbrs), offset, offset_len, detail, N);

	tmp_ptr[v] = v;
}

extern "C" void get_data_sim(const char* filename, _t** offset, _t* offset_len, _t**detail, size_t* detail_len, _t* N)
{
	Neighbors nrbrs = File<adj_matrix>::read(filename);
	*N = nrbrs.size();
	fill_nrbrs(nrbrs, offset, offset_len, detail, detail_len, N);
}

extern "C" void get_data(const char* filename, _t** offset, _t* offset_len, _t**detail, size_t* detail_len, _t* N)
{
	Neighbors nrbrs = File<adj_matrix>::read(filename);
	*N = nrbrs.size();

	// Create a star graph
	Graph graph, dup_graph;
	{
		std::vector<Vertex> v_map;
		//create graphs
		for (size_t i = 0; i < nrbrs.size(); ++i)
		{
			v_map.push_back(boost::add_vertex(graph));
			p_bar(i, nrbrs.size(), " adding vertices ...");
		}
		fprintf(stderr, "\n");

		// make edges
		const size_t nrb_size = nrbrs.size();
		for (size_t i = 0; i < nrb_size; ++i)
		{
			for (size_t j = 0; j < nrbrs[i].size(); ++j)
			{
				if (nrbrs[i][j] < nrb_size)// filter for invalid edges
				{
					boost::add_edge(v_map[i], v_map[nrbrs[i][j]], graph);
				}
			}
			p_bar(i, nrb_size, " adding edges ...");
		}
		fprintf(stderr, "\n");
	}

	//print_graph(graph);
	
	timeval start, stop;
	gettimeofday(&start, 0);
	find_bc(graph, *N);
	gettimeofday(&stop, 0);
	fprintf(stderr,"## betweenes-cent distribution %.6f sec\n\n", (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6));		

	double min_b_c = 0.0;
	fprintf(stderr, "t : ");
	std::cin >> min_b_c;
	Neighbors updated_nrbrs;
	updated_nrbrs.resize(nrbrs.size());

	gettimeofday(&start, 0);
	update_graph(updated_nrbrs, graph, min_b_c, *N);
	gettimeofday(&stop, 0);
	fprintf(stderr,"## get updated graph %.6f sec\n\n", (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6));		

	fill_nrbrs(updated_nrbrs, offset, offset_len, detail, detail_len, N);

//	print_graph(graph);
/*
	std::cout << "_____" << std::endl;
	for(_t i = 0; i < updated_nrbrs.size(); ++i)
	{
		for(_t j = 0; j < updated_nrbrs[i].size(); ++j)
		{
			std::cout << updated_nrbrs[i][j]<<" ";
		}
		std::cout << std::endl;
	}	

	std::cout << "_____" << std::endl;
*/
}

extern "C" void run_on_gpu(_t* offset, _t offset_len, _t* detail, size_t detail_len, _t N, int taskId, int num_tasks)
{
	size_t mem_size;
	{
		mem_size = 10*1024;
		cudaError_t e1 = cudaThreadSetLimit(cudaLimitStackSize, mem_size);
		fprintf(stderr, "[gpu %d] stack requested %G KB\n", taskId, (double)mem_size/1024);
		//cudaError(e1); 
		if(e1 != cudaSuccess)
		{
			cudaThreadGetLimit(&mem_size, cudaLimitStackSize);
			fprintf(stderr, "[gpu %d -error] stack allocation : available %G KB\n", taskId, (double)mem_size/1024);
		}
	}
	{	
		mem_size = 1024*1024*1024;
		fprintf(stderr, "[gpu %d] heap requested %G MB\n", taskId, (double)mem_size/1024/1024); 
		cudaError_t e2 = cudaThreadSetLimit(cudaLimitMallocHeapSize, mem_size);
		//cudaError(e2);
		if(e2  != cudaSuccess)
		{
			cudaThreadGetLimit(&mem_size, cudaLimitMallocHeapSize);
			fprintf(stderr, "[gpu %d -error] heap allocation : available %G KB\n", taskId, (double)mem_size/1024);
		}
	}
	{
		mem_size = 1024*1024*1024;
		fprintf(stderr, "[gpu %d] printf mem requested %G MB\n", taskId, (double)mem_size/1024/1024);
		cudaError_t e3 = cudaThreadSetLimit(cudaLimitPrintfFifoSize,  mem_size); 
		//cudaError(e3);
		if(e3 != cudaSuccess)
		{		
			cudaThreadGetLimit(&mem_size, cudaLimitPrintfFifoSize);
			fprintf(stderr, "[gpu %d -error] printf mem allocation : available %G KB\n", taskId, (double)mem_size/1024);
		}
	}

	{
		_t *dev_offset = 0, *dev_detail = 0;
		cuda_mem<_t> offsetDeviceMem(&dev_offset, offset_len);
		cuda_mem<_t> detailDeviceMem(&dev_detail, detail_len); 

		offsetDeviceMem.cpyToDevice(offset);
		detailDeviceMem.cpyToDevice(detail);

		size_t elements = ((N * 3) * N * (N + 1) / 2);
		{
			fprintf(stderr, " device heap allocations on GPU %d\n", taskId);
			fprintf(stderr, "[device_mem] offset %G KB\n", (double)offset_len * sizeof(_t)/1024);
			fprintf(stderr, "[device_mem] detail %G KB\n", (double)detail_len * sizeof(_t)/1024);
			//cout << "elements " << elements << endl;
			fprintf(stderr, "[device_mem] approximate total heap allocation %G MB\n", (double)elements * sizeof(_t)/1024/1024);
			fprintf(stderr, "[device_mem] tids %G KB\n", (double) N * sizeof(_t)/1024);
		}
	
		_t* dev_tids, *tids = new _t[N];
		cuda_mem<_t> tidsDeviceMem(&dev_tids, N);
		
		//parallel<<<10, 128>>>(N, dev_offset, offset_len, dev_detail, dev_tids, taskId, num_tasks);
		//parallel<<<10, 512>>>(N, dev_offset, offset_len, dev_detail, dev_tids, taskId, num_tasks);
		parallel<<<20, 512>>>(N, dev_offset, offset_len, dev_detail, dev_tids, taskId, num_tasks);


		tidsDeviceMem.cpyFromDevice(&tids);

		fprintf(stderr, "cuda threads on GPU %d\t", taskId);
		for(_t i = 0; i < N; ++i)
		{
			if(tids[i] != -1)
				fprintf(stderr, "%d ", tids[i]);
		}
		fprintf(stderr, " done \n");
		
		delete[] tids;
		delete[] detail;
		delete[] offset;
	}
//	gettimeofday(&stop, 0);
//	fprintf(stdout,"[cuda] time = %.6f sec\n\n", (stop.tv_sec+stop.tv_usec*1e-6)-(start.tv_sec+start.tv_usec*1e-6));

	//cudaDeviceReset();
}

extern "C" void allocate(_t** ptr, size_t N)
{
	*ptr = new _t[N];
}
/*
int old_main()
{
	Neighbors nrbrs = File<adj_matrix>::read("graph.txt");
	_t* offset, *detail; _t offset_len, N; size_t detail_len;

	N = nrbrs.size();
	// Create a star graph
	Graph graph;
	{
		std::vector<Vertex> v_map;
		//create graphs
		for (size_t i = 0; i < nrbrs.size(); ++i)
		{
			v_map.push_back(boost::add_vertex(graph));
		}
		
		// make edges
		const size_t nrb_szie = nrbrs.size();
		for (size_t i = 0; i < nrb_szie; ++i)
		{
			for (size_t j = 0; j < nrbrs[i].size(); ++j)
			{
				if (nrbrs[i][j] < nrb_szie)
				{
					boost::add_edge(v_map[i], v_map[nrbrs[i][j]], graph);
				}
			}
		}
	}

	//print_graph(graph);
	find_bc(graph, N);
	//print_graph(graph);

	double min_b_c = 0.6;
	Neighbors updated_nrbrs;
	updated_nrbrs.resize(nrbrs.size());

	update_graph(updated_nrbrs, graph, min_b_c, N);

	fill_nrbrs(updated_nrbrs, &offset, &offset_len, &detail, &detail_len, &N);



//	_t *offset = 0, offset_len = 0, *detail = 0, N;	
//	size_t detail_len = 0;
//	get_neighbors(&offset, &offset_len, &detail, &detail_len, &N);

	size_t mem_size;
	{
		mem_size = 10*1024;
		cudaError_t e1 = cudaThreadSetLimit(cudaLimitStackSize, mem_size);
		cout << "[gpu] stack requested " << (double)mem_size/1024 << " KB"<< endl;
		//cudaError(e1); 
		if(e1 != cudaSuccess)
		{
			cudaThreadGetLimit(&mem_size, cudaLimitStackSize);
			cout << "[gpu-error] stack allocation : available " << (double)mem_size/1024 << " KB"<< endl;
		}
	}
	{	
		mem_size = 1024*1024*1024;
		cout << "[gpu] heap requested " << (double)mem_size/1024/1024 << " MB"<< endl;
		cudaError_t e2 = cudaThreadSetLimit(cudaLimitMallocHeapSize, mem_size);
		//cudaError(e2);
		if(e2  != cudaSuccess)
		{
			cudaThreadGetLimit(&mem_size, cudaLimitMallocHeapSize);
			cout << "[gpu-error] heap allocation : available " << (double)mem_size/1024 << " KB"<< endl;
		}
	}
	{
		mem_size = 1024*1024;
		cout << "[gpu] printf mem requested " << (double)mem_size/1024 << " KB"<< endl;
		cudaError_t e3 = cudaThreadSetLimit(cudaLimitPrintfFifoSize,  mem_size); 
		//cudaError(e3);
		if(e3 != cudaSuccess)
		{		
			cudaThreadGetLimit(&mem_size, cudaLimitPrintfFifoSize);
			cout << "[gpu-error] printf mem allocation : available " << (double)mem_size/1024 << " KB"<< endl;
		}
	}
	
	timeval start, stop;
	gettimeofday(&start, 0);

	{
		_t *dev_offset = 0, *dev_detail = 0;
		cuda_mem<_t> offsetDeviceMem(&dev_offset, offset_len);
		cuda_mem<_t> detailDeviceMem(&dev_detail, detail_len); 

		offsetDeviceMem.cpyToDevice(offset);
		detailDeviceMem.cpyToDevice(detail);

		size_t elements = ((N * 3) * N * (N + 1) / 2);
		{
			cout << "device heap allocations " << endl;
			cout << "[device_mem] offset " << (double)offset_len * sizeof(_t)/1024<< " KB" << endl;
			cout << "[device_mem] detail " << (double)detail_len * sizeof(_t)/1024<< " KB" << endl;
			//cout << "elements " << elements << endl;
			cout << "[device_mem] approximate total heap allocation "<< (double)elements * sizeof(_t)/1024/1024<< " MB" << endl;
			cout << "[device_mem] tids " << (double) N * sizeof(_t)/1024<< " KB" << endl;
		}
	
		_t* dev_tids, *tids = new _t[N];
		cuda_mem<_t> tidsDeviceMem(&dev_tids, N);
	
		//int dummyId = 0;	
		parallel<<<10, 128>>>(N, dev_offset, offset_len, dev_detail, dev_tids, 0, 1);
		
		tidsDeviceMem.cpyFromDevice(&tids);

		cout << "cuda threads ";
		for(_t i = 0; i < N; ++i)
			cout << tids[i] << " ";
		cout << " done " << endl;
		
		delete[] tids;
		delete[] detail;
		delete[] offset;
	}
	gettimeofday(&stop, 0);
	fprintf(stderr,"[cuda] time = %.6f sec\n\n", (stop.tv_sec+stop.tv_usec*1e-6)-(start.tv_sec+start.tv_usec*1e-6));

	cudaDeviceReset();

	return 0;
}

*/
