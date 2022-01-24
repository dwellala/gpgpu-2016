#pragma once
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <fstream>
#include <vector>
#include <functional>
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

// For clustering
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>


#define gpuError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort)  exit(EXIT_FAILURE); //exit(code);
   }
}

template<typename T>
struct cuda_mem
{
	T** m_ptr;
	size_t m_n;
	cuda_mem(T** ptr, size_t n):m_ptr(ptr),
		m_n(n*sizeof(T))
	{
		cudaError_t e = cudaMalloc(&(*m_ptr), m_n);
		gpuError(e);

		if(e != cudaSuccess)
		{
			fprintf(stderr, "[cuda] error in cudaMalloc toDvc\n");
		}
	}

	~cuda_mem()
	{
		cudaError_t e = cudaFree(*m_ptr);
		gpuError(e);
		
		if(e != cudaSuccess)
		{	
			fprintf(stderr, "[cuda] error device memory de-allocation\n");
		}
	}

	void cpyFromDevice(T** dst)
	{
		cudaError_t e = cudaMemcpy(*dst, *m_ptr, m_n, cudaMemcpyDeviceToHost); 
		gpuError(e);	
		
 		if (e != cudaSuccess)                                                                                 
	    {   
			fprintf(stderr, "[cuda] error in memcpy of device to host \n");             
	    }			
	}

	void cpyToDevice(T* src)
	{
		cudaError_t e = cudaMemcpy(*m_ptr, src, m_n, cudaMemcpyHostToDevice);
		gpuError(e);
	
	    if (e != cudaSuccess)                                                                                 
	    {   
			fprintf(stderr, "[cuda] error in memcpy of Host to Device \n");  
	    }
	}
};



// Graph edge properties (bundled properties)
struct EdgeProperties {};
typedef boost::adjacency_list< boost::setS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperties > Graph;
typedef Graph::vertex_descriptor Vertex;
typedef Graph::edge_descriptor Edge;

typedef short _t;
typedef std::vector<_t> Neighbor;
typedef std::vector<Neighbor> Neighbors;	

void p(const Neighbors& neighbors)
{
	for(size_t i = 0; i < neighbors.size(); ++i)
	{
		for(size_t j = 0; j < neighbors.at(i).size(); ++j)
		{
			fprintf(stderr, "%d ", neighbors.at(i).at(j));	
		}
		fprintf(stderr, "\n");
	}
}

extern "C" void p_bar(size_t curr, size_t total, const char* str)
{
	//int count = 0;
	//cout << "Will load in 10 Sec " << endl << "Loading ";
	//for(count;count < 10; ++count){

	fprintf(stderr,	"[%d ] %s ", (curr+1)*100/total, str);
	fprintf(stderr, " \r");
	fflush(stderr);
		//sleep(1);
	//}
	//cout << endl << "Done" <<endl;
	//return 0;
}

enum file_type{edges_pair, adj_matrix};

template <file_type T>
struct File
{};

template <>
struct File<edges_pair>
{
	static Neighbors read(const std::string file_name)
	{
		std::ifstream inf(file_name.c_str());
		size_t first = 0, first_prev, second;
		Neighbors neighbors;
		Neighbor neighbor;
		while (!inf.eof())
		{
			first_prev = first;
			inf >> first;
			inf >> second;
			if (first == second)
			{
				fprintf(stderr, "[edges_pair] self loop found at node %d \n", first);
				neighbors.clear();
				inf.close();
				return neighbors;
			}
			if (first_prev == first)
				neighbor.push_back(second);
			else
			{
				neighbors.push_back(neighbor);
				neighbor.clear();
				neighbor.push_back(second);
			}
		}
		neighbors.push_back(neighbor);
		inf.close();
		return neighbors;
	}
};

template <>
struct File<adj_matrix>
{
	static Neighbors read(const std::string& file_name)
	{
		size_t N;
		std::ifstream inf(file_name.c_str());
		inf >> N;
		size_t edge;
		Neighbors neighbors;
		Neighbor neighbor;
		for (size_t i = 0; i < N; ++i)
		{
			neighbor.clear();
			for (size_t j = 0; j < N; ++j)
			{
				inf >> edge;
				if (edge == 1)
				{
					neighbor.push_back(j);
					if (i == j)
					{
						fprintf(stderr, "[adj_matrix]self loop found at node %d \n", i);
						neighbors.clear();
						inf.close();
						return neighbors;
					}
				}
			}
			neighbors.push_back(neighbor);
		}
		inf.close();
		return neighbors;
	}
};

void fill_nrbrs(const Neighbors& neighbors, _t** offset, _t* offset_len, _t** detail, size_t* detail_len, _t* N)
{
	*N = neighbors.size();

	*offset_len = *N;
	*detail_len = 0;
	(*offset) = new _t[*offset_len];

	for (_t i = 0; i < *N; ++i)
	{
		*detail_len += neighbors[i].size();
		(*offset)[i] = neighbors[i].size();
	}

	(*detail) = new _t[*detail_len];
	size_t index = 0;
	for (_t i = 0; i < *N; ++i)
	{
		for (size_t j = 0; j < neighbors[i].size(); ++j, index++)
		{
			(*detail)[index] = neighbors[i][j];
		}
	}
}

void find_bc(Graph& graph, double threshold)
{
	// std::map used for convenient initialization
	typedef std::map<Edge, int> StdEdgeIndexMap;
	StdEdgeIndexMap my_e_index;
	// associative property map needed for iterator property map-wrapper
	typedef boost::associative_property_map< StdEdgeIndexMap > EdgeIndexMap;
	EdgeIndexMap e_index(my_e_index);

	// We use setS as edge-container -> no automatic indices
	// -> Create and set it explicitly
	int i = 0;
	BGL_FORALL_EDGES(edge, graph, Graph)
	{
		my_e_index.insert(std::pair< Edge, int >(edge, i));
		++i;
	}

	// Define EdgeCentralityMap
	std::vector< double > e_centrality_vec(boost::num_edges(graph), 0.0);
	// Create the external property map
	boost::iterator_property_map< std::vector< double >::iterator, EdgeIndexMap > e_centrality_map(e_centrality_vec.begin(), e_index);


	// Define VertexCentralityMap
	typedef boost::property_map< Graph, boost::vertex_index_t>::type VertexIndexMap;
	{
		VertexIndexMap v_index = get(boost::vertex_index, graph);
		std::vector< double > v_centrality_vec(boost::num_vertices(graph), 0.0);
		// Create the external property map
		boost::iterator_property_map< std::vector< double >::iterator, VertexIndexMap > v_centrality_map(v_centrality_vec.begin(), v_index);

		// Define the done-object:
		// 'false' means here that no normalization is performed, so edge centrality-values can become big
		// If set to 'true', values will range between 0 and 1 but will be more difficult to use for this
		// illustrative example.
		boost::bc_clustering_threshold< double > terminate(threshold, graph, false);

		// Do the clustering
		// Does also calculate the brandes_betweenness_centrality and stores it in e_centrality_map
		betweenness_centrality_clustering(graph, terminate, e_centrality_map);
	}

	typedef std::map<double, _t> CentDistribution;
	CentDistribution dist_map;
	BGL_FORALL_EDGES(edge, graph, Graph)
	{
		++(dist_map[e_centrality_map[edge]]);
		//std::cout << edge << ": " << e_centrality_map[edge] << std::endl;
	}

	//fprintf(stderr, "# edges\t|\tb-c\t|\tcount\n");

	//size_t count = 0;
	fprintf(stderr, "betweenes cent distribution [bc: count]\n");
	for(CentDistribution::reverse_iterator it = dist_map.rbegin(); it != dist_map.rend(); ++it)
	{
		//count += it->second;	
		//std::cout << count << "\t|\t" <<it->first << "\t|\t" << it->second << std::endl;
		//fprintf(stderr, "%d\t|\t%G\t|\t%d\n", count, it->first, it->second);
		fprintf(stderr, "[%G: %d] ", it->first, it->second);
	}
	fprintf(stderr, "\n## bc range [%G - %G]\n", dist_map.rbegin()->first, dist_map.begin()->first);
}

void update_graph(Neighbors& updated_nrbrs, Graph& graph, double threshold, double N)
{
	//print_graph(graph);
	// std::map used for convenient initialization
	typedef std::map<Edge, int> StdEdgeIndexMap;
	StdEdgeIndexMap my_e_index;
	// associative property map needed for iterator property map-wrapper
	typedef boost::associative_property_map< StdEdgeIndexMap > EdgeIndexMap;
	EdgeIndexMap e_index(my_e_index);

	// We use setS as edge-container -> no automatic indices
	// -> Create and set it explicitly
	int i = 0;
	BGL_FORALL_EDGES(edge, graph, Graph)
	{
		my_e_index.insert(std::pair< Edge, int >(edge, i));
		++i;
	}

	// Define EdgeCentralityMap
	std::vector< double > e_centrality_vec(boost::num_edges(graph), 0.0);
	// Create the external property map
	boost::iterator_property_map< std::vector< double >::iterator, EdgeIndexMap > e_centrality_map(e_centrality_vec.begin(), e_index);


	// Define VertexCentralityMap
	typedef boost::property_map< Graph, boost::vertex_index_t>::type VertexIndexMap;
	{
		VertexIndexMap v_index = get(boost::vertex_index, graph);
		std::vector< double > v_centrality_vec(boost::num_vertices(graph), 0.0);
		// Create the external property map
		boost::iterator_property_map< std::vector< double >::iterator, VertexIndexMap > v_centrality_map(v_centrality_vec.begin(), v_index);

		// Define the done-object:
		// 'false' means here that no normalization is performed, so edge centrality-values can become big
		// If set to 'true', values will range between 0 and 1 but will be more difficult to use for this
		// illustrative example.
		boost::bc_clustering_threshold< double > terminate(N, graph, false);

		// Do the clustering
		// Does also calculate the brandes_betweenness_centrality and stores it in e_centrality_map
		betweenness_centrality_clustering(graph, terminate, e_centrality_map);
	}

	_t target, source, size = updated_nrbrs.size();
	BGL_FORALL_EDGES(edge, graph, Graph)
	{
		source = boost::source(edge, graph);
		target = boost::target(edge, graph);

		//filtered and make it undirected
		if (e_centrality_map[edge] > threshold && size > source && size > target)
		{
			updated_nrbrs[source].push_back(target);
			updated_nrbrs[target].push_back(source);
		}
		//	std::cout << "[" << boost::source(edge, graph) << "|" << boost::target(edge, graph) << "]" << ": " << e_centrality_map[edge] << std::endl;
	}
}


