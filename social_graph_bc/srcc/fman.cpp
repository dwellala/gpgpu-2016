#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

enum file_type{edges_pair, adj_matrix};

typedef short _t;
typedef std::vector<_t> Neighbor;
typedef std::vector<Neighbor> Neighbors;	

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
			/*
			if (first == second)
			{
				fprintf(stderr, "[edges_pair] self loop found at node %d \n", first);
				neighbors.clear();
				inf.close();
				return neighbors;
			}
			*/
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
/*
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
*/
struct fman
{
	typedef size_t _t;
	static void undir_graph(size_t N, size_t dense)
	{
		time_t t;
		srand((unsigned)time(&t));
		_t* ptr = new _t[N * N];
		memset(ptr, 0, sizeof(_t)* N * N);
		_t i = 0, j = 0;

		const size_t lim = N * N * dense / 100;
		for (_t k = 0; k < lim; ++k)
		{
			while (i == j)
			{
				i = rand() % N;
				j = rand() % N;
			}

			ptr[N*i + j] = 1;
			ptr[N*j + i] = 1;
			i = j = 0;
		}

		std::ofstream file;
		file.open("graph.txt", std::ofstream::out);
		file << N << std::endl;
		for (_t y = 0; y < N; ++y)
		{
			for (_t x = 0; x < N; ++x)
			{
				file << ptr[y*N + x] << " ";
			}
			file << std::endl;
		}
		file.close();

		delete[] ptr;
	}

	static void analysis_mode(const std::string& file1, const std::string& file2)
	{
		typedef std::map<std::string, bool> AnMap;
		AnMap anMap;

		std::ifstream file(file1.c_str(), std::ifstream::in);
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line);
			if(line[0] == '{')
				anMap[line] = false;
		}
		file.close();

		file.open(file2.c_str(), std::ifstream::in);
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line);

			AnMap::iterator it = anMap.find(line);
			if (it != anMap.end())
				it->second = true;
		}
		file.close();

		//analysis
		size_t count = 0;
		for (AnMap::iterator it = anMap.begin(); it != anMap.end(); ++it)
		{
			if (it->second)
				++count;
		}
		
		std::cout << "cliques " << file1 << " : "<< anMap.size() << std::endl;
		std::cout << "cliques " << file2 << " : " << count << std::endl;
		std::cout << file1 << "/" << file2 << " availability : " << count * 100 / anMap.size() << " % " << std::endl;
	}
};

void from_real(size_t N, const std::string& file)
{
	const Neighbors& nbrs = File<edges_pair>::read(file);	
	if(N > nbrs.size())
	{
		std::cout << "N is too large " << std::endl;
		return;
	}

	_t* ptr = new _t[N * N];
	memset(ptr, 0, sizeof(_t)* N * N);

	for(size_t i = 0; i < N; ++i)
	{
		for(size_t j = 0; j < nbrs[i].size(); ++j)
		{
			//valid i j
			size_t _j = nbrs[i][j];
			if(_j < N && i != _j)
			{
				ptr[i+_j*N] = 1;
				ptr[_j+i*N] = 1;
			}
		}
	}	

	std::ofstream ofile;
	ofile.open("graph.txt", std::ofstream::out);
	ofile << N << std::endl;
	for (_t y = 0; y < N; ++y)
	{
		for (_t x = 0; x < N; ++x)
		{
			ofile << ptr[y*N + x] << " ";
		}
		ofile << std::endl;
	}
	ofile.close();

	delete[] ptr;
}

int main(int argc, char* argv[])
{
//	from_real(1000, "CA-GrQc.txt");

	if (argc < 4)
	{
		std::cout << "./fman <-g> <N> <dense>	for generate graphs" << std::endl;
		std::cout << "./fman <-a> <file1> <file2>	for analyze out-put cliques" << std::endl;
		std::cout << "./fman <-s> <N> <file2> part of real-social graph" << std::endl;

		return -1;
	}

	if (0 == strcmp(argv[1], "-g"))
	{
		fman::undir_graph(atoi(argv[2]), atoi(argv[3]));
	}
	else if (0 == strcmp(argv[1], "-a"))
	{
		fman::analysis_mode(argv[2], argv[3]);
	}
	else if (0 == strcmp(argv[1], "-s"))
	{
		from_real(atoi(argv[2]), argv[3]);
	}

	return 0;
}
