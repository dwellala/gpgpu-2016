#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <sys/time.h>
typedef short _t;

void run_on_gpu(_t* offset, _t offset_len, _t* detail, size_t detail_len, _t N, int, int);

void get_data(const char* filename, _t** offset, _t* offset_len, _t**detail, size_t* detail_len, _t* N);
void get_data_sim(const char* filename, _t** offset, _t* offset_len, _t**detail, size_t* detail_len, _t* N);

void p_bar(size_t, size_t, const char*);
void allocte(_t** ptr, size_t N);

int main(int argc, char **argv)
{	
	int numTasks, taskId;
	if(MPI_Init(&argc, &argv) != MPI_SUCCESS)
	{
		fprintf(stderr, "[mpi]Error with init\n");
		return 1;
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskId);
	
	struct timeval start, stop;
	if(taskId == 0)//master
	{
		_t* offset, *detail;
		_t offset_len, N;
		size_t detail_len;				
		
		get_data_sim("graph.txt", &offset, &offset_len, &detail, &detail_len, &N);

		gettimeofday(&start, 0);
		if(numTasks > 1)
		{
			int slave;
			for(slave = 1; slave < numTasks; ++slave)
			{
				MPI_Send(&offset_len, 1, MPI_SHORT, slave, 1, MPI_COMM_WORLD);
				MPI_Send(&detail_len, 1, MPI_UNSIGNED, slave, 1, MPI_COMM_WORLD);
				MPI_Send(&N, 1, MPI_SHORT, slave, 1, MPI_COMM_WORLD);
				MPI_Send(offset, offset_len, MPI_SHORT, slave, 1, MPI_COMM_WORLD);
				MPI_Send(detail, detail_len, MPI_SHORT, slave, 1, MPI_COMM_WORLD);
			}
			gettimeofday(&stop, 0);
			fprintf(stderr,"## data communication among GPU(S) %.6f sec\n\n", (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6));		
		}
		
		gettimeofday(&start, 0);
		run_on_gpu(offset, offset_len, detail, detail_len, N, taskId, numTasks);
		gettimeofday(&stop, 0);
		fprintf(stderr,"## process on GPU %d %.6f sec\n\n", taskId, (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6));		

	}
	else
	{
		_t* offset, *detail;
		_t offset_len, N;
		size_t detail_len;				
	
		MPI_Status status;
		MPI_Recv(&offset_len, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&detail_len, 1, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&N, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD, &status);

		allocate(&offset, offset_len);
		allocate(&detail, detail_len);
		MPI_Recv(offset, offset_len, MPI_SHORT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(detail, detail_len, MPI_SHORT, 0, 1, MPI_COMM_WORLD, &status);	

		gettimeofday(&start, 0);
		run_on_gpu(offset, offset_len, detail, detail_len, N, taskId, numTasks);			
		gettimeofday(&stop, 0);
		fprintf(stderr,"## process on GPU %d %.6f sec\n\n", taskId, (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6));		

	}

	MPI_Finalize();
	return 0;
}
