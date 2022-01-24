CCPP = g++
CMPI = mpicc
CCUDA = nvcc 

CFLAGS = -c
REL = -O3#-g
CUDA_REL = -O3#-g -G -lineinfo -Xcompiler -rdynamic

all: serial parallel rmobj

rm:
	rm parallel serial

parallel: cluster.o parallel.o 
	mpicc -o parallel cluster.o parallel.o -lstdc++ -lcudart -L /usr/local/cuda-5.5/lib
 
cluster.o: srcg/cluster.c
	mpicc $(REL) -c srcg/cluster.c

parallel.o: srcg/parallel.cu
	nvcc $(CUDA_REL) -c -arch=sm_21 srcg/parallel.cu

single: 
	mpirun -np 1 --hosts node0 ./parallel

cluster: 
	mpirun -np 2 --hosts node0,node1 ./parallel

rmobj:
	rm *.o

serial: srcc/serial.cpp
	$(CCPP) $(REL) -o serial srcc/serial.cpp

#parallel_g: srcg/parallel.cu
#	$(CCUDA) $(REL) -g -G -lineinfo -Xcompiler -rdynamic -o parallel -arch=sm_21 srcg/parallel.cu

#serial: srcc/serial.cpp
#	$(CCPP) $(REL) -O3 -o serial srcc/serial.cpp

#parallel_o: srcg/parallel.cu
#	$(CCUDA) $(REL) -O3 -o parallel -arch=sm_21 srcg/parallel.cu

