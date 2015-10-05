INC =  -I/usr/local/arrayfire/include -I/usr/local/cuda/include -I/opt/cuda/include -I/opt/cuda/nvvm/include
LIBS= -lafcpu
LIB_PATHS=-L/usr/lib -L/opt/cuda/nvvm/lib64/
INCLUDES=-I/usr/include
FLAGS =  -m64 -Wall -O3 -DNDEBUG -std=c++11

all :
	g++ ${FLAGS} ${INC} ${LIBS} ${LIB_PATHS} -o part main.cpp
