PREFIX=/home/tulyakov/torch/install

CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH -lpng
LDFLAGS_CPP=-L$(PREFIX)/lib -lluaT -lTH 

all: libdynprog.so 

libdynprog.so: dynprog.cpp
	g++ -fPIC -o libdynprog.so -shared dynprog.cpp $(CFLAGS) $(LDFLAGS_CPP)

clean:
	rm -f libdynprog.so
