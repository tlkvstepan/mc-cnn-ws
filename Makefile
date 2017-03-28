PREFIX=$(Torch_ROOT)

CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH -lpng
LDFLAGS_CPP=-L$(PREFIX)/lib -lluaT -lTH 

all: libdprog_kbest.so 

libdprog_kbest.so: dprog_kbest.cpp
	g++ -std=c++11 -fPIC -o libdprog_kbest.so -shared dprog_kbest.cpp $(CFLAGS) $(LDFLAGS_CPP)

clean:
	rm -f libdprog_kbest.so
