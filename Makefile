PREFIX=$(Torch_ROOT)

CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH -lpng
LDFLAGS_CPP=-L$(PREFIX)/lib -lluaT -lTH 

all: libdprog.so 

libdprog.so: dprog.cpp
	g++ -fPIC -o libdprog.so -shared dprog.cpp $(CFLAGS) $(LDFLAGS_CPP)

clean:
	rm -f libdprog.so
