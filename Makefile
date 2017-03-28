PREFIX=$(Torch_ROOT)

CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH -lpng
LDFLAGS_CPP=-L$(PREFIX)/lib -lluaT -lTH 

all: libdp_kbest.so

libdp_kbest.so: dp_kbest.cpp
	g++ -fPIC -o libdp_kbest.so -shared dp_kbest.cpp $(CFLAGS) $(LDFLAGS_CPP)

clean:
	rm -f libdp_kbest.so
