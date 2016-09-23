extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include <algorithm>    //min

#include <stdio.h>

#include <math.h>
#include <iostream>

#include "luaT.h"

#define SUB2IND_2D_TORCH(col, row, width, height) SUB2IND_3D_TORCH(col, row, 0, width, height, 1)
#define SUB2IND_3D_TORCH(col, row, ch, width, height, nb_ch) ( ( (ch) * (height) + (row) ) * (width) + (col) ) 

#include<TH/TH.h>


void trace(float *aE, float *aP, float *T, int w, int h)
{  
  // find max energy in last column
  int maxcol = 0;
  for( int ncol = 1; ncol < w; ++ncol )
    if( aE[SUB2IND_2D_TORCH(maxcol, h-1, w, h)] < aE[SUB2IND_2D_TORCH(ncol, h-1, w, h)] ) maxcol = ncol; 
  
  T[h-1] = maxcol;    

  // propogate best energy back
  for( int nrow = h-2; nrow >= 0; --nrow )
    T[nrow] = aP[SUB2IND_2D_TORCH((int)(T[nrow+1]), nrow+1, w, h)];

}

void accumulate(const float *E, float *aE, float* S, float *P, int w, int h)
{

  /*

  We start from right side of energy array and continue to the bottom.
  Only steps down, down-right and right are allowed (we want to have continuous line).
  
  aE - cost of best path to current cell	
  P  - index of previous cell 
  S  - number of steps in best path to current cell

  */
  
  // top, left, left-top
  int dx[3] = { 0, -1, -1};
  int dy[3] = {-1,  0, -1};
  int nb_neig = 3;

  // initialize top row
  for( int ncol = 0; ncol < w; ++ncol ){
  	int idx = SUB2IND_2D_TORCH(0, ncol, w, h); 
    aE[idx] = E[idx];
    P[idx] = 0;
    S[idx] = 0;  
  }   
  
  // go from top row down, computing best accumulated energy in every position 
  for( int nrow = 1; nrow < h; ++nrow) 
  {
    
    curIdx = SUB2IND_2D_TORCH(nrow, 0, w, h);
    topIdx = SUB2IND_2D_TORCH(nrow-1, 0, w, h);

    // to first cell of the row we can move only from:
    // (1) top cell, or (2) we can start 
    float startE = E[curIdx];
    float fromTopE = (aE[topIdx] + E[curIdx]) / (S[topIdx] + 1)  
    if startE > fromTopE  
    {
    	aE[curIdx] = E[curIdx]; 
    	P[curIdx] = 0; 
    	S[curIdx] = 1;
    }else{
    	aE[curIdx] = E[curIdx] + aE[topIdx];
    	P[curIdx] = topIdx;
    	S[curIdx] = S[topIdx] + 1;
    }	

    // to each cell we can arrive from left, top or left top 
    for( int ncol = 1; ncol < w; ++ncol )
    {
    	curIdx = SUB2IND_2D_TORCH(nrow, ncol, w, h);
    	for( int nneig = 0; nneig < nb_neig; ++nneigh )
    	{
    		prevIdx = SUB2IND_2D_TORCH(nrow + dy[nneig], ncol + dx[nneig], w, h);
    	

    	}	

    	topIdx = SUB2IND_2D_TORCH(nrow-1, ncol, w, h);
    	leftIdx = SUB2IND_2D_TORCH(nrow, ncol-1, w, h);
		topLeftIdx = SUB2IND_2D_TORCH(nrow-1, ncol-1, w, h);

    	float fromTopE 	= (aE[topIdx] + E[curIdx]) / (S[topIdx] + 1)  
    	float fromLeftE = (aE[leftIdx] + E[curIdx]) / (S[leftIdx] + 1)  
		float fromTopLeftE = (aE[topLeftIdx] + E[curIdx]) / (S[topLeftIdx] + 1)  

		if fromTopE > fromLeftE
			if
        int idx = SUB2IND_2D_TORCH(ncol, nrow-1, w, h);
        if( maxVal[ncol-1] < aE[idx] )
        { 
          maxVal[ncol] = aE[idx];
          maxIdx[ncol] = ncol;
        }else{
          maxIdx[ncol] = maxIdx[ncol-1];
          maxVal[ncol] = maxVal[ncol-1];
        }
    }
  
    // non-strict monotonicity: we can never go left
    for( int ncol = 0; ncol < w; ++ncol )
    {
      //  std::cout << maxVal[ncol] << " ";   
      int idx = SUB2IND_2D_TORCH(ncol, nrow, w, h);   
      aE[idx] = maxVal[ncol] + E[idx];
      aP[idx] = maxIdx[ncol];
    }
     // std::cout << "\n";   
   }
  
  delete[] maxIdx;
  delete[] maxVal;

}


int compute(lua_State *L)
{
   
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *aE_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *aP_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");

    THFloatTensor *indices_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    THFloatTensor *values_ = (THFloatTensor*)luaT_checkudata(L, 5, "torch.FloatTensor");

    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
    
    //std::cout << "width " << w << "\n";   
    //std::cout << "height " << h << "\n";   
    

    float *E = THFloatTensor_data(E_);
    float *aE = THFloatTensor_data(aE_);
    float *aP = THFloatTensor_data(aP_);
    float *indices = THFloatTensor_data(indices_);
    float *values = THFloatTensor_data(values_);

    //std::cout << "height " << E[SUB2IND_2D_TORCH(3,2, w, h)] << "\n";   

    //double *aE = new double[h*w];
    //double *aP = new double[h*w];
    
    accumulate(E, aE, aP, w, h);
    trace(aE, aP, indices, w, h);
 
    for( int nrow = 0; nrow < h; ++nrow ){
        values[nrow] = E[SUB2IND_2D_TORCH((int)indices[nrow], nrow, w, h)];
    }
    
   // delete[] aE;
   // delete[] aP;

    return 0;
}

static const struct luaL_Reg funcs[] = {
    {"compute", compute},
    {NULL, NULL}
};

extern "C" int luaopen_libdprog(lua_State *L) {
    luaL_openlib(L, "dprog", funcs, 0);
    return 1;
}
