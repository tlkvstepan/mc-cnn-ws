extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>    //min
#include <omp.h>

#include <stdio.h>

#include <math.h>
#include <iostream>

#include "luaT.h"

#define I2D(col, row, width, height) I3D(col, row, 0, width, height, 1)
#define I3D(col, row, ch, width, height, nb_ch) ( ( (ch) * (height) + (row) ) * (width) + (col) ) 

#include<TH/TH.h>


void trace(float *aE, float *aS, float *aP,  float *T, int w, int h)
{  
  // find max average energy in last column
  int bestCol = 0;
  int idx = SUB2IND_2D_TORCH(bestCol, h-1, w, h);
  float maxAvgE = (aE[idx]) / (aS[idx] + 1e-10);
  for( int ncol = 0; ncol < w; ++ncol )
  {
    idx = SUB2IND_2D_TORCH(ncol, h-1, w, h);
    float curAvgE = (aE[idx]) / (aS[idx] + 1e-10);
    if( maxAvgE  < curAvgE ){
      bestCol = ncol; 
      maxAvgE = curAvgE;
      
    }
  }
  
  // propogate best energy back
  T[SUB2IND_2D_TORCH(bestCol, h-1, w, h)] = 1;
  idx = aP[SUB2IND_2D_TORCH(bestCol, h-1, w, h)];
  while (idx >= 0) 
  {
    T[idx] = 1;    
    idx = aP[idx];
  }    
}

bool isValid(x, y, dim, dispMax)
{
	if(x >= y) return false;
	else if( x < (y - dispMax) ) return false;
	else if( (x < 0) || x >= dim ) return false;
	else if( (y < 0) || y >= dim ) return false;
	else return true;
}

void acc(const float *E, float *aE, float *aL, float *T, int dim, int dispMax, int pathNum )
{
  /*
  	E  is w x h energy array
  	aE is w x h x k accumulated energy of best path to cell
  	aL is w x h x k accumulated length of the best path to cell
  	T  is w x h x k previous cell in the best path 
  */

  int x, y, k;
  int dx[3] = {-1,  0, -1};
  int dy[3] = {0 , -1, -1};

  // make mask
  

  // initialization
  x = 0;
  for( y = 1; y <= dispMax; ++y ) // 0 disparity is not allowed
  for( k = 0; k <= pathNum-1; ++k ){
  	
  	aE[I3D(x, y, k, dim, dim, pathNum)] = E[I2D(x, y, dim, dim)];
  	aL[I3D(x, y, k, dim, dim, pathNum)] = 1;
  	T[I3D(x, y, k, dim, dim, pathNum)] = -1; // begining
  
  }
    
  // process column by column starting from second
  for( x = 1; x < dim-1; ++x){
    
    yMin = x + 1;
    yMax = x + dispMax;
    
    for( y = yMin; y <= yMax; ++y){

    	if ~isValid(x, y, dim, dispMax) continue;

    	for( k = 0; k < pathNum

	    	for( n = 0; n <= 2; ++n  ){ 
	  	       
	  	       yy = y + dy[n];
	  	       xx = x + dx[n];

	  	       if ~isValid(xx, yy, dim, dispMax) continue;



	  	    
	  	    }
  	}


  	ncolMin = nrow;  
  	ncolMax = ncolMin + disp_max;
  	for( int ncol = ncolMin; ncol < ncolMax; ++ncol )
    {
      // *** if el is masked skip it
      if( mask[curIdx] == 0 ){
        curIdx++;     
        continue;
      } 
      
      float bestPathAvgE =  0;
      float bestPathLen =  0;
      float bestPathE =  0;
      int   bestPathTraceBack = 0;
      
      for( int nneig = 0; nneig < nb_neig; ++nneig )
    	{
      
        prevIdx = curIdx + didx[nneig];
        
        // *** if el is masked skip it
        if( mask[prevIdx] == 0 ) continue;
        
        float curPathE;
        float curPathLen;

        curPathE   = pathE[prevIdx] + E[curIdx];
        curPathLen = pathLen[prevIdx] + 1;
    
        float curPathAvgE = (curPathE) / (curPathLen + 1e-10);
                
        if( bestPathAvgE <= curPathAvgE ) 
        {
          bestPathTraceBack = prevIdx;
          bestPathE = curPathE;
          bestPathAvgE = curPathAvgE;
          bestPathLen = curPathLen;
        }
      }	
      
      pathE[curIdx] = bestPathE;
      pathLen[curIdx] = bestPathLen;
      traceBack[curIdx] = bestPathTraceBack;
      curIdx++;     
    }
    
  }    

}



int findNonoccPath(lua_State *L)
{
    // input
    THFloatTensor *path_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *pathNonOcc_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    float occTh = luaL_checknumber(L, 3);
        
    int w = THFloatTensor_size(path_, 1);
    int h = THFloatTensor_size(path_, 0);

    float *path = THFloatTensor_data(path_);                // Energy
    float *pathNonOcc = THFloatTensor_data(pathNonOcc_);                // Energy
        
    int sumCol[2000] = {0};
    int sumRow[2000] = {0};
        
    int idx = 0;    
    for( int nrow = 0; nrow < h; ++nrow )
    for( int ncol = 0; ncol < w; ++ncol ){ 
        sumCol[ncol] += path[idx];
        sumRow[nrow] += path[idx];
        ++idx;
    }
        
     // mark as occluded 
    
    idx = 0;
    for( int nrow = 0; nrow < h; ++nrow )
    for( int ncol = 0; ncol < w; ++ncol ){ 
      pathNonOcc[idx] = path[idx];
      if( path[idx] == 1 )
      {
        if( sumCol[ncol] > occTh || sumRow[nrow] > occTh )
        {
            pathNonOcc[idx] = 0;
        }
      }
      ++idx;
    }
     
    
  return 0;
}

/*
  This function computes path with maximum average cost (occlusions do not contribute to cost).
*/
int compute(lua_State *L)
{
    // input
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *pathE_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *pathLen_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    THFloatTensor *traceBack_ = (THFloatTensor*)luaT_checkudata(L, 5, "torch.FloatTensor");
    
    // output
    THFloatTensor *pathOpt_  = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    
    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
    int k = THFloatTensor_size(pathE_, 2);
   
    float *E = THFloatTensor_data(E_);                // Energy
    float *pathE = THFloatTensor_data(pathE_);
    float *pathLen = THFloatTensor_data(pathLen_);
    float *traceBack = THFloatTensor_data(traceBack_);
    
    float *pathOpt = THFloatTensor_data(pathOpt_);          // Optimal path
        
    accumulate(E, E, pathE, pathLen, traceBack, w, h);
    trace(pathE, pathLen, traceBack, pathOpt, w, h);
        
   return 0;
}

static const struct luaL_Reg funcs[] = {
    {"compute", compute},
    {"findNonoccPath", findNonoccPath},
    {NULL, NULL}
};

extern "C" int luaopen_libdp_kbest(lua_State *L) {
    luaL_openlib(L, "dp_kbest", funcs, 0);
    return 1;
}
