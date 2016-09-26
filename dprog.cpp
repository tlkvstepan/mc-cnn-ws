extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include <limits>
#include <algorithm>    //min

#include <stdio.h>

#include <math.h>
#include <iostream>

#include "luaT.h"

#define SUB2IND_2D_TORCH(col, row, width, height) SUB2IND_3D_TORCH(col, row, 0, width, height, 1)
#define SUB2IND_3D_TORCH(col, row, ch, width, height, nb_ch) ( ( (ch) * (height) + (row) ) * (width) + (col) ) 

#include<TH/TH.h>


void trace(float *aE, float *aS, float *aP, float *T, int w, int h)
{  
  // find max average energy in last column
  int maxcol = 0;
  for( int ncol = 1; ncol < w; ++ncol )
  {
    if( aE[SUB2IND_2D_TORCH(maxcol, h-1, w, h)] / aS[SUB2IND_2D_TORCH(maxcol, h-1, w, h)]  < aE[SUB2IND_2D_TORCH(ncol, h-1, w, h)] / aS[SUB2IND_2D_TORCH(ncol, h-1, w, h)] ) maxcol = ncol; 
  }
  
  // propogate best energy back
  T[SUB2IND_2D_TORCH(maxcol, h-1, w, h)] = 1;
  int idx = aP[SUB2IND_2D_TORCH(maxcol, h-1, w, h)];
  while (idx >= 0) 
  {
    T[idx] = 1;    
    idx = aP[idx];
  }  
  
}

void accumulate(const float *E, float *accE, float* S, float *P, int w, int h)
{

  /*

  We start from top of energy array and continue to the bottom (looking for maximum average energy path).
  Only steps down, down-right and right are allowed (we want to have continuous line).
  Step down and right correspond to occlusion and incur occlusion penalty.
  Step down-righ incur matching cost 
    
  aE - cost of best path to current cell	
  P  - index of previous cell 
  S  - number of steps along 
  
  */
  
  // from top, left, left-top
  int dx[3] = { 0, -1, -1};
  int dy[3] = {-1,  0, -1};
  int nb_neig = 3;
  
  // initialize top row
  for( int ncol = 0; ncol < w; ++ncol ){
  	int idx = SUB2IND_2D_TORCH(ncol, 0, w, h); 
    accE[idx] = E[idx];
    P[idx] = -1;
    S[idx] = 1;
  }   
  
  // go from top row down, computing best accumulated energy in every position 
  for( int nrow = 1; nrow < h; ++nrow) 
  {
    int curIdx, prevIdx;
    
    // to each first cell in row we could start or arrive from top
    curIdx = SUB2IND_2D_TORCH(0, nrow, w, h);
    prevIdx = SUB2IND_2D_TORCH(0, nrow - 1, w, h);
    
    if( E[curIdx] > (accE[prevIdx] + E[curIdx]) / (S[prevIdx] + 1) )
    {
      // we start here
      accE[curIdx] = E[curIdx];
      P[curIdx] = -1;
      S[curIdx] = 1;
    }else{
      // we came from top
      accE[curIdx] = E[curIdx] + accE[prevIdx];
      P[curIdx] = prevIdx;
      S[curIdx] = S[prevIdx] + 1;
    }
    
   
    // to each cell we can arrive from left, top or left top 
    for( int ncol = 1; ncol < w; ++ncol )
    {
      curIdx = SUB2IND_2D_TORCH(ncol, nrow, w, h);
    	
      float bestE =  0;
      float bestAvgE =  0;
      float bestS =  0;
      int bestPrevIdx = -1;
      
      for( int nneig = 0; nneig < nb_neig; ++nneig )
    	{
      
        prevIdx = SUB2IND_2D_TORCH(ncol + dx[nneig], nrow + dy[nneig], w, h);
        
        float curE = accE[prevIdx];
        
        if( nneig == 2 ) curE += E[curIdx];  // if add match energy
        
        // compute average energy
        float curS = S[prevIdx] + 1;
        float avgE = curE / curS;
        
        if( bestPrevIdx == -1 || bestAvgE < avgE ) 
        {
          bestPrevIdx = prevIdx;
          bestE = curE;
          bestAvgE = avgE;
          bestS = curS;
        }
       
    	}	
      
      accE[curIdx] = bestE;
      P[curIdx] = bestPrevIdx;
      S[curIdx] = bestS;
     
      
    }
    
  }    

}


int compute(lua_State *L)
{
   
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *aE_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *aS_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *aP_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");

    THFloatTensor *T_ = (THFloatTensor*)luaT_checkudata(L, 5, "torch.FloatTensor");
  //  THFloatTensor *values_ = (THFloatTensor*)luaT_checkudata(L, 6, "torch.FloatTensor");

    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
    
    std::cout << "width " << w << "\n";   
    std::cout << "height " << h << "\n";   
    

    float *E = THFloatTensor_data(E_);
    float *aE = THFloatTensor_data(aE_);
    float *aS = THFloatTensor_data(aS_);
    float *aP = THFloatTensor_data(aP_);
    float *T = THFloatTensor_data(T_);
  //  float *values = THFloatTensor_data(values_);

    //std::cout << "height " << E[SUB2IND_2D_TORCH(3,2, w, h)] << "\n";   

    //double *aE = new double[h*w];
    //double *aP = new double[h*w];
    
    accumulate(E, aE, aS, aP, w, h);
    trace(aE, aS, aP, T, w, h);
 
   // for( int nrow = 0; nrow < h; ++nrow ){
   //     values[nrow] = E[SUB2IND_2D_TORCH((int)indices[nrow], nrow, w, h)];
   // }
    
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
