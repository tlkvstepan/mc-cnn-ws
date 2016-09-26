extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include <limits>
#include <cmath>
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
    
    if( aE[SUB2IND_2D_TORCH(maxcol, h-1, w, h)] / aS[SUB2IND_2D_TORCH(maxcol, h-1, w, h)]  < 
        aE[SUB2IND_2D_TORCH(ncol, h-1, w, h)] / aS[SUB2IND_2D_TORCH(ncol, h-1, w, h)] ) maxcol = ncol; 
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

void accumulate(const float *E, float *aE, float* aS, float *traceBack, int w, int h)
{

  /*

  We start from top of energy array and continue to the bottom (looking for maximum average energy path).
  Only steps down, down-right and right are allowed (we want to have continuous line).
  Step down and right correspond to occlusion and incur occlusion penalty.
  Step down-righ incur matching cost 
    
  E  - energy of each cell  
  aE - energy of best path to cell	
  traceBack  - index of previous cell 
  aS  - number of steps along best path to cell
  
  */
  
  // from top, left, left-top
  int dx[3] = { 0, -1, -1};
  int dy[3] = {-1,  0, -1};
  int nb_neig = 3;
  
  // initialize top row
  for( int ncol = 0; ncol < w; ++ncol ){
  	int idx = SUB2IND_2D_TORCH(ncol, 0, w, h); 
    aE[idx] = E[idx];
    traceBack[idx] = -1;
    aS[idx] = 1;
  }   
  
  // go from top row down, computing best accumulated energy in every position 
  for( int nrow = 1; nrow < h; ++nrow) 
  {
    int curIdx, prevIdx;
    
    // to each first cell in row we could start or arrive from top
    curIdx = SUB2IND_2D_TORCH(0, nrow, w, h);
    prevIdx = SUB2IND_2D_TORCH(0, nrow - 1, w, h);
    
    if( E[curIdx] > (aE[prevIdx] + E[curIdx]) / (aS[prevIdx] + 1) )
    {
      // we start here
      aE[curIdx] = E[curIdx];
      traceBack[curIdx] = -1;
      aS[curIdx] = 1;
    }else{
      // we came from top
      aE[curIdx] = E[curIdx] + aE[prevIdx];
      traceBack[curIdx] = prevIdx;
      aS[curIdx] = aS[prevIdx] + 1;
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
        
        float curE = aE[prevIdx];
        
        if( nneig == 2 ) curE += E[curIdx];  // if add match energy
        
        // compute average energy
        float curS = aS[prevIdx] + 1;
        float avgE = curE / curS;
        
        if( bestPrevIdx == -1 || bestAvgE < avgE ) 
        {
          bestPrevIdx = prevIdx;
          bestE = curE;
          bestAvgE = avgE;
          bestS = curS;
        }
       
    	}	
      
      aE[curIdx] = bestE;
      traceBack[curIdx] = bestPrevIdx;
      aS[curIdx] = bestS;
           
    }
    
  }    

}


int compute(lua_State *L)
{
    // input
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    
    // output
    THFloatTensor *pathOpt_  = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *costRef2Pos_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *costPos2Ref_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    THFloatTensor *indexRef2Pos_ = (THFloatTensor*)luaT_checkudata(L, 5, "torch.FloatTensor");
    THFloatTensor *indexPos2Ref_ = (THFloatTensor*)luaT_checkudata(L, 6, "torch.FloatTensor");

    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
   
    float *E = THFloatTensor_data(E_);                // Energy
    float *pathOpt = THFloatTensor_data(pathOpt_);          // Optimal path
    float *costRef2Pos = THFloatTensor_data(costRef2Pos_);      // Match cost (or -2 if no match)
    float *costPos2Ref = THFloatTensor_data(costPos2Ref_);      // Match cost (or -2 if no match)
    float *indexRef2Pos = THFloatTensor_data(indexRef2Pos_);      // Match cost (or -2 if no match)
    float *indexPos2Ref = THFloatTensor_data(indexPos2Ref_);      // Match cost (or -2 if no match)

    float *aE = new float[w*h];
    float *aSteps = new float[h*w];
    float *traceBack = new float[h*w];
        
    accumulate(E, aE, aSteps, traceBack, w, h);
    trace(aE, aSteps, traceBack, pathOpt, w, h);
    
    int *sumCol = new int[w];
    int *sumRow = new int[h];
    for( int nrow = 0; nrow < h; ++nrow ){
      for( int ncol = 0; ncol < w; ++ncol ){ 
        int idx = SUB2IND_2D_TORCH(ncol, nrow, w, h);
        sumCol[ncol] += pathOpt[idx];
        sumRow[nrow] += pathOpt[idx];
      }
    }
    
    // mark as occluded 
    std::fill(costRef2Pos, costRef2Pos + w, HUGE_VALF);
    std::fill(costPos2Ref, costPos2Ref + w, HUGE_VALF);
    for( int nrow = 0; nrow < h; ++nrow ){
      for( int ncol = 0; ncol < w; ++ncol ){ 
        int idx = SUB2IND_2D_TORCH(ncol, nrow, w, h);
        if( pathOpt[idx] == 1 )
        {
          if( sumCol[ncol] == 1 && sumRow[nrow] == 1 ){ // check if occluded 
             costRef2Pos[nrow] = (float)E[idx];
             costPos2Ref[ncol] = (float)E[idx];
             indexRef2Pos[nrow] = ncol;
             indexPos2Ref[ncol] = nrow;             
          }
        }
      }
    }
    
    delete[] aE;
    delete[] aSteps;
    delete[] traceBack;
    delete[] sumCol;
    delete[] sumRow;
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
