extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include <limits>
#include <cmath>
#include <algorithm>    //min
#include <omp.h>

#include <stdio.h>

#include <math.h>
#include <iostream>

#include "luaT.h"

#define SUB2IND_2D_TORCH(col, row, width, height) SUB2IND_3D_TORCH(col, row, 0, width, height, 1)
#define SUB2IND_3D_TORCH(col, row, ch, width, height, nb_ch) ( ( (ch) * (height) + (row) ) * (width) + (col) ) 

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

void accumulate(const float *E, const float* mask, float *pathE, float *pathLen, float *traceBack, int w, int h)
{
  
  int dx[3] = { 0, -1, -1};
  int dy[3] = {-1,  0, -1};
  int didx[3];
  didx[0] = -w;
  didx[1] = -1;
  didx[2] = -w-1;
  int nb_neig = 3;
  
  int curPathE, curPathLen;
  int curIdx, prevIdx;
  
  // *first element of first row we just initialize
  // *if mask permits
  if( mask[0] > 0 )
  {
    pathE[0] = E[0];
    pathLen[0] = 1;
    traceBack[0] = -1;  // mark begining
  }
  
  // * first row
  // * we can come only from left cell 
  for( int ncol = 1; ncol < w; ++ncol )
  {
    // ** if cur el is masked skip 
    if( mask[ncol] == 0 ) continue;
    // ** if prev el is masked skip
    if( mask[ncol-1] == 0 ) continue;
    
    curPathE = pathE[ncol-1] + E[ncol];
    curPathLen = pathLen[ncol-1] + 1;
    
    pathE[ncol] = curPathE;
    pathLen[ncol] = curPathLen;
    traceBack[ncol] = ncol-1;  
  }
  
  // * all other rows
  for( int nrow = 1; nrow < h; ++nrow) 
  {
  
    // ** first element of row we just initialize
    // ** if mask permits
    curIdx = nrow*w;
    if( mask[curIdx] > 0 )
    {
      pathE[curIdx] = E[curIdx];
      pathLen[curIdx] = 1;
      traceBack[curIdx] = -1;
    }
    
    // ** other elements of the row 
    curIdx++;
    for( int ncol = 1; ncol < w; ++ncol )
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


int findMaxForRows(lua_State *L)
{
    // input
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *row_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *matchIdx_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *matchE_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    
    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
    int n = THFloatTensor_size(row_, 0);

    float *E = THFloatTensor_data(E_);         
    float *row = THFloatTensor_data(row_);              
    float *matchIdx = THFloatTensor_data(matchIdx_);         
    float *matchE = THFloatTensor_data(matchE_);         
    
    
    for( int nrow = 0; nrow < n; ++nrow ) {
      float bestCol = 0;
      float maxE = -HUGE_VALF; 
      for( int col = 0; col < w; ++col ){
        int idx = SUB2IND_2D_TORCH(col, row[nrow], w, h);
        if( maxE < E[idx] )
        {
          maxE = E[idx];
          bestCol = col;
        }
      }
      matchIdx[nrow] = bestCol;
      matchE[nrow] = maxE;
    }
     
  return 0;
}

int findMaxForCols(lua_State *L)
{
    // input
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *col_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *matchIdx_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *matchE_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    
    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
    int n = THFloatTensor_size(col_, 0);

    float *E = THFloatTensor_data(E_);         
    float *col = THFloatTensor_data(col_);              
    float *matchIdx = THFloatTensor_data(matchIdx_);         
    float *matchE = THFloatTensor_data(matchE_);         
        
    for( int ncol = 0; ncol < n; ++ncol ) {
      float bestRow = 0;
      float maxE = -HUGE_VALF; 
      for( int row = 0; row < h; ++row ){
        int idx = SUB2IND_2D_TORCH(col[ncol], row, w, h);
        if( maxE < E[idx] )
        {
          maxE = E[idx];
          bestRow = row;
        }
      }
      matchIdx[ncol] = bestRow;
      matchE[ncol] = maxE;
    }
     
  return 0;
}

int collect(lua_State *L)
{
    // input
    THFloatTensor *mat_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *val_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *col_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *row_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    
    int n = THFloatTensor_size(row_, 0);
    int w = THFloatTensor_size(mat_, 1);
    int h = THFloatTensor_size(mat_, 0);

    float *mat = THFloatTensor_data(mat_);         
    float *val = THFloatTensor_data(val_);              
    float *col = THFloatTensor_data(col_);         
    float *row = THFloatTensor_data(row_);              
    
    for( int i = 0; i < n; ++i ) {
      int idx = SUB2IND_2D_TORCH(col[i], row[i], w, h);
      mat[idx] = val[i];
    }
     
  return 0;
}


/*
  This function masks enegy that corresponds to optimal path and its neighbourhood. 
*/

int maskE(lua_State *L)
{
    // input
    THFloatTensor *path_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    float distMin = luaL_checknumber(L, 3);
    
    int w = THFloatTensor_size(path_, 1);
    int h = THFloatTensor_size(path_, 0);
        
    float *path = THFloatTensor_data(path_);                // Energy
    float *E = THFloatTensor_data(E_);                // Energy
    
    int idx = 0;
    for( int y = 0; y < h; ++y )
    for( int x = 0; x < w; ++x ){ 
        
        if( path[idx] == 1 ) {
          
          for( int dy = -distMin; dy <= distMin; ++dy )
          for( int dx = -distMin; dx <= distMin; ++dx ){  
            
            int xx = x + dx;
            int yy = y + dy;
            
            if( xx >= 0 && yy >= 0 && xx < w && yy < h )
            {
              int idxN = SUB2IND_2D_TORCH(xx, yy, w, h);
              E[idxN] = -HUGE_VALF;
            }
          }
        
        }
        
        ++idx;
    }
    
  return 0;
}


/*
  Given optima path function finds nonoccluded path
*/

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
    THFloatTensor *aE_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *aS_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    THFloatTensor *traceBack_ = (THFloatTensor*)luaT_checkudata(L, 5, "torch.FloatTensor");
    
    // output
    THFloatTensor *pathOpt_  = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    
    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
   
    float *E = THFloatTensor_data(E_);                // Energy
    float *pathOpt = THFloatTensor_data(pathOpt_);          // Optimal path
    float *aE = THFloatTensor_data(aE_);
    float *aS = THFloatTensor_data(aS_);
    float *traceBack = THFloatTensor_data(traceBack_);
    float *mask = new float[w*h];    
        
    //for( int i = 0; i < h*w; ++i ) 
    //if( E[i] == 0 ) mask[i] = 0; else mask[i] = 1;
    
    accumulate(E, E, aE, aS, traceBack, w, h);
    trace(aE, aS,traceBack, pathOpt, w, h);
        
    delete[] mask;    
   return 0;
}

static const struct luaL_Reg funcs[] = {
    {"compute", compute},
    {"findNonoccPath", findNonoccPath},
    {"maskE", maskE},
    {"collect", collect},
    {"findMaxForCols", findMaxForCols},
    {"findMaxForRows", findMaxForRows},
    {NULL, NULL}
};

extern "C" int luaopen_libdprog(lua_State *L) {
    luaL_openlib(L, "dprog", funcs, 0);
    return 1;
}
