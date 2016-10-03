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

void accumulate(const float *E, float *pathE, float *pathLen, float *traceBack, int w, int h)
{

  /*

  Our goal is to find maximum average energy path from left to bottom column.
  We start from left col of energy array and continue to the right column.
  Only steps down, down-right and right are allowed from top to bottom.
  Steps are also constraint by the mask (steps are only allowed were mask is 1)
  Average energy of the path is SUM(Energy) / SUM(Steps).
  
  _E_     - energy of each cell  
  _pathE_  - energy of optimal path to the cell. 	
  _pathLen_  - length of optimal path to cell.
  _traceBack_  - index of previous cell 
  
  */
  
  int dx[3] = { 0, -1, -1};
  int dy[3] = {-1,  0, -1};
  int nb_neig = 3;
  
  // start from first column
  for( int nrow = 0; nrow < h; ++nrow )
  {
    int idx = SUB2IND_2D_TORCH(0, nrow, w, h);
    pathE[idx] = E[idx];
    pathLen[idx] = 1;
    traceBack[idx] = -1;  // mark begining
  }
  
  // go from left to right column
  for( int ncol = 1; ncol < h; ++ncol) 
  {
    int curIdx, prevIdx;
    
    // to first cell in col we could arrive only from left
    curIdx = SUB2IND_2D_TORCH(ncol, 0, w, h);
    prevIdx = SUB2IND_2D_TORCH(ncol-1, 0, w, h);
    
    pathE[curIdx] = pathE[prevIdx] + E[prevIdx];
    pathLen[curIdx]++;
    
    // to other cells we can arrive from left, top or left top 
    for( int nrow = 1; nrow < w; ++nrow )
    {
      curIdx = SUB2IND_2D_TORCH(ncol, nrow, w, h);
    	
      float bestPathAvgE =  0;
      float bestPathLen =  0;
      float bestPathE =  0;
      int   bestPathTraceBack = -1;
      
      for( int nneig = 0; nneig < nb_neig; ++nneig )
    	{
      
        prevIdx = SUB2IND_2D_TORCH(ncol + dx[nneig], nrow + dy[nneig], w, h);
        
        float curPathE;
        float curPathLen;

          curPathE   = pathE[prevIdx] + E[curIdx];
          curPathLen = pathLen[prevIdx] + 1;
    
        float curPathAvgE = (curPathE) / (curPathLen + 1e-10);
                
        if( bestPathTraceBack == -1 || bestPathAvgE <= curPathAvgE ) 
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
      mat[idx] += val[i];
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

  //  std::cout << "distMin " << distMin << "\n";   
        
    float *path = THFloatTensor_data(path_);                // Energy
    float *E = THFloatTensor_data(E_);                // Energy
    
    for( int y = 0; y < h; ++y )
    for( int x = 0; x < w; ++x ){ 
        
        int idx = SUB2IND_2D_TORCH(x, y, w, h);
        
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
    
    int w = THFloatTensor_size(path_, 1);
    int h = THFloatTensor_size(path_, 0);

    float *path = THFloatTensor_data(path_);                // Energy
    float *pathNonOcc = THFloatTensor_data(pathNonOcc_);                // Energy
        
     int *sumCol = new int[w];
     int *sumRow = new int[h];
     
     for( int nrow = 0; nrow < h; ++nrow ) sumRow[nrow] = 0;
     for( int ncol = 0; ncol < w; ++ncol ) sumCol[ncol] = 0;
          
     for( int nrow = 0; nrow < h; ++nrow )
     for( int ncol = 0; ncol < w; ++ncol ){ 
        int idx = SUB2IND_2D_TORCH(ncol, nrow, w, h);
        sumCol[ncol] += path[idx];
        sumRow[nrow] += path[idx];
     }
        
     // mark as occluded 
     for( int nrow = 0; nrow < h; ++nrow )
     for( int ncol = 0; ncol < w; ++ncol ){ 
       int idx = SUB2IND_2D_TORCH(ncol, nrow, w, h);
       pathNonOcc[idx] = path[idx];
       if( path[idx] == 1 )
       {
          if( sumCol[ncol] > 1 || sumRow[nrow] > 1 )
          {
            pathNonOcc[idx] = 0;
          }
        }
     }
     
  delete[] sumCol;
  delete[] sumRow;
    
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
    
    // output
    THFloatTensor *pathOpt_  = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    
    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
   
    float *E = THFloatTensor_data(E_);                // Energy
    float *pathOpt = THFloatTensor_data(pathOpt_);          // Optimal path
    float *aE = THFloatTensor_data(aE_);
    float *aS = THFloatTensor_data(aS_);
    
    float *traceBack = new float[h*w];
        
    accumulate(E, aE, aS, traceBack, w, h);
    trace(aE, aS,traceBack, pathOpt, w, h);
        
 //   delete[] aE;
    delete[] traceBack;
   
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
