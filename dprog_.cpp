extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

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

void accumulate(const float *E, float *aE, float *aP, int w, int h)
{

  // this is analog of integral image for max operator for line:
  // maxVal[i] is maximum value in [0, i]
  // maxVal[i] is index value of the maximum value
  float *maxIdx = new float[w];
  float *maxVal = new float[w];

  // initialize top row of accumulated energy to energy
  for( int ncol = 0; ncol < w; ++ncol ){
    aE[SUB2IND_2D_TORCH(ncol, 0, w, h)] = E[SUB2IND_2D_TORCH(ncol, 0, w, h)];
    aP[SUB2IND_2D_TORCH(ncol, 0, w, h)] = 0;  
  }   
  
  // go from top row down, computing best accumulated energy in every position 
  for( int nrow = 1; nrow < h; ++nrow) 
  {
    
    maxIdx[0] = 0;
    maxVal[0] = aE[SUB2IND_2D_TORCH(0, nrow-1, w, h)];

    for( int ncol = 1; ncol < w; ++ncol )
    {
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
