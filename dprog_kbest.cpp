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
#include <cassert>
#include "luaT.h"
#include <TH/TH.h>

#define I2D(col, row, width, height) I3D(col, row, 0, width, height, 1)
#define I3D(col, row, ch, width, height, nb_ch) ( ( (ch) * (height) + (row) ) * (width) + (col) ) 


std::vector<float> reorder(std::vector<float> x, std::vector<int> index){
  std::vector<float> y(x.size());
  for (int i = 0; i < x.size(); ++i) y[i] = x[index[i]];
  std::swap(x,y);
  return x;
}

std::vector<int> get_sorting_index(std::vector<float>x){

    // find sorting indices for x
  std::vector<int> index(x.size());
  for( int n = 0; n < x.size(); n++ ) index[n] = n;

  auto comparator = [&x](int a, int b){ return x[a] < x[b]; };
    std::sort(index.begin(), index.end(), comparator);

    return index;
}


bool isValid(int x, int y, int dim, int dispMax)
{
  if(x >= y) return false;
  else if( x < (y - dispMax) ) return false;
  else if( (x < 0) || x >= dim ) return false;
  else if( (y < 0) || y >= dim ) return false;
  else return true;
}

void* simulate_(float *E, int dim, int dispMax)
{
  std::fill_n(E, dim*dim, 0);

  std::random_device rd; 
  std::mt19937 eng(rd());

  int dx[3] = {1, 0, 1};
  int dy[3] = {0, 1, 1};
  int y = dispMax; 
  int xMin = y - dispMax;
  int xMax = y - 1;
  std::uniform_int_distribution<> distrX(xMin, xMax); 
  std::uniform_real_distribution<> distrNoise(0, 1); 
  int x = distrX(eng);

  for (int i = 0; i < dim*dim; ++i) E[i] += distrNoise(eng);

  E[I2D(x, y, dim, dim)] = 0;
  std::uniform_int_distribution<> distrN(0, 2); 
  while ( (x < dim-1) && (y < dim-1) ){
    int xx, yy;
    do {
      int n = distrN(eng);
      xx = x + dx[n];
      yy = y + dy[n];
    } while ( !isValid(xx, yy, dim, dispMax) );
    x = xx;
    y = yy;

    E[I2D(x, y, dim, dim)] = 0;
  }

 
  
}

void S3D(int i, int &x, int &y, int &ch, int width, int height, int nb_ch )
{
  int tmp = i / width;
  x = i % width;
  ch = tmp / height;
  y = tmp % height;
}

void trace(float *P, float *aE, float *T, int dim, int dispMax, int pathNum)
{  
  int x;
  int y = dim - 1;
  int xMin = y - dispMax;
  int xMax = y - 1;
  
  std::vector<float> vecE;
  std::vector<float> vecT;
  for( x = xMin; x <= xMax; ++x )
  {
    for (int k = 0; k < pathNum; ++k)
    {
      vecE.push_back( aE[I3D(x, y, k, dim, dim, pathNum)] );
      vecT.push_back( T[I3D(x, y, k, dim, dim, pathNum)] );
    }

  }

  std::vector<int> sort_index = get_sorting_index(vecE);
  vecT = reorder(vecT, sort_index);
  vecE = reorder(vecE, sort_index);

  for (int k = 0; k < pathNum; ++k)
  {
    int idx = vecT[k];
    while ( idx > 0 ){
      int ch;
      S3D(idx, x, y, ch, dim, dim, pathNum );
      P[ I3D(x, y, k, dim, dim, pathNum) ] = 1;
      idx = T[idx];
    }
    int ch;
    S3D(idx, x, y, ch, dim, dim, pathNum );
    P[ I3D(x, y, k, dim, dim, pathNum) ] = 1;
  }

}

void acc(const float *E, float *aE, float *T, int dim, int dispMax, int pathNum )
{
  int x, y, xx, yy, k, n;
  int dx[3] = {-1,  0, -1};
  int dy[3] = {0 , -1, -1};


  // initialization
  // we ignore first dispMax rows, since they might not have a match
  y = dispMax;
  int xMin = y - dispMax;
  int xMax = y - 1; 

  for( x = xMin; x <= xMax; ++x ){

    std::vector<float> vecE;
    for( k = 0; k < pathNum; ++k ){

      vecE.push_back( E[I2D(x, y, dim, dim)] );
      T[I3D(x, y, k, dim, dim, pathNum)] = -1.0; // begining
    }

    std::sort(vecE.begin(), vecE.end());
    for( k = 0; k < pathNum; ++k ) aE[I3D(x, y, k, dim, dim, pathNum)] = vecE[(k >= vecE.size())? vecE.size()-1 : k]; 
  
  }

  // process row by row
  for( y = dispMax+1; y < dim; ++y){

    xMin = y - dispMax;
    xMax = y - 1;

    for( x = xMin; x <= xMax; ++x){

      std::vector<float> vecE;
      std::vector<float> vecT;

      for( n = 0; n < 3; ++n){

        xx = x + dx[n];
        yy = y + dy[n];

        if (!isValid(xx, yy, dim, dispMax) ) continue;

        for( k = 0; k < pathNum; ++k){

          int trace_back_index = I3D(xx, yy, k, dim, dim, pathNum);
          vecE.push_back( aE[trace_back_index] + E[I2D(x, y, dim, dim)] );
          vecT.push_back( trace_back_index );

        }

      }

      std::vector<int> sort_index = get_sorting_index(vecE);
      vecT = reorder(vecT, sort_index);
      vecE = reorder(vecE, sort_index);

      for( k = 0; k < pathNum; ++k ){
        T[I3D(x, y, k, dim, dim, pathNum)] = vecT[(k >= vecT.size())? vecT.size()-1 : k]; 
        aE[I3D(x, y, k, dim, dim, pathNum)] = vecE[(k >= vecT.size())? vecT.size()-1 : k];  
      }  

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

int compute(lua_State *L)
{
    // input
    THFloatTensor *E_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *aE_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    THFloatTensor *T_ = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
    int dispMax = luaL_checkinteger(L, 5);

    // output
    THFloatTensor *P_  = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    
    int w = THFloatTensor_size(aE_, 2);
    int h = THFloatTensor_size(aE_, 1);
    int pathNum = THFloatTensor_size(aE_, 0);
   
    std::cout << "pathNum: " << pathNum << "\n";
    std::cout << "h: " << h << "\n";
    std::cout << "w: " << w << "\n";

    assert( h == w );
    int dim = h;

    float *E = THFloatTensor_data(E_);                // Energy
    float *aE = THFloatTensor_data(aE_);
    float *T = THFloatTensor_data(T_);

    float *P = THFloatTensor_data(P_);

    acc(E, aE, T, dim, dispMax, pathNum);
    trace(P, aE, T, dim, dispMax, pathNum);

   return 0;
}

int simulate(lua_State *L)
{
    // input
    int dispMax = luaL_checkinteger(L, 2);

    // output
    THFloatTensor *E_  = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    
    int w = THFloatTensor_size(E_, 1);
    int h = THFloatTensor_size(E_, 0);
    
    assert( h == w );
    int dim = h;

    float *E = THFloatTensor_data(E_);                // Energy
    
    simulate_(E, dim, dispMax);

   return 0;
}

static const struct luaL_Reg funcs[] = {
    {"simulate", simulate},
    {"compute", compute},
    {"findNonoccPath", findNonoccPath},
    {NULL, NULL}
};

extern "C" int luaopen_libdprog_kbest(lua_State *L) {
    luaL_openlib(L, "dprog_kbest", funcs, 0);
    return 1;
}
