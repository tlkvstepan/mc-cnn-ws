
#include <vector> 
#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <algorithm>


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

float* sim(int dim, int dispMax)
{
	float *arr = new float[dim*dim];
	std::fill_n(arr,dim*dim,0);

	std::random_device rd; 
    std::mt19937 eng(rd());

    int dx[3] = {1, 0, 1};
    int dy[3] = {0, 1, 1};
    int y = dispMax; 
	int xMin = y - dispMax;
	int xMax = y - 1;
    std::uniform_int_distribution<> distrX(xMin, xMax); 
	int x = distrX(eng);

	arr[I2D(x, y, dim, dim)] = 0;
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

		arr[I2D(x, y, dim, dim)] = 0;
	}
	return arr;
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
	int xMin = y - 1;
	int xMax = y - dispMax;

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

int main(){

	int dim = 1000;
	int dispMax = 100;
	int pathNum = 10;
	float *E = sim(dim, dispMax);
	float *aE = new float[dim*dim*pathNum];
	float *T = new float[dim*dim*pathNum];
	float *P = new float[dim*dim*pathNum];
	acc(E, aE, T, dim, dispMax, pathNum );
	trace(P, aE, T, dim, dispMax, pathNum );

	delete [] E;
	return 0;
}
