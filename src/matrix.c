//#include <stdio.h>
//#include <stdlib.h>
#include "main.h"
/*
 * Matrix operations
 */

//-------------------Dot product--------------------------------------------
//float** dotprod( float **m1, int m1_r, int m1_c, float **m2, int m2_r, int m2_c)
//{
//  /*
//   * Dot product between matrices/vectors of size (row r, column c)
//   */
//  // Allocate rows for the output matrix/vector...
//  float** res = (float **)malloc(m1_r * sizeof(float *));
//  // for each row allocate columns...
//  for( int row=0; row<m1_r; row++ ){
//    res[row] = (float *)malloc(m2_c * sizeof(float));
//  }

//  for(int row1=0; row1<m1_r; row1++){          // Row of first matrix 
//    for(int col2=0; col2<m2_c; col2++){     // columns of the second matrix 
//      res[row1][col2] = 0;
//      for(int col1=0; col1<m1_c; col1++){   // Column of the first matrix  
//        res[row1][col2] += m1[row1][col1] * m2[col1][col2];
//      }
//    }
//  }
//  return res;
//}
//---------------------------------------------------------------------------

//------------------------Dot Produc between two square matrices-------------
float* dotM2M( float* aM, float* bM )
{
  /*
   * Matrix dot product between the qudratic matrix a and quadratic matrix b
   */
  // Allocate DIM*DIM rows, each row is a pointer to float...
  float* res = (float *)malloc(DIM * DIM * sizeof(float));

  for( int i =0; i< DIM; ++i ){              // Row of first matrix
    for ( int j=0; j< DIM; ++j ) {           // column of second matrix
      *(res + i * DIM + j) = 0;                  //initialize the out-matrix location
      for ( int k=0; k< DIM; ++k ) {         // column of first matrix
        *(res + i * DIM + j)  += *(aM + i * DIM + k) * *(bM + k * DIM + j);
      }
    }
  }
  return res;
}
//--------------------------------------------------------------------------
//
//----------Dot product between square matrix and a column vector-----------
 
float* dotM2V( float* aM, float* vV )
{
  /*
   * Dot product between matrix a and column vector v
   */
  // Allocate DIM rows for the resulting vector...
  float*  res = (float *)malloc(DIM * sizeof(float));

  for( int i = 0; i<DIM; ++i )            // Row of the a matrix
  {
      *(res + i) = 0;                    // initialize the output vector
      for( int j = 0; j<DIM; ++j )         // rows of the v vector
      {
        *(res + i) += *(aM + i * DIM + j) * *(vV + j);
      }
  }
  return res;
}
//--------------------------------------------------------------------------
//
//-----------Dot product between square matrix and a scalar-----------------
float* dotM2S( float* aM, float sS )
{
  /*
   * Multiply Matrix with a scalar
   */
  // Allocate DIM*DIM rows, each 
  float* res = (float *)malloc(DIM * DIM * sizeof(float));

  for( int i = 0; i<DIM; ++i )       // rows of the matrix 
  {
    for( int j = 0; j<DIM; ++j)      // cols of the matrix 
    {
      *(res + i * DIM + j) = *(aM + i * DIM + j) * sS;
    }
  }
  return res;
}
//-------------------------------------------------------------------------
//
//------------------Row a vector dotproduct with col b vector--------------
//--------------------with scalar output-----------------------------------
float dotV2V_S( float* v1V, float* v2V )
{
  /* 
   * Dotproduct betweem row vector and col vector with a scalar result
   */
  float res = 0;
  for( int i=0; i<DIM; i++)
  {
    res += *(v1V + i) * *(v2V + i);
  }
  return res;
}
//-------------------------------------------------------------------------
//
//------------------Col A vector dotproduct with row B vector--------------
//--------------------with a matrix as output------------------------------
float* dotV2V_M( float* v1V, float* v2V )
{
  /* 
   * Dotproduct between a column vector and a row vector which results in a matrix 
   */
  // allocate DIM*DIM rows for the output matrix 
  float* res = (float *)malloc(DIM * DIM * sizeof(float));

  for( int row=0; row<DIM; row++ ){
    for( int col=0; col<DIM; col++ ){
      *(res + row * DIM + col) = *(v1V + row) * *(v2V + col);
    }
  }
  return res;
}
//------------------Product vector with scalar-----------------------------
float* dotV2S( float* vV, float sS )
{
  /* 
   * mutiply a vector with a scalar 
   */
  // allocate DIM rows for the output vector 
  float* res = (float *)malloc(DIM * sizeof(float));

  for( int i = 0; i<DIM; ++i )
  {
    *(res + i) = *(vV + i) * sS;
  }
  return res;
}
//-------------------Add vectors-------------------------------------------
float* addV2V( float* v1V, float* v2V )
{
  /*
   * Addition of two vectors
   */
  // Allocate DIM rows for the result vector
  float* res = (float *)malloc(DIM * sizeof(float));

  for( int i = 0; i<DIM; ++i )
  {
    *(res + i) = *(v1V + i) + *(v2V + i);
  }
  return res;
}
//-------------------------------------------------------------------------
//
//-----------------Add Matrix----------------------------------------------
float* addM2M( float* m1M, float* m2M )
{
  /* 
   * Addition of two quadratic matrices
   */ 
  // Allocate DIM rows for the result matrix 
  float* res = (float *)malloc(DIM * DIM * sizeof(float));

  for( int i=0; i<DIM; ++i ){
    for( int j=0; j<DIM; ++j ){
      *(res + i * DIM + j) = *(m1M + i * DIM + j) + *(m2M + i * DIM + j);
    }
  }
  return res;
}
//------------------------------------------------------------------------
//
//-------------------Subtract one matrix from another matrix--------------
float* subsM2M( float* m1M, float* m2M )
{
  /*
   * Substract one Matrix from another matrix 
   */
  // allocate DIM*DIM rows for the result matrix 
  float* res = (float *)malloc(DIM * DIM * sizeof(float));

  for( int row=0; row<DIM; row++){
    for( int col=0; col<DIM; col++){
      *(res + row * DIM + col) = *(m1M + row * DIM + col) - *(m2M + row * DIM + col);
    }
  } 
  return res;
}
//-------------------Transpose Square Matrix-------------------------------
float* MatT( float* aM )
{
  /*
   * Transpose a square matrix
   */
  // 1. Allocate DIM*DIM rows for the transposed matrix
  float* res = (float *)malloc(DIM * DIM * sizeof(float));

  for( int i = 0; i<DIM; i++ ){
    for( int j = 0; j<DIM; j++ ){
      *(res + i * DIM + j) = *(aM + j * DIM + i);
    }
  }
  return res;
}
//-------------------------------------------------------------------------
//
//--------------------Initialize matrix ZEROS-------------------------------
void zeros( float m[DIM][DIM] )
{
  for( int i = 0; i<DIM; ++i )
  {
    for( int j = 0; j<DIM; ++j )
    {
      m[i][j] = 0;
    }
  }
}
//-------------------------------------------------------------------------
//
//---------------Eye Matrix initializaton----------------------------------
void eye( float e[DIM][DIM] )
{
  for( int i = 0; i<DIM; ++i )
  {
    for( int j = 0; j<DIM; ++j )
    {
      if( i == j ) e[i][j] = 1;
      else         e[i][j] = 0;
    }
  }
}

