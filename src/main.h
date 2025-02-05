// Kalman Header File:
#ifndef KALMAN_H
#define KALMAN_H

#define DIM 3       // Dimensions...
#define dt (1.0/60)   // Timestep...

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Utility functions */
const char* getfiled(char*, int);                           // Helper function to read (x,y) values from file...
void readMatrix( int, int, float*, FILE* );
void readVector( int, float*, FILE* );
void writeMatrix( int, int, float*, FILE* );
void writeVector( int, float*, FILE* );

/* Matrix operations */
//extern float** dotprod( float**, int, int, float**, int, int);

extern float*  dotM2M  ( float*, float* );  // Helper function for matrix dot product
extern float*  dotM2V  ( float*, float* );  // Matrix to vector dot product
extern float*  dotM2S  ( float*, float  );  // matrix with scalar multiplication
extern float*  dotV2S  ( float*, float  );  // Multiply vector with scalar 
extern float   dotV2V_S( float*, float* );  // Dot product between a row vector and a column vector with a scalar result 
extern float*  dotV2V_M( float*, float* );  // Dot product betweem a col vectoe amd a roe vector resultig in a matrix
extern float*  MatT    ( float*         );  // Matrix transpose
extern float*  addV2V  ( float*, float* );  // Vector with vector additiom
extern float*  addM2M  ( float*, float* );  // Matrix with matrix addition
extern float*  subsM2M ( float*, float* );  // Substract one matrix from a second

/* Initialize Marix */
extern void zeros( float[DIM][DIM] );
extern void eye  ( float[DIM][DIM] );
 

/* Kalman filter functions */
extern void initKalman( void );
extern void predict(float, float, float, float*, float*, float*,
                    float*, float*, float*, float* );
extern void update(float, float, float, float*, float*, float*,
                  float*, float*, float*, float* );



#endif

