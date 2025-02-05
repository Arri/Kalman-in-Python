/*************************************************************************
*   Utility Function for the Kalman Filter
*************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"



void readMatrix(int rows, int cols, float* matRead, FILE* file)
{
  /*
   * Function to read a matrix from file
   */
  float val=0;
  for(int i=0; i<rows; i++)
  {
    for(int j=0; j<cols; j++)
    {
      fscanf(file, "%f", &val);
      *(matRead + i * DIM + j) = val;
   }
  }
  return;
}

void readVector(int rows, float* vecRead, FILE* file)
{
  /*
   * Function to read a vector from file
   */
  float val;
  for(int i=0; i<rows; i++)
  {
    fscanf(file, "%f[^\n]\n", &(val));
    *(vecRead + i) = val;
    printf(" Read %f\n", val);
  }
  return;
}

void writeMatrix( int rows, int cols, float* matWrite, FILE* file )
{
  /*
   * Function to write matrix to file
   */
  for( int i = 0; i<rows; i++ )
  {
    for( int j = 0; j<cols; j++ )
    {
      fprintf(file, "%f ", *(matWrite + i * DIM + j));
    }
  }
  fprintf(file, "\n");
  return;

}

void writeVector( int rows, float* vecWrite, FILE* file )
{
  /*
   * Function to write vector to file 
   */
  for( int i = 0; i<rows; i++ )
  {
    fprintf(file, "%f ", *(vecWrite + i));
  }
  fprintf(file, "\n");
  return;
}
