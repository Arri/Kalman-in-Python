/*
 * Kalman Filter
 * # Based on: https://github.com/zziz/kalman-filter and wickipedia (https://en.wikipedia.org/wiki/Kalman_filter)
 * # ################################################################################################################
 * # Fk, the state-transition model;
 * # Hk, the observation model;
 * # Qk, the covariance of the process noise;
 * # Rk, the covariance of the observation noise;
 * # and sometimes Bk, the control-input model, for each time-step, k, as described below.
 * # ################################################################################################################
 *
 * Author: Arasch U. Lagies
 * First Version: 1/9/2020
 * Last Update: 1/9/2020
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"

const char* paramfile = "/home/pi/kalman/kalman/data/parameters.txt";  
const char* fileName  = "/home/pi/kalman/accelerometer/python-lis3dh-master/datadump.txt";   //"/home/pi/Projects/kalman/data/random.txt";
const char* resultFile = "/home/pi/kalman/kalman/data/resultFile.csv";
const int xCol = 1;     // Test data x-values read from column...
const int yCol = 2;     // Test data y-values read from column...
const int zCol = 3;     // Test data z-values read from column...

/* Kalman Start-Parameters */
float F[DIM][DIM] = { {1,dt,dt}, {dt,1,dt}, {dt,dt,1} };                             // The state transition model
float H[DIM]      = { 1,1,1 };                                                   // The observation model (maps tue space into the observed sapace)
float B[DIM]      = { 0,0,0 };                                                   // The control-input model (=0 here ==> the applied model is velocity=const.)
float Q[DIM][DIM] = { {0.05, 0.05, 0.05}, {0.05, 0.05, 0.05}, {0.05, 0.05, 0.05} };   // The covariance of the process noise
float R[1]        = { 0.5 };                                                     // The covariance of the observation noise
float P[DIM][DIM] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };                         // A posteriori error covariance matrix (a measure of the estimated accuracy of the state estimate)
float x[DIM]      = { 1.0,0,0 };                                                 // prediction x-coordinate
float y[DIM]      = { 0,1.0,0 };                                                 // prediction y-coordinate
float z[DIM]      = { 0,0,1.0 };                                                 // prediction z-coordinate
float I[DIM][DIM] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };                         // Eye matrix for update calculation of the covariance matrix

/* Function for reading test file data */
const char* getfield(char* line, int num)
{
  const char* tok;
  for (tok = strtok(line, "\t");
    tok && *tok && tok!=NULL;
    tok = strtok(NULL, "\t\n"))
  {
  if (!--num)  
    return tok;
  }
  return NULL;
}


int main()
{
  char apply[sizeof(char)];
  char line[1024];
  float pred_xVal, pred_yVal, pred_zVal;
  /* Pointer for the parameters */
  float* x_pt = (float*)malloc(DIM * sizeof(float));
  float* y_pt = (float*)malloc(DIM * sizeof(float));
  float* z_pt = (float*)malloc(DIM * sizeof(float));
  float* p_pt = (float*)malloc(DIM * sizeof(float));
  float* i_pt = (float*)malloc(DIM * DIM * sizeof(float));
  float* r_pt = (float*)malloc(1 * sizeof(float));
  float* q_pt = (float*)malloc(DIM * DIM * sizeof(float));
  float* b_pt = (float*)malloc(DIM * sizeof(float));
  float* h_pt = (float*)malloc(DIM * sizeof(float));
  float* f_pt = (float*)malloc(DIM * DIM * sizeof(float));

  /* Check if a file with Kalman Parameters exists */
  FILE* params = fopen(paramfile, "r");      // File containing parameters trained in previous run
  FILE* results = fopen(resultFile, "w");    // File containing results -- measured values vs kalman predictions
  if (params != NULL) {
    printf("[INFO] A paramater file with learned parameters was found.\n");
    printf(" Do you want to use those parameters?\n");
    scanf("%s", apply);
    printf("typed in %s\n", apply);
  }
  if ( params != NULL && apply[0] == 'y' ) {
    readMatrix( DIM, DIM, f_pt, params );   // Read trained values for F 
    readVector( DIM, h_pt, params );        // Read trained values for H
    readVector( DIM, b_pt, params );        // Read trained values for B
    readMatrix( DIM, DIM, q_pt, params );   // Read trained values for Q
    readVector( 1, r_pt , params );         // Read trained values for R  
    readMatrix( DIM, DIM, p_pt, params );   // Read trained values for P
    readVector( DIM, x_pt, params );        // Read trained values for x
    readVector( DIM, y_pt, params );        // Read trained values for y
    readVector( DIM, z_pt, params );        // Read trained values for z
    readMatrix( DIM, DIM, i_pt, params );   // Read trained values for I
  } else {
    printf("[INFO] No paremeter file found...\n");
    x_pt = &x[0];
    y_pt = &y[0];
    z_pt = &z[0];
    p_pt = &P[0][0];
    i_pt = &I[0][0];
    r_pt = &R[0];
    q_pt = &Q[0][0];
    b_pt = &B[0];
    h_pt = &H[0];
    f_pt = &F[0][0];
  }
  fclose(params);

  /* Test Data */
	FILE* stream = fopen(fileName, "r");
	if (stream == NULL) {
		printf("[ERROR] Could not find the data file. Please fix the path...!");
		return -1;
	} else {
    printf("[INFO] Found a data-file for testing...\n");
  }

  
  // Loop through data...
  while (fgets(line, 1024, stream))
  {
    char* tmp1 = strdup(line);
    char* tmp2 = strdup(line);
    char* tmp3 = strdup(line);
    float xVal = (float)strtod(getfield(tmp1, xCol),NULL);
    float yVal = (float)strtod(getfield(tmp2, yCol),NULL);
    float zVal = (float)strtod(getfield(tmp3, zCol),NULL);

    //===============================================================
    // 1. Kalman prediction ...
    predict( 0, 0, 0, x_pt, y_pt, z_pt, f_pt, b_pt, p_pt, q_pt );
    //===============================================================

    pred_xVal = dotV2V_S( h_pt, x_pt );
    pred_yVal = dotV2V_S( h_pt, y_pt );
    pred_zVal = dotV2V_S( h_pt, z_pt );
    printf("[INFO] The Measured values are (x = %f, y = %f, z = %f), The predicted values by the Kalman filter are (x = %f, y = %f, z = %f). \n", 
				xVal, yVal, zVal, pred_xVal, pred_yVal, pred_zVal ); 
    fprintf(results, "%f, %f, %f, %f, %f, %f \n", xVal, yVal, zVal, pred_xVal, pred_yVal, pred_zVal );

    //===============================================================
    // 2. Update kalman parameters...
    update( xVal, yVal, zVal, x_pt, y_pt, z_pt, p_pt, h_pt, r_pt, i_pt );  
    //===============================================================
    free(tmp1);
    free(tmp2);
    free(tmp3);
  }
  fclose(results);

  printf("FINAL... \n");
  for(int row=0; row<DIM; row++) {
    printf("x(%d) = %f, y(%d) = %f, z(%d) = %f\n", row, x_pt[row], row, y_pt[row], row, z_pt[row]);
    for(int col=0; col<DIM; col++) {
      printf("P(%d, %d) = %f\n", row, col, *(p_pt + row*DIM + col));
    }
  }

  // Close the test data file... 
  fclose(stream);

  printf("Closing\n");

  /* Write trained kalman parameter into file for later use */
  params = fopen(paramfile, "r");
  if( params != NULL ) {
    printf("Do you want to overwrite existig kalman parameter file with the new parameter?");
    scanf("%s", apply);
    fclose(params);
    if( apply[0] == 'y' ){
      params = fopen(paramfile, "w");
      writeMatrix( DIM, DIM, f_pt, params );   // write trained values for F 
      writeVector( DIM, h_pt, params );        // write trained values for H 
      writeVector( DIM, b_pt, params );        // write trained values for B  
      writeMatrix( DIM, DIM, q_pt, params );   // write trained values for Q 
      writeVector( 1, r_pt, params );          // write trained values for R 
      writeMatrix( DIM, DIM, p_pt, params );   // write trained values for P  
      writeVector( DIM, x_pt, params );        // write trained values for x  
      writeVector( DIM, y_pt, params );        // write trained values for y  
      writeMatrix( DIM, DIM, i_pt, params );   // write trained values for I 
    }
  }
  fclose(params);

  return 0;
}




