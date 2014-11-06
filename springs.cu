/*
 * Author: Kasjan Siwek
 * 
 * Application simulates NxN masses connected by springs. At time 0 we place
 * M charges in the system. Each charge causes nearby masses (those that
 * are in radius R_m from the charge) to instantly travel to the middle
 * of said charge. Those masses then stay there infinetely. We then 
 * simulate how the rest of the system behaves till it stops (due to
 * friction).
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <device_functions.h>

#include <iostream>
#include <cfloat>
#include <stdio.h>

#define BLOCK_SIZE 16

#define GAMMA 0.999
#define DT 0.01
#define EPSILON 1e-3

// Simulates one step of simulation
__global__ void simulate(int N, float K, float *vel_x, float *vel_y, float *pos_x, float *pos_y, float *b_vel_x, float *b_vel_y, float *b_pos_x, float *b_pos_y, int * last_ch) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	float f_x = 0;
	float f_y = 0;
	
	if (row < N - 1 && col < N - 1
			&& row != 0 && col != 0
			&& vel_x[row * N + col]!= FLT_MAX
			&& vel_y[row * N + col] != FLT_MAX) {

		float p_xl = pos_x[(row - 1) * N + col];
		float p_xb = pos_x[row * N + col - 1];
		float p_x = pos_x[row * N + col];
		float p_xt = pos_x[row * N + col + 1];
		float p_xr = pos_x[(row + 1) * N + col];
		
		float p_yl = pos_y[(row - 1) * N + col];
		float p_yb = pos_y[row * N + col - 1];
		float p_y = pos_y[row * N + col];
		float p_yt = pos_y[row * N + col + 1];
		float p_yr = pos_y[(row + 1) * N + col];
		
		float v_x = vel_x[row * N + col];

		float v_y = vel_y[row * N + col];

		b_pos_x[row * N + col] = p_x + v_x * DT;
		b_pos_y[row * N + col] = p_y + v_y * DT;

		f_x = K * (p_xl + p_xr + p_xt + p_xb - 4 * p_x);
		f_y = K * (p_yl + p_yr + p_yt + p_yb - 4 * p_y);

		b_vel_x[row * N + col] = v_x * GAMMA + f_x * DT;
		b_vel_y[row * N + col] = v_y * GAMMA + f_y * DT;
	}
	
	if ((abs(f_x) > EPSILON || abs(f_y) > EPSILON)) {
		*last_ch += 1;
	}
}

int main() {
	cudaError_t err = cudaSuccess;

	int N, M;
	float K;
	std::cin >> N >> M >> K;

	int numOfElements = N * N;
	size_t size = numOfElements * sizeof(float);

	// Malloc on host
	float *h_velocity_x = (float *)malloc(size);
	float *h_velocity_y = (float *)malloc(size);
	float *h_position_x = (float *)malloc(size);
	float *h_position_y = (float *)malloc(size);

	// Initialization
	for (int i = 0; i < numOfElements; ++i) {
		int x = i / N;
		int y = i % N;
		h_position_x[i] = x;
		h_position_y[i] = y;
		h_velocity_x[i] = 0;
		h_velocity_y[i] = 0;
        // I assume that neither mass is gonna reach FLT_MAX velocity,
        // so I save that value for static masses
		if (x == 0 || y == 0) {
			h_velocity_x[i] = FLT_MAX;
			h_velocity_y[i] = FLT_MAX;
		}
	}

	// Initializing charges
	for (int k = 0; k < M; ++k) {
		float X, Y, R;
		std::cin >> X >> Y >> R;
		for (int i = 0; i < numOfElements; ++i) {
				int x = i / N;
				int y = i % N;
				if ( (X-x)*(X-x) + (Y-y)*(Y-y) <= R*R ) {
					h_position_x[i] = X;
					h_position_y[i] = Y;
					h_velocity_x[i] = FLT_MAX;
					h_velocity_y[i] = FLT_MAX;
				}
			}
	}

	// Device malloc
	float *d_velocity_x = NULL;
    cudaMalloc((void **)&d_velocity_x, size);
	float *d_velocity_y = NULL;
	cudaMalloc((void **)&d_velocity_y, size);
	float *d_position_x = NULL;
	cudaMalloc((void **)&d_position_x, size);
	float *d_position_y = NULL;
	cudaMalloc((void **)&d_position_y, size);

	float *d_velocity_x2 = NULL;
    cudaMalloc((void **)&d_velocity_x2, size);
	float *d_velocity_y2 = NULL;
	cudaMalloc((void **)&d_velocity_y2, size);
	float *d_position_x2 = NULL;
	cudaMalloc((void **)&d_position_x2, size);
	float *d_position_y2 = NULL;
	cudaMalloc((void **)&d_position_y2, size);

	int *counter = NULL;
	cudaMalloc((void **)&counter, sizeof(int));

	// Initialize eps counter
	int aux_cnt = 1;
	cudaMemcpy(counter, &aux_cnt, sizeof(int), cudaMemcpyHostToDevice);

	// Copy data to device
	cudaMemcpy(d_velocity_x, h_velocity_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocity_y, h_velocity_y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_position_x, h_position_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_position_y, h_position_y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocity_x2, h_velocity_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocity_y2, h_velocity_y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_position_x2, h_position_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_position_y2, h_position_y, size, cudaMemcpyHostToDevice);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, 2 * BLOCK_SIZE);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

	bool parity = true;
	float *vel_x, *b_vel_x, *vel_y, *b_vel_y, *pos_x, *b_pos_x, *pos_y, *b_pos_y;
	int change;
	int previous = 0;
	int last_change;
	do {

		vel_x = parity ? d_velocity_x : d_velocity_x2;
		vel_y = parity ? d_velocity_y : d_velocity_y2;
		pos_x = parity ? d_position_x : d_position_x2;
		pos_y = parity ? d_position_y : d_position_y2;

		b_vel_x = parity ? d_velocity_x2 : d_velocity_x;
		b_vel_y = parity ? d_velocity_y2 : d_velocity_y;
		b_pos_x = parity ? d_position_x2 : d_position_x;
		b_pos_y = parity ? d_position_y2 : d_position_y;

		parity = parity ? false : true;

		simulate<<<dimGrid, dimBlock>>>(N, K, vel_x, vel_y, pos_x, pos_y, b_vel_x, b_vel_y, b_pos_x, b_pos_y, counter);

		cudaMemcpy(&last_change, counter, sizeof(int), cudaMemcpyDeviceToHost);

		change = last_change - previous;
		previous = last_change;
	} while (change);

	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to launch simulate kernel (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	
	// Copy data back to host
	cudaMemcpy(h_velocity_x, parity ? d_velocity_x2 : d_velocity_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_velocity_y, parity ? d_velocity_y2 : d_velocity_y, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_position_x, parity ? d_position_x2 : d_position_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_position_y, parity ? d_position_y2 : d_position_y, size, cudaMemcpyDeviceToHost);

	// Free memory on device
	cudaFree(d_velocity_x);
	cudaFree(d_velocity_y);
	cudaFree(d_position_x);
	cudaFree(d_position_y);
	cudaFree(d_velocity_x2);
	cudaFree(d_velocity_y2);
	cudaFree(d_position_x2);
	cudaFree(d_position_y2);
	cudaFree(counter);

	// Print data
	for (int i = 0; i < numOfElements; ++i) {
		std::cout << h_position_x[i] << " " << h_position_y[i] << std::endl;
	}

	// Free memory on host
	free(h_position_x);
	free(h_position_y);
	free(h_velocity_x);
	free(h_velocity_y);

	return 0;
}
