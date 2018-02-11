/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"


////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: P = M * N
//! Mw is M's width and Nw is N's width
////////////////////////////////////////////////////////////////////////////////
    __global__ void
matrixMul( float* P, float* M, float* N, int Mw, int Nw)
{
    int bx = blockIdx.x;     int by = blockIdx.y;
    int tx = threadIdx.x;    int ty = threadIdx.y;
    __shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

    // ===================================================================
    // Code segment 1
    // Determine the update values for the tile indices in the loop
    // ===================================================================
	
	// Si el tamaño del bloque es menor que el tamaño de fila,
	// dividir el copiado de filas en bloques

	// Total de bloques en que se dividen filas y columnas
	int totalBloques = Mw / BLOCK_SIZE;

int sizeFila = gridDim.y * blockDim.y;
int fila = (blockDim.x * bx) + tx;
int columna = (blockDim.y * by) + ty;
int posicion = (fila * sizeFila) + columna;

int posicionFila = fila * sizeFila;
int posicionColumna = (blockDim.y * by) + ty;


	// Utilizado para depurar
	/*if (bx == 1 && by == 1) {
		printf("Numero de bloques por fila: %d, Numero Threads por bloque: %d, Threads en X: %d, Threads en Y: %d , fila = %d, columna: %d, posicion: %d\n", totalBloques, threadsPorBloque, blockDim.x, blockDim.y, posicionFila, columna, posicion);

	}*/


    // ===================================================================
    // Code segment 2
    // Do matrix-matrix multiplication inside a tile
    // ===================================================================
    int contadorBloque;
    float pSub = 0;

    // Cada fila y columna esta dividida en un numero de bloques de tamaño BLOCK_SIZE
    for (contadorBloque = 0; contadorBloque < totalBloques; ++contadorBloque) {

        // Load a tile from M and N into the shared memory arrays

	// Calcular la posicion de acceso a M (filas a desplazarse)
	int posicionM = posicionFila + (contadorBloque * BLOCK_SIZE) + ty;

	// Utilizado para depurar
	/*if ((bx == 0 && by == 0)) {
		printf("tx: %d, ty: %d, contadorBloque: %d, posicionM: %d\n", tx, ty, contadorBloque, posicionM);
	}*/

	// Calcular la posicion de acceso a N (columnas a desplazarse)
	int posicionN = posicionColumna + (tx * sizeFila) + (sizeFila * contadorBloque * BLOCK_SIZE);

	// Utilizado para depurar
	/*if ((bx == 0 && by == 0)) {
		printf("tx: %d, ty: %d, contadorBloque: %d, posicionN: %d\n", tx, ty, contadorBloque, posicionN);
	}*/

	Ms[tx][ty] = M[posicionM];
	Ns[tx][ty] = M[posicionN];

        // Synchronize the threads
	__syncthreads();
	
        // Multiply the two tiles together, each thread accumulating
        // the partial sum of a single dot product.
        for (int i = 0; i < BLOCK_SIZE; i++) { // bucle dado
		// En la iteracion se obtiene un valor parcial de la casilla
		pSub += Ms[tx][i] * Ns[i][ty];
        }

        // Synchronize again.
	__syncthreads();

	// Utilizado para depurar
	/*if ((bx == 1 && by == 1)) {
		printf("tx: %d, ty: %d, Suma: %f\n", tx, ty, pSub);
	}*/

    }

    // ===================================================================
    // Code segment 3
    // Store the data back to global memory
    // ===================================================================

	P[posicion] = pSub;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
