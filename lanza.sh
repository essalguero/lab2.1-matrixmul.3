#!/bin/sh

# Para decirle a SGE que sh es el shell del job
#$ -S /bin/sh

# Para decirle a SGE el nombre de nuestro trabajo 
#$ -N matrixMul_2

#Nos movemos al directorio actual
#$ -cwd

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

#Primero Compilamos
make 

#Comprobamos que el ejecutable existe
if [ ! -x ../../bin/linux/release/lab2.1-matrixmul ]; then
   echo "Error en compilacion"
   exit 1
fi

#Ahora lanzamos la ejecucion en SGE del ejecutable
####../../bin/linux/release/lab1.1-matrixmul 8
####../../bin/linux/release/lab1.1-matrixmul 128
####../../bin/linux/release/lab1.1-matrixmul 512
####../../bin/linux/release/lab1.1-matrixmul 3072
../../bin/linux/release/lab2.1-matrixmul 4096
