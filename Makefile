all: naive_parallel

naive_parallel: ./src/naive_parallel.c
	gcc ./src/naive_parallel.c -lOpenCL -lm -o naive_parallel.out
	
clean:
	rm ./src/naive_parallel.c
