all: barnes_hut_cpu_test

barnes_hut: ./src/barnes_hut.c
	gcc ./src/barnes_hut.c -lOpenCL -lm -o barnes_hut.out

barnes_hut_test: ./src/barnes_hut_for_testing.c
	gcc ./src/barnes_hut_for_testing.c -lOpenCL -lm -o barnes_hut_for_testing.out
	
barnes_hut_cpu_test: ./src/barnes_hut_cpu.c
	gcc ./src/barnes_hut_cpu.c -lOpenCL -lm -o barnes_hut_cpu.out

naive_parallel: ./src/naive_parallel.c
	gcc ./src/naive_parallel.c -lOpenCL -lm -o naive_parallel.out

naive_parallel_test: ./src/naive_parallel_for_testing.c
	gcc ./src/naive_parallel_for_testing.c -lOpenCL -lm -o naive_parallel_for_testing.out