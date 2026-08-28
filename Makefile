all: barnes_hut_testv2_1

barnes_hut: ./src/barnes_hut.c
	gcc ./src/barnes_hut.c -lOpenCL -lm -o barnes_hut.out

barnes_hut_test: ./src/barnes_hut_for_testing.c
	gcc ./src/barnes_hut_for_testing.c -lOpenCL -lm -o barnes_hut_for_testing.out

barnes_hut_testv2: ./src/barnes_hut_for_testing_v2.c
	gcc ./src/barnes_hut_for_testing_v2.c -lOpenCL -lm -o barnes_hut_for_testing_v2.out

barnes_hut_testv2_1: ./src/barnes_hut_for_testing_v2_1.c
	gcc ./src/barnes_hut_for_testing_v2_1.c -lOpenCL -lm -o barnes_hut_for_testing_v2_1.out
	
barnes_hut_cpu_test: ./src/barnes_hut_cpu.c
	gcc ./src/barnes_hut_cpu.c -lOpenCL -lm -o barnes_hut_cpu.out

naive_parallel: ./src/naive_parallel.c
	gcc ./src/naive_parallel.c -lOpenCL -lm -o naive_parallel.out

naive_parallel_test: ./src/naive_parallel_for_testing.c
	gcc ./src/naive_parallel_for_testing.c -lOpenCL -lm -o naive_parallel_for_testing.out