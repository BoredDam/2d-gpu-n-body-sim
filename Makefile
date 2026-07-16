all: barnes_hut

barnes_hut: ./src/barnes_hut.c
	gcc ./src/barnes_hut.c -lOpenCL -lm -o barnes_hut.out
	
clean:
	rm ./src/barnes_hut.out
