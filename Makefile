%:
	nvcc -o ./solution$@/main ./solution$@/main.cu
	./solution$@/main