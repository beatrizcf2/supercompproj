CC = g++
CFLAGS = -O3 -Wall -pedantic -std=c++11

heuristico: heuristico.o funcoes.o 
	$(CC) $(CFLAGS) -o heuristico heuristico.o funcoes.o

heuristico.o: heuristico.cpp funcoes.h
	$(CC) $(CFLAGS) -c heuristico.cpp

funcoes.o: funcoes.h funcoes.cpp
	$(CC) $(CFLAGS) -c funcoes.cpp