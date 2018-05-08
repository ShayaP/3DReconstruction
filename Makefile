CC= g++
CXX= g++

CXXFLAGS= -g -std=c++11
LDFLAGS= -g -std=c++11

default: test

test:
	g++ test.cpp -std=c++11 -o test `pkg-config --cflags --libs opencv`

clean: 
	rm -f test *.o *.gch
