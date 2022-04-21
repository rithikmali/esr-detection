LDFLAGS = -lgsl -lgslcblas -lm -lsndfile -DUSE_KISS_FFT

a: test.o inference.o
	g++ test.o inference.o -o a $(LDFLAGS)

test.o: test.cpp inference.hpp
	g++ -c test.cpp inference.hpp

inference.o: inference.cpp inference.hpp
	g++ -c inference.cpp inference.hpp

clean:
	-rm -f test.o inference.o inference.h.gch
