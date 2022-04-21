LDFLAGS = -lgsl -lgslcblas -lm

a: test.o inference.o
	gcc test.o inference.o -o a $(LDFLAGS)

test.o: test.c inference.h
	gcc -c test.c inference.h

inference.o: inference.c inference.h
	gcc -c inference.c inference.h

clean:
	-rm -f test.o inference.o inference.h.gch
