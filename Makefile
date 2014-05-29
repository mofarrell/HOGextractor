

CXXFLAGS=-msse4.1 -std=c++0x -O2 -funroll-all-loops -DDEBUG #-DDOUBLE#-DDEBUG#-march=native #-DDOUBLE#-DAVX -DDOUBLE -DDEBUG
OBJS=features_pedro.mexa64 \
     features_pedro_single.mexa64 \
     features_madmex.mexa64 \
     features_lookup.mexa64 \
     features_lookup_madmex.mexa64


all:	$(OBJS)

run:	$(OBJS)
	matlab -nojvm -nodisplay -nosplash -nodesktop -r "type=1;run('madmex.m');"

%.mexa64:	%.cc	vector_intrinsics.h	best_o_lookup.h Makefile
	mex CXXFLAGS='$$CXXFLAGS $(CXXFLAGS)' $<

clean:
	rm *.mexa64

