

CXXFLAGS=-msse4.1 -std=c++0x
OBJS=features_pedro.mexa64 \
     features_pedro_float.mexa64 \
     features_madmex.mexa64


all:	$(OBJS)

run:	$(OBJS)
	matlab -nodisplay -nosplash -nodesktop -r "run('madmex.m');"

%.mexa64:	%.cc	vector_intrinsics.h
	mex CXXFLAGS='$$CXXFLAGS $(CXXFLAGS)' $<

clean:
	rm *.mexa64

