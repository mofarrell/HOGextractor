

CXXFLAGS=-msse4.1
OBJS=features_pedro.mexa64 features_madmex.mexa64


all:	$(OBJS)

run:	$(OBJS)
	matlab -nodisplay -nosplash -nodesktop -r "run('madmex.m');"

%.mexa64:	%.cc
	mex CXXFLAGS='$$CXXFLAGS $(CXXFLAGS)' $<

clean:
	rm *.mexa64

