HOGextractor
============

Credit goes to Pedro Felzenswalb for his initial HOG.

The best version is located in the features_lookup_madmex.cc (and associated files).  This takes advantage of both intel intrinsics, and lookup speedups. An example matlab file that uses this is madmex.m.
These files should compile fairly straight forwardly to be used with matlab.  The makefile should make it clear.


There is excess code laying about.  Hopefully it is clear enough to sort through.
