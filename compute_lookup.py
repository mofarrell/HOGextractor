import sys

# unit vectors used to compute gradient orientation
uu = [1.0000, 
  0.9397, 
  0.7660, 
  0.500, 
  0.1736, 
  -0.1736, 
  -0.5000, 
  -0.7660, 
  -0.9397]
vv = [0.0000, 
  0.3420, 
  0.6428, 
  0.8660, 
  0.9848, 
  0.9848, 
  0.8660, 
  0.6428, 
  0.3420]

def get_best_o(size):
  ans = []
  for c in xrange(size):
    c = c - size/2
    row = []
    for r in xrange(size):
      r = r - size/2
      
      best_dot = 0
      best_o = 0
      for o in xrange(9):
        dot = (c/float(size/2))*uu[o]+(r/float(size/2))*vv[o]
        if (dot > best_dot):
          best_dot = dot
          best_o = o
        elif (-dot > best_dot):
          best_dot = -dot
          best_o = o + 9
      row.append(best_o)
    ans.append(row)
  return ans

def gen_header():
  if (len(sys.argv) < 3):
    print "Usage: python compute_lookup.py [size] [filename]"
    exit()
  size = int(sys.argv[1])-1
  filename = sys.argv[2]
  print "Generating lookup table of size", str(size)
  with open(filename, "wb") as f:
    f.write("// Generated orientation lookup table\n\n")

    f.write("#define LOOKUP_SIZE %d\n\n" % (size/2))
    
    f.write("static const char best_o_lookup[%d][%d] = {\n" % (size, size))

    for c in get_best_o(size):
      f.write("\t{\n")
      #for r in c:
      f.write(",".join([str(v) for v in c]))
      f.write("\t},\n")
    f.write("};\n")

gen_header()



