// unit vectors used to compute gradient orientation
var uu = [1.0000, 
    0.9397, 
    0.7660, 
    0.500, 
    0.1736, 
    -0.1736, 
    -0.5000, 
    -0.7660, 
    -0.9397];
var vv = [0.0000, 
    0.3420, 
    0.6428, 
    0.8660, 
    0.9848, 
    0.9848, 
    0.8660, 
    0.6428, 
    0.3420];

var yb = [
-1,
-1,
-1,
0,
0,
0,
1,
1,
1
];
var xb = [
1,
1,
1,
-1,
-1,
-1,
1,
1,
1
];

function hog(width, height, sbin, frame, canvas) {
  for (var x = 1; x < width; x+=sbin) {
    for (var y = 1; y < height; y+=sbin) {
      // first color channel
      var dy = frame.data[4*((y+1)*width + x) + 0] - 
              frame.data[4*((y-1)*width + x) + 0];
      var dx = frame.data[4*(y*width + (x+1)) + 0] - 
              frame.data[4*(y*width + (x-1)) + 0];
      var v = dx*dx + dy*dy;

      // second color channel
      var dy2 = frame.data[4*((y+1)*width + x) + 1] - 
              frame.data[4*((y-1)*width + x) + 1];
      var dx2 = frame.data[4*(y*width + (x+1)) + 1] - 
              frame.data[4*(y*width + (x-1)) + 1];
      var v2 = dx*dx + dy*dy;

      // third color channel
      var dy3 = frame.data[4*((y+1)*width + x) + 2] - 
              frame.data[4*((y-1)*width + x) + 2];
      var dx3 = frame.data[4*(y*width + (x+1)) + 2] - 
              frame.data[4*(y*width + (x-1)) + 2];
      var v3 = dx*dx + dy*dy;

      // pick channel with strongest gradient
      if (v2 > v) {
        v = v2;
        dx = dx2;
        dy = dy2;
      } 
      if (v3 > v) {
        v = v3;
        dx = dx3;
        dy = dy3;
      }

      // snap to one of 18 orientations
      var best_dot = 0;
      var best_o = 0;
      for (var o = 0; o < 9; o++) {
        var dot = uu[o]*dx + vv[o]*dy;
        for (var i = -sbin/2; i < sbin/2; i++) {
          if ((4*((y+i*yb[o])*width + (x+i*xb[o])) + 3) > frame.data.length)
            break;
          //frame.data[4*((y+1)*width + (x)) + 3] = best_dot;
          //frame.data[4*((y-1)*width + (x)) + 3] = best_dot;
          frame.data[4*((y+i*yb[o])*width + (x+i*xb[o])) + 3] = 
            255-7*dot;
        }
        if (dot > best_dot) {
          best_dot = dot;
          best_o = o;
        } else if (-dot > best_dot) {
          best_dot = -dot;
          best_o = o+9;
        }
      }

      //frame.data[4*(y*width + x) + 3] = 255-best_dot;
/*      for (var i = 0; i < sbin; i++) {
        //frame.data[4*((y+1)*width + (x)) + 3] = best_dot;
        //frame.data[4*((y-1)*width + (x)) + 3] = best_dot;
        frame.data[4*((y+i*yb[best_o])*width + (x+i*xb[best_o])) + 3] = 
          255-10*best_dot;
      }
  */    // add to 4 histograms around pixel using linear varerpolation
      /*var xp = (x+0.5)/sbin - 0.5;
      var yp = (y+0.5)/sbin - 0.5;
      var ixp = Math.floor(xp);
      var iyp = Math.floor(yp);
      var vx0 = xp-ixp;
      var vy0 = yp-iyp;
      var vx1 = 1.0-vx0;
      var vy1 = 1.0-vy0;
      v = Math.sqrt(v);*/
/*
      if (ixp >= 0 && iyp >= 0) {
      }

      if (ixp+1 < blocks[1] && iyp >= 0) {
      }

      if (ixp >= 0 && iyp+1 < blocks[0]) {
      }

      if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
      }*/
    }
  }

  return;
}


