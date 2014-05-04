// Original Code by Pedro Felzenszwalb
// Modified by Michael and Bram to include
// Intel vector intrinsics.
//

#include <math.h>
#define DOUBLE
#include "vector_intrinsics.h"
#include "mex.h"

// small value, used to avoid division by zero
#define eps 0.0001

#define DEBUG
#ifdef DEBUG
#undef NDEBUG
#include <assert.h>
#define ASSERT(b) assert((b))
#define dbg_printf(...) printf(__VA_ARGS__)
#else
#define ASSERT(b) (void)(b);
#define dbg_printf(...) do {} while (0);
#endif  // DEBUG

static const vreal point5 = set_real(0.5f);
static const vreal one = set_real(1.0f);

// unit vectors used to compute gradient orientation
real uu[9] = {1.0000, 
  0.9397, 
  0.7660, 
  0.500, 
  0.1736, 
  -0.1736, 
  -0.5000, 
  -0.7660, 
  -0.9397};
real vv[9] = {0.0000, 
  0.3420, 
  0.6428, 
  0.8660, 
  0.9848, 
  0.9848, 
  0.8660, 
  0.6428, 
  0.3420};

static inline real min(real x, real y) { return (x <= y ? x : y); }
static inline real max(real x, real y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a real color image and a bin size 
// returns HOG features
mxArray *process(const mxArray *mximage, const mxArray *mxsbin) {
  real *im = (real *)mxGetPr(mximage);
  const int *dims = mxGetDimensions(mximage);
  if (mxGetNumberOfDimensions(mximage) != 3 ||
      dims[2] != 3 ||
      mxGetClassID(mximage) != mxCLASS)
    mexErrMsgTxt("Invalid input");

  int sbin = (int)mxGetScalar(mxsbin);

  // memory for caching orientation histograms & their norms
  int blocks[2];
  blocks[0] = (int)round((real)dims[0]/(real)sbin);
  blocks[1] = (int)round((real)dims[1]/(real)sbin);
  real *hist = (real *)mxCalloc(blocks[0]*blocks[1]*18, sizeof(real));
  real *norm = (real *)mxCalloc(blocks[0]*blocks[1], sizeof(real));

  // memory for HOG features
  int out[3];
  out[0] = max(blocks[0]-2, 0);
  out[1] = max(blocks[1]-2, 0);
  out[2] = 27+4;
  mxArray *mxfeat = mxCreateNumericArray(3, out, mxCLASS, mxREAL);
  real *feat = (real *)mxGetPr(mxfeat);

  int visible[2];
  visible[0] = blocks[0]*sbin;
  visible[1] = blocks[1]*sbin;

  // Vectorized loop
  const vreal vsbin = set_real((real)sbin);
  int ystop = (dims[0]-2) - ((dims[0]-2)-1) % SIMD_WIDTH;
  ASSERT(ystop <= visible[0]-1);  // The rest is done sequentially
  ASSERT((ystop - 1) % 4 == 0);  // Must be a multiple of 4 for SSE instructions
  dbg_printf("ystop  diff to visible[0]-1 %d\n", (visible[0]-1)-ystop);
  for (int x = 1; x < visible[1]-1; x++) {
    real xp = ((real)x+0.5)/(real)sbin - 0.5;
    nat ixp = (nat)floor(xp);
    real vx0 = xp-ixp;
    real vx1 = 1.0-vx0;
    for (int y = 1; y < ystop; y += SIMD_WIDTH) {
      // Make sure we don't access anything bad
      ASSERT(y+SIMD_WIDTH-1 <= dims[0]-2);

      // compute gradient fist color channel (RED)
      // Code replaced:
      // real *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
      // real dy = *(s+1) - *(s-1);
      // real dx = *(s+dims[0]) - *(s-dims[0]);
      // real v = dx*dx + dy*dy;
      real *sp = im + min(x, dims[1]-2)*dims[0] + y;
      vreal s = load_vreal(sp);
      
      vreal sp1 = load_vreal((sp + 1));
      vreal sm1 = load_vreal((sp - 1));
      vreal dy = sub_vreal(sp1, sm1);

      vreal spd0 = load_vreal((sp + dims[0]));
      vreal smd0 = load_vreal((sp - dims[0]));
      vreal dx = sub_vreal(spd0, smd0);

      vreal vl = mul_vreal(dx, dx);
      vreal vr = mul_vreal(dy, dy);
      vreal v = add_vreal(vl, vr);


      // compute gradient second color channel (GREEN)
      // Code replaced:
      // s += dims[0]*dims[1];
      // real dy2 = *(s+1) - *(s-1);
      // real dx2 = *(s+dims[0]) - *(s-dims[0]);
      // real v2 = dx2*dx2 + dy2*dy2;
      sp += dims[0]*dims[1];
      s = load_vreal(sp);
      
      sp1 = load_vreal((sp + 1));
      sm1 = load_vreal((sp - 1));
      vreal dy2 = sub_vreal(sp1, sm1);

      spd0 = load_vreal((sp + dims[0]));
      smd0 = load_vreal((sp - dims[0]));
      vreal dx2 = sub_vreal(spd0, smd0);

      vl = mul_vreal(dx2, dx2);
      vr = mul_vreal(dy2, dy2);
      vreal v2 = add_vreal(vl, vr);


      // compute gradient third color channel (BLUE)
      // Code replaced:
      // s += dims[0]*dims[1];
      // real dy3 = *(s+1) - *(s-1);
      // real dx3 = *(s+dims[0]) - *(s-dims[0]);
      // real v3 = dx3*dx3 + dy3*dy3;
      sp += dims[0]*dims[1];
      s = load_vreal(sp);
      
      sp1 = load_vreal((sp + 1));
      sm1 = load_vreal((sp - 1));
      vreal dy3 = sub_vreal(sp1, sm1);

      spd0 = load_vreal((sp + dims[0]));
      smd0 = load_vreal((sp - dims[0]));
      vreal dx3 = sub_vreal(spd0, smd0);

      vl = mul_vreal(dx3, dx3);
      vr = mul_vreal(dy3, dy3);
      vreal v3 = add_vreal(vl, vr);


      // pick channel with strongest gradient
      // Code replaced:
      // if (v2 > v) {
      //   v = v2;
      //   dx = dx2;
      //   dy = dy2;
      // } 
      // if (v3 > v) {
      //   v = v3;
      //   dx = dx3;
      //   dy = dy3;
      // }
      vmask mask = cmpgt_vreal(v2, v);
      v = or_vreal(and_vreal(mask, v2),
                    andnot_vreal(mask, v));
      dx = or_vreal(and_vreal(mask, dx2),
                     andnot_vreal(mask, dx));
      dy = or_vreal(and_vreal(mask, dy2),
                     andnot_vreal(mask, dy));

      mask = cmpgt_vreal(v3, v);
      v = or_vreal(and_vreal(mask, v3),
                    andnot_vreal(mask, v));
      dx = or_vreal(and_vreal(mask, dx3),
                     andnot_vreal(mask, dx));
      dy = or_vreal(and_vreal(mask, dy3),
                     andnot_vreal(mask, dy));


      // snap to one of 18 orientations
      // Code replaced:
      // real best_dot = 0;
      // int best_o = 0;
      // for (int o = 0; o < 9; o++) {
      //   real dot = uu[o]*dx + vv[o]*dy;
      //   if (dot > best_dot) {
      //     best_dot = dot;
      //     best_o = o;
      //   } else if (-dot > best_dot) {
      //     best_dot = -dot;
      //     best_o = o+9;
      //   }
      // }
      vreal best_dot = set_real(0.0f);
      vnat best_o = set_nat(0);
      for (int o = 0; o < 9; o++) {
        vreal uuo = load_real(uu + o);
        vreal vvo = load_real(vv + o);
        uuo = mul_vreal(uuo, dx);
        vvo = mul_vreal(vvo, dy);
        vreal dot = add_vreal(uuo, vvo);

        mask = cmpgt_vreal(dot, best_dot);
        vnat vo = set_nat(o);
        best_dot = or_vreal(and_vreal(mask, dot),
                            andnot_vreal(mask, best_dot));
        best_o = or_vnat(and_vnat(mask, vo),
                         andnot_vnat(mask, best_o));

        dot = neg_vreal(dot);
        mask = cmpgt_vreal(dot, best_dot);
        vo = set_nat(o + 9);
        best_dot = or_vreal(and_vreal(mask, dot),
                            andnot_vreal(mask, best_dot));
        best_o = or_vnat(and_vnat(mask, vo),
                         andnot_vnat(mask, best_o));
      }


      // Update histograms
      // Replaced code:  Some code outside inner loop
      // real xp = ((real)x+0.5)/(real)sbin - 0.5;
      // real yp = ((real)y+0.5)/(real)sbin - 0.5;
      // int ixp = (int)floor(xp);
      // int iyp = (int)floor(yp);
      // real vx0 = xp-ixp;
      // real vy0 = yp-iyp;
      // real vx1 = 1.0-vx0;
      // real vy1 = 1.0-vy0;
      // v = sqrt(v);
      vreal vy = set_real((real)y);
      vy = add_vreal(vy, SIMD_WIDTH_IDX_REAL);
      vreal yp = add_vreal(vy, point5);
      yp = div_vreal(yp, vsbin);
      yp = sub_vreal(yp, point5);
      vreal fyp = floor_vreal(yp);
      vnat iyp = vreal_convertto_vnat(fyp);
      vreal vy0 = sub_vreal(yp, fyp);
      vreal vy1 = sub_vreal(one, vy0);
      v = sqrt_vreal(v);

      real vs[SIMD_WIDTH];
      store_vreal(vs, v);

      nat best_os[SIMD_WIDTH];
      store_vnat(best_os, best_o);

      nat iyps[SIMD_WIDTH];
      store_vnat(iyps, iyp);
      
      real vy0s[SIMD_WIDTH];
      store_vreal(vy0s, vy0);
      
      real vy1s[SIMD_WIDTH];
      store_vreal(vy1s, vy1);

      // Finish updating histograms sequentially.  This is a scatter.
      for (int yoff = 0; yoff < SIMD_WIDTH; yoff ++) {
        // add to 4 histograms around pixel using linear interpolation

        if (ixp >= 0 && iyps[yoff] >= 0) {
          *(hist + ixp*blocks[0] + iyps[yoff] + (best_os[yoff])*blocks[0]*blocks[1]) += 
            vx1*vy1s[yoff]*(vs[yoff]);
        }

        if (ixp+1 < blocks[1] && iyps[yoff] >= 0) {
          *(hist + (ixp+1)*blocks[0] + iyps[yoff] + (best_os[yoff])*blocks[0]*blocks[1]) += 
            vx0*vy1s[yoff]*(vs[yoff]);
        }

        if (ixp >= 0 && iyps[yoff]+1 < blocks[0]) {
          *(hist + ixp*blocks[0] + (iyps[yoff]+1) + (best_os[yoff])*blocks[0]*blocks[1]) += 
            vx1*vy0s[yoff]*(vs[yoff]);
        }

        if (ixp+1 < blocks[1] && iyps[yoff]+1 < blocks[0]) {
          *(hist + (ixp+1)*blocks[0] + (iyps[yoff]+1) + (best_os[yoff])*blocks[0]*blocks[1]) += 
            vx0*vy0s[yoff]*(vs[yoff]);
        }
      }
    }
    for (int y = ystop; y < visible[0]-1; y++) {
      // first color channel
      real *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
      real dy = *(s+1) - *(s-1);
      real dx = *(s+dims[0]) - *(s-dims[0]);
      real v = dx*dx + dy*dy;

      // second color channel
      s += dims[0]*dims[1];
      real dy2 = *(s+1) - *(s-1);
      real dx2 = *(s+dims[0]) - *(s-dims[0]);
      real v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += dims[0]*dims[1];
      real dy3 = *(s+1) - *(s-1);
      real dx3 = *(s+dims[0]) - *(s-dims[0]);
      real v3 = dx3*dx3 + dy3*dy3;

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
      real best_dot = 0;
      int best_o = 0;
      for (int o = 0; o < 9; o++) {
        real dot = uu[o]*dx + vv[o]*dy;
        if (dot > best_dot) {
          best_dot = dot;
          best_o = o;
        } else if (-dot > best_dot) {
          best_dot = -dot;
          best_o = o+9;
        }
      }

      // add to 4 histograms around pixel using linear interpolation
      real yp = ((real)y+0.5)/(real)sbin - 0.5;
      int iyp = (int)floor(yp);
      real vy0 = yp-iyp;
      real vy1 = 1.0-vy0;
      v = sqrt(v);

      if (ixp >= 0 && iyp >= 0) {
        *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
          vx1*vy1*v;
      }

      if (ixp+1 < blocks[1] && iyp >= 0) {
        *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
          vx0*vy1*v;
      }

      if (ixp >= 0 && iyp+1 < blocks[0]) {
        *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
          vx1*vy0*v;
      }

      if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
        *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
          vx0*vy0*v;
      }
    }
  }

  // compute energy in each block by summing over orientations
  for (int o = 0; o < 9; o++) {
    real *src1 = hist + o*blocks[0]*blocks[1];
    real *src2 = hist + (o+9)*blocks[0]*blocks[1];
    real *dst = norm;
    real *end = norm + blocks[1]*blocks[0];
    while (dst < end) {
      *(dst++) += (*src1 + *src2) * (*src1 + *src2);
      src1++;
      src2++;
    }
  }

  // compute features
  for (int x = 0; x < out[1]; x++) {
    for (int y = 0; y < out[0]; y++) {
      real *dst = feat + x*out[0] + y;      
      real *src, *p, n1, n2, n3, n4;

      p = norm + (x+1)*blocks[0] + y+1;
      n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + (x+1)*blocks[0] + y;
      n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + x*blocks[0] + y+1;
      n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + x*blocks[0] + y;      
      n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

      real t1 = 0;
      real t2 = 0;
      real t3 = 0;
      real t4 = 0;

      // contrast-sensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 18; o++) {
        real h1 = min(*src * n1, 0.2);
        real h2 = min(*src * n2, 0.2);
        real h3 = min(*src * n3, 0.2);
        real h4 = min(*src * n4, 0.2);
        *dst = 0.5 * (h1 + h2 + h3 + h4);
        t1 += h1;
        t2 += h2;
        t3 += h3;
        t4 += h4;
        dst += out[0]*out[1];
        src += blocks[0]*blocks[1];
      }

      // contrast-insensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 9; o++) {
        real sum = *src + *(src + 9*blocks[0]*blocks[1]);
        real h1 = min(sum * n1, 0.2);
        real h2 = min(sum * n2, 0.2);
        real h3 = min(sum * n3, 0.2);
        real h4 = min(sum * n4, 0.2);
        *dst = 0.5 * (h1 + h2 + h3 + h4);
        dst += out[0]*out[1];
        src += blocks[0]*blocks[1];
      }

      // texture features
      *dst = 0.2357 * t1;
      dst += out[0]*out[1];
      *dst = 0.2357 * t2;
      dst += out[0]*out[1];
      *dst = 0.2357 * t3;
      dst += out[0]*out[1];
      *dst = 0.2357 * t4;
    }
  }

  mxFree(hist);
  mxFree(norm);
  return mxfeat;
}

// matlab entry point
// F = features_pedro(image, bin)
// image should be color with real values
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 2)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");
  plhs[0] = process(prhs[0], prhs[1]);
}



