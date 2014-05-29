// Original Code by Pedro Felzenszwalb
// Modified by Michael and Bram to include
// Intel vector intrinsics.
//

#include <math.h>
#include "vector_intrinsics.h"
#include "mex.h"

#include "util.h"

// small value, used to avoid division by zero
#define eps 0.0001
//size of lookup table
#include "best_o_lookup.h"


static const vreal veps = set_real(eps);
static const vreal point5 = set_real(0.5f);
static const vreal point2 = set_real(0.2f);
static const vreal point2357 = set_real(0.2357f);
static const vreal one = set_real(1.0f);
static const vreal lookup_size = set_real(LOOKUP_SIZE);
static const vnat lookup_size_nat = set_nat(LOOKUP_SIZE);
static const vnat one_nat = set_nat(1);

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

  const vnat blocks0 = set_nat(blocks[0]);
  const vnat blocks1 = set_nat(blocks[1]);

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
  int ystop = (dims[0]-2-SIMD_WIDTH) - ((dims[0]-2-SIMD_WIDTH)-1) % SIMD_WIDTH;
  dbg_printf("ystop  diff to visible[0]-1 %d\n", (visible[0]-1)-ystop);
  ASSERT(ystop <= visible[0]-1);  // The rest is done sequentially
  ASSERT((ystop - 1) % 4 == 0);  // Must be a multiple of 4 for SSE instructions
  for (int x = 1; x < visible[1]-1; x++) {
    real xp = ((real)x+0.5)/(real)sbin - 0.5;
    nat ixp = (nat)floor(xp);
    real vx0 = xp-ixp;
    real vx1 = 1.0-vx0;
    vnat vixp = set_nat(ixp);
    vreal vvx0 = set_real(vx0);
    vreal vvx1 = set_real(vx1);
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
      //vreal best_dot = set_real(0.0f);
      //vnat best_o = set_nat(0);
      //for (int o = 0; o < 9; o++) {
      //  vreal uuo = load_real(uu + o);
      //  vreal vvo = load_real(vv + o);
      //  uuo = mul_vreal(uuo, dx);
      //  vvo = mul_vreal(vvo, dy);
      //  vreal dot = add_vreal(uuo, vvo);

      //  mask = cmpgt_vreal(dot, best_dot);
      //  vnat vo = set_nat(o);
      //  best_dot = or_vreal(and_vreal(mask, dot),
      //                      andnot_vreal(mask, best_dot));
      //  best_o = or_vnat(and_vnat(mask, vo),
      //                   andnot_vnat(mask, best_o));

      //  dot = neg_vreal(dot);
      //  mask = cmpgt_vreal(dot, best_dot);
      //  vo = set_nat(o + 9);
      //  best_dot = or_vreal(and_vreal(mask, dot),
      //                      andnot_vreal(mask, best_dot));
      //  best_o = or_vnat(and_vnat(mask, vo),
      //                   andnot_vnat(mask, best_o));
      //}
      vnat lookup_x = vreal_convertto_vnat(mul_vreal(lookup_size, dx));
      lookup_x = add_vnat(lookup_x, lookup_size_nat);

      vnat lookup_y = vreal_convertto_vnat(mul_vreal(lookup_size, dy));
      lookup_y = add_vnat(lookup_y, lookup_size_nat);

      nat lookup_xs[SIMD_WIDTH];
      store_vnat(lookup_xs, lookup_x);
      nat lookup_ys[SIMD_WIDTH];
      store_vnat(lookup_ys, lookup_y);

      nat best_os[SIMD_WIDTH];     
      for (int i = 0; i < SIMD_WIDTH; i++) {
        ASSERT(lookup_xs[i] >= 0 && lookup_xs[i] < LOOKUP_SIZE*2+1);
        ASSERT(lookup_ys[i] >= 0 && lookup_ys[i] < LOOKUP_SIZE*2+1);
        best_os[i] = best_o_lookup[lookup_xs[i]][lookup_ys[i]];
      }

      vnat best_o = load_vnat(best_os);
      // Update histograms
      // Replaced code:  Some code outside inner loop
      // real xp = ((real)x+0.5)/(real)sbin - 0.5;
      // real yp = ((real)y+0.5)/(real)sbin - 0.5;
      // int ixp = (int)floor(xp);
      // int iyp = (int)floor(yp);
      // real vx0 = xp-ixp;
      // real vx1 = 1.0-vx0;
      // real vy1 = 1.0-vy0;
      // v = sqrt(v);
      vreal vy = set_real((real)y);
      vy = add_vreal(vy, SIMD_WIDTH_IDX_REAL);
      vreal yp = add_vreal(vy, point5);
      yp = div_vreal(yp, vsbin);
      yp = sub_vreal(yp, point5);
      vreal fyp = floor_vreal(yp);
      vnat viyp = vreal_convertto_vnat(fyp);
      vreal vvy0 = sub_vreal(yp, fyp);
      vreal vvy1 = sub_vreal(one, vvy0);
      v = sqrt_vreal(v);

      nat iyps[SIMD_WIDTH];
      store_vnat(iyps, viyp);
      
      nat histp[4][SIMD_WIDTH];
      real valplus[4][SIMD_WIDTH];

      vnat vhistpup = viyp;
      vnat vhistpdown = add_vnat(viyp, one_nat);
      vnat vhistpleft = mul_vnat(vixp, blocks0);
      vnat vhistpright = mul_vnat(add_vnat(vixp, one_nat), blocks0);
      vnat vhistpbase = mul_vnat(mul_vnat(best_o, blocks0), blocks1);

      vnat vhistp = add_vnat(add_vnat(vhistpup, vhistpleft), vhistpbase);
      vreal vvalplus = mul_vreal(mul_vreal(vvx1, vvy1), v);

      store_vnat(histp[0], vhistp);
      store_vreal(valplus[0], vvalplus);

      vhistp = add_vnat(add_vnat(vhistpup, vhistpright), vhistpbase);
      vvalplus = mul_vreal(mul_vreal(vvx0, vvy1), v);

      store_vnat(histp[1], vhistp);
      store_vreal(valplus[1], vvalplus);
      
      vhistp = add_vnat(add_vnat(vhistpdown, vhistpleft), vhistpbase);
      vvalplus = mul_vreal(mul_vreal(vvx1, vvy0), v);

      store_vnat(histp[2], vhistp);
      store_vreal(valplus[2], vvalplus);
      
      vhistp = add_vnat(add_vnat(vhistpdown, vhistpright), vhistpbase);
      vvalplus = mul_vreal(mul_vreal(vvx0, vvy0), v);

      store_vnat(histp[3], vhistp);
      store_vreal(valplus[3], vvalplus);
      // Finish updating histograms sequentially.  This is a scatter.
      for (int yoff = 0; yoff < SIMD_WIDTH; yoff ++) {
        // add to 4 histograms around pixel using linear interpolation

        if (ixp >= 0 && iyps[yoff] >= 0) {
          ASSERT(histp[0][yoff] < blocks[0]*blocks[1]*18);
          *(hist + histp[0][yoff]) += valplus[0][yoff];
        }

        if (ixp+1 < blocks[1] && iyps[yoff] >= 0) {
          ASSERT(histp[1][yoff] < blocks[0]*blocks[1]*18);
          *(hist + histp[1][yoff]) += valplus[1][yoff];
        }

        if (ixp >= 0 && iyps[yoff]+1 < blocks[0]) {
          ASSERT(histp[2][yoff] < blocks[0]*blocks[1]*18);
          *(hist + histp[2][yoff]) += valplus[2][yoff];
        }

        if (ixp+1 < blocks[1] && iyps[yoff]+1 < blocks[0]) {
          ASSERT(histp[3][yoff] < blocks[0]*blocks[1]*18);
          *(hist + histp[3][yoff]) += valplus[3][yoff];
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

      ASSERT(dx<=1.0f && dx >= -1.0f);
      ASSERT(dy<=1.0f && dy >= -1.0f);
      // snap to one of 18 orientations using lookup
      int lookup_x = (int)(LOOKUP_SIZE*dx)+LOOKUP_SIZE;
      ASSERT(lookup_x >= 0 && lookup_x < LOOKUP_SIZE*2+1);
      int lookup_y = (int)(LOOKUP_SIZE*dy)+LOOKUP_SIZE;
      ASSERT(lookup_y >= 0 && lookup_y < LOOKUP_SIZE*2+1);
      int best_o = (int)best_o_lookup[lookup_x][lookup_y];

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
    real *src2 = src1 + 9*blocks[0]*blocks[1];
    real *dst = norm;
    real *end = norm + blocks[1]*blocks[0];
    real *endstop = dst + (end - dst) % SIMD_WIDTH;
    while (dst < endstop) {
      vreal vsrc1 = load_vreal(src1);
      vreal vsrc2 = load_vreal(src2);
      vsrc1 = add_vreal(vsrc1, vsrc2);
      store_vreal(dst, mul_vreal(vsrc1, vsrc1));
      dst += SIMD_WIDTH;
      src1 += SIMD_WIDTH;
      src2 += SIMD_WIDTH;
    }
    while (dst < end) {
      *(dst++) += (*src1 + *src2) * (*src1 + *src2);
      src1++;
      src2++;
    }
  }

  // compute features
  for (int x = 0; x < out[1]; x++) {
    int ystop = out[0] - (out[0] % SIMD_WIDTH);
    for (int y = 0; y < ystop; y+=SIMD_WIDTH) {
      real *dst = feat + x*out[0] + y;      
      real *src, *p;
      vreal vdst, vsrc, vp, vp1, vpblocks0, vpblocks0p1;
      vreal n1, n2, n3, n4;

      p = norm + (x+1)*blocks[0] + y+1;
      vp = load_vreal(p);
      vp1 = load_vreal(p+1);
      vpblocks0 = load_vreal(p + blocks[0]);
      vpblocks0p1 = load_vreal(p + blocks[0] + 1);
      n1 = add_vreal(vp, vp1);
      n1 = add_vreal(n1, vpblocks0);
      n1 = add_vreal(n1, vpblocks0p1);
      n1 = add_vreal(n1, veps);
      n1 = sqrt_vreal(n1);
      n1 = div_vreal(one, n1);

      p = norm + (x+1)*blocks[0] + y;
      vp = load_vreal(p);
      vp1 = load_vreal(p+1);
      vpblocks0 = load_vreal(p + blocks[0]);
      vpblocks0p1 = load_vreal(p + blocks[0] + 1);
      n2 = add_vreal(vp, vp1);
      n2 = add_vreal(n2, vpblocks0);
      n2 = add_vreal(n2, vpblocks0p1);
      n2 = add_vreal(n2, veps);
      n2 = sqrt_vreal(n2);
      n2 = div_vreal(one, n2);
      
      p = norm + x*blocks[0] + y+1;
      vp = load_vreal(p);
      vp1 = load_vreal(p+1);
      vpblocks0 = load_vreal(p + blocks[0]);
      vpblocks0p1 = load_vreal(p + blocks[0] + 1);
      n3 = add_vreal(vp, vp1);
      n3 = add_vreal(n3, vpblocks0);
      n3 = add_vreal(n3, vpblocks0p1);
      n3 = add_vreal(n3, veps);
      n3 = sqrt_vreal(n3);
      n3 = div_vreal(one, n3);

      p = norm + x*blocks[0] + y;      
      vp = load_vreal(p);
      vp1 = load_vreal(p+1);
      vpblocks0 = load_vreal(p + blocks[0]);
      vpblocks0p1 = load_vreal(p + blocks[0] + 1);
      n4 = add_vreal(vp, vp1);
      n4 = add_vreal(n4, vpblocks0);
      n4 = add_vreal(n4, vpblocks0p1);
      n4 = add_vreal(n4, veps);
      n4 = sqrt_vreal(n4);
      n4 = div_vreal(one, n4);

      vreal t1 = set_real(0);
      vreal t2 = set_real(0);
      vreal t3 = set_real(0);
      vreal t4 = set_real(0);

      // contrast-sensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 18; o++) {
        vsrc = load_vreal(src);
        vreal h1 = min_vreal(mul_vreal(vsrc, n1), point2);
        vreal h2 = min_vreal(mul_vreal(vsrc, n2), point2);
        vreal h3 = min_vreal(mul_vreal(vsrc, n3), point2);
        vreal h4 = min_vreal(mul_vreal(vsrc, n4), point2);
        vdst = mul_vreal(point5, add_vreal(add_vreal(h1, h2),
                                           add_vreal(h3, h4)));
        store_vreal(dst, vdst);
        t1 = add_vreal(t1, h1);
        t2 = add_vreal(t2, h2);
        t3 = add_vreal(t3, h3);
        t4 = add_vreal(t4, h4);
        dst += out[0]*out[1];
        src += blocks[0]*blocks[1];
      }

      // contrast-insensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 9; o++) {
        vsrc = load_vreal(src + 9*blocks[0]*blocks[1]);
        vsrc = add_vreal(vsrc, load_vreal(src));
        vreal h1 = min_vreal(mul_vreal(vsrc, n1), point2);
        vreal h2 = min_vreal(mul_vreal(vsrc, n2), point2);
        vreal h3 = min_vreal(mul_vreal(vsrc, n3), point2);
        vreal h4 = min_vreal(mul_vreal(vsrc, n4), point2);
        vdst = mul_vreal(point5, add_vreal(add_vreal(h1, h2),
                                           add_vreal(h3, h4)));
        store_vreal(dst, vdst);
        dst += out[0]*out[1];
        src += blocks[0]*blocks[1];
      }

      // texture features
      vdst = mul_vreal(point2357, t1);
      store_vreal(dst, vdst);
      dst += out[0]*out[1];
      vdst = mul_vreal(point2357, t2);
      store_vreal(dst, vdst);
      dst += out[0]*out[1];
      vdst = mul_vreal(point2357, t3);
      store_vreal(dst, vdst);
      dst += out[0]*out[1];
      vdst = mul_vreal(point2357, t4);
      store_vreal(dst, vdst);
    }
    ///////////////////////////////////
    // Continue sequentially after SIMD
    ///////////////////////////////////
    for (int y = ystop; y < out[0]; y++) {
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



