#include <math.h>
#include "smmintrin.h"
#include "xmmintrin.h"
#include "emmintrin.h"
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

static const __m128 SIGNMASK = 
               _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
static const __m128 point5 = _mm_set_ps1(0.5f);
static const __m128 one = _mm_set_ps1(1.0f);
static const __m128 v0123 = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);

// unit vectors used to compute gradient orientation
float uu[9] = {1.0000, 
  0.9397, 
  0.7660, 
  0.500, 
  0.1736, 
  -0.1736, 
  -0.5000, 
  -0.7660, 
  -0.9397};
float vv[9] = {0.0000, 
  0.3420, 
  0.6428, 
  0.8660, 
  0.9848, 
  0.9848, 
  0.8660, 
  0.6428, 
  0.3420};

static inline float min(float x, float y) { return (x <= y ? x : y); }
static inline float max(float x, float y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a float color image and a bin size 
// returns HOG features
mxArray *process(const mxArray *mximage, const mxArray *mxsbin) {
  float *im = (float *)mxGetPr(mximage);
  const int *dims = mxGetDimensions(mximage);
  if (mxGetNumberOfDimensions(mximage) != 3 ||
      dims[2] != 3 ||
      mxGetClassID(mximage) != mxSINGLE_CLASS)
    mexErrMsgTxt("Invalid input");

  int sbin = (int)mxGetScalar(mxsbin);

  // memory for caching orientation histograms & their norms
  int blocks[2];
  blocks[0] = (int)round((float)dims[0]/(float)sbin);
  blocks[1] = (int)round((float)dims[1]/(float)sbin);
  float *hist = (float *)mxCalloc(blocks[0]*blocks[1]*18, sizeof(float));
  float *norm = (float *)mxCalloc(blocks[0]*blocks[1], sizeof(float));

  // memory for HOG features
  int out[3];
  out[0] = max(blocks[0]-2, 0);
  out[1] = max(blocks[1]-2, 0);
  out[2] = 27+4;
  mxArray *mxfeat = mxCreateNumericArray(3, out, mxSINGLE_CLASS, mxREAL);
  float *feat = (float *)mxGetPr(mxfeat);

  int visible[2];
  visible[0] = blocks[0]*sbin;
  visible[1] = blocks[1]*sbin;

  // Vectorized loop
  const __m128 vsbin = _mm_set_ps1((float)sbin);
  int ystop = (dims[0]-2) - ((dims[0]-2)-1) % 4;
  ASSERT(ystop <= visible[0]-1);  // The rest is done sequentially
  ASSERT((ystop - 1) % 4 == 0);  // Must be a multiple of 4 for SSE instructions
  dbg_printf("ystop  diff to visible[0]-1 %d\n", (visible[0]-1)-ystop);
  for (int x = 1; x < visible[1]-1; x++) {
    float xp = ((float)x+0.5)/(float)sbin - 0.5;
    int ixp = (int)floor(xp);
    float vx0 = xp-ixp;
    float vx1 = 1.0-vx0;
    for (int y = 1; y < ystop; y += 4) {
      ASSERT(y+3 <= dims[0]-2);  // Make sure we don't access anything bad
      // compute gradient fist color channel (RED)
      // Code replaced:
      // float *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
      // float dy = *(s+1) - *(s-1);
      // float dx = *(s+dims[0]) - *(s-dims[0]);
      // float v = dx*dx + dy*dy;
      float *sp = im + min(x, dims[1]-2)*dims[0] + y;
      __m128 s = _mm_loadu_ps(sp);
      
      __m128 sp1 = _mm_loadu_ps((sp + 1));
      __m128 sm1 = _mm_loadu_ps((sp - 1));
      __m128 dy = _mm_sub_ps(sp1, sm1);

      __m128 spd0 = _mm_loadu_ps((sp + dims[0]));
      __m128 smd0 = _mm_loadu_ps((sp - dims[0]));
      __m128 dx = _mm_sub_ps(spd0, smd0);

      __m128 vl = _mm_mul_ps(dx, dx);
      __m128 vr = _mm_mul_ps(dy, dy);
      __m128 v = _mm_add_ps(vl, vr);


      // compute gradient second color channel (GREEN)
      // Code replaced:
      // s += dims[0]*dims[1];
      // float dy2 = *(s+1) - *(s-1);
      // float dx2 = *(s+dims[0]) - *(s-dims[0]);
      // float v2 = dx2*dx2 + dy2*dy2;
      sp += dims[0]*dims[1];
      s = _mm_loadu_ps(sp);
      
      sp1 = _mm_loadu_ps((sp + 1));
      sm1 = _mm_loadu_ps((sp - 1));
      __m128 dy2 = _mm_sub_ps(sp1, sm1);

      spd0 = _mm_loadu_ps((sp + dims[0]));
      smd0 = _mm_loadu_ps((sp - dims[0]));
      __m128 dx2 = _mm_sub_ps(spd0, smd0);

      vl = _mm_mul_ps(dx2, dx2);
      vr = _mm_mul_ps(dy2, dy2);
      __m128 v2 = _mm_add_ps(vl, vr);


      // compute gradient third color channel (BLUE)
      // Code replaced:
      // s += dims[0]*dims[1];
      // float dy3 = *(s+1) - *(s-1);
      // float dx3 = *(s+dims[0]) - *(s-dims[0]);
      // float v3 = dx3*dx3 + dy3*dy3;
      sp += dims[0]*dims[1];
      s = _mm_loadu_ps(sp);
      
      sp1 = _mm_loadu_ps((sp + 1));
      sm1 = _mm_loadu_ps((sp - 1));
      __m128 dy3 = _mm_sub_ps(sp1, sm1);

      spd0 = _mm_loadu_ps((sp + dims[0]));
      smd0 = _mm_loadu_ps((sp - dims[0]));
      __m128 dx3 = _mm_sub_ps(spd0, smd0);

      vl = _mm_mul_ps(dx3, dx3);
      vr = _mm_mul_ps(dy3, dy3);
      __m128 v3 = _mm_add_ps(vl, vr);


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
      __m128 mask = _mm_cmpgt_ps(v2, v);
      v = _mm_or_ps(_mm_and_ps(mask, v2),
                    _mm_andnot_ps(mask, v));
      dx = _mm_or_ps(_mm_and_ps(mask, dx2),
                     _mm_andnot_ps(mask, dx));
      dy = _mm_or_ps(_mm_and_ps(mask, dy2),
                     _mm_andnot_ps(mask, dy));

      mask = _mm_cmpgt_ps(v3, v);
      v = _mm_or_ps(_mm_and_ps(mask, v3),
                    _mm_andnot_ps(mask, v));
      dx = _mm_or_ps(_mm_and_ps(mask, dx3),
                     _mm_andnot_ps(mask, dx));
      dy = _mm_or_ps(_mm_and_ps(mask, dy3),
                     _mm_andnot_ps(mask, dy));


      // snap to one of 18 orientations
      // Code replaced:
      // float best_dot = 0;
      // int best_o = 0;
      // for (int o = 0; o < 9; o++) {
      //   float dot = uu[o]*dx + vv[o]*dy;
      //   if (dot > best_dot) {
      //     best_dot = dot;
      //     best_o = o;
      //   } else if (-dot > best_dot) {
      //     best_dot = -dot;
      //     best_o = o+9;
      //   }
      // }
      __m128 best_dot = _mm_set_ps1(0.0f);
      __m128i best_o = _mm_set1_epi32(0);
      for (int o = 0; o < 9; o++) {
        __m128 uuo = _mm_load_ps1(uu + o);
        __m128 vvo = _mm_load_ps1(vv + o);
        uuo = _mm_mul_ps(uuo, dx);
        vvo = _mm_mul_ps(vvo, dy);
        
        __m128 dot = _mm_add_ps(uuo, vvo);
        mask = _mm_cmpgt_ps(dot, best_dot);
        __m128i vo = _mm_set1_epi32(o);
        best_dot = _mm_or_ps(_mm_and_ps(mask, dot),
                             _mm_andnot_ps(mask, best_dot));
        best_o = _mm_or_si128(_mm_and_si128(_mm_castps_si128(mask), vo),
                              _mm_andnot_si128(_mm_castps_si128(mask), best_o));

        dot = _mm_xor_ps(dot, SIGNMASK);
        mask = _mm_cmpgt_ps(dot, best_dot);
        vo = _mm_set1_epi32(o + 9);
        best_dot = _mm_or_ps(_mm_and_ps(mask, dot),
                             _mm_andnot_ps(mask, best_dot));
        best_o = _mm_or_si128(_mm_and_si128(_mm_castps_si128(mask), vo),
                              _mm_andnot_si128(_mm_castps_si128(mask), best_o));
      }


      // Update histograms
      // Replaced code:  Some code outside inner loop
      // float xp = ((float)x+0.5)/(float)sbin - 0.5;
      // float yp = ((float)y+0.5)/(float)sbin - 0.5;
      // int ixp = (int)floor(xp);
      // int iyp = (int)floor(yp);
      // float vx0 = xp-ixp;
      // float vy0 = yp-iyp;
      // float vx1 = 1.0-vx0;
      // float vy1 = 1.0-vy0;
      // v = sqrt(v);
      __m128 vy = _mm_set_ps1((float)y);
      vy = _mm_add_ps(vy, v0123);
      __m128 yp = _mm_add_ps(vy, point5);
      yp = _mm_div_ps(yp, vsbin);
      yp = _mm_sub_ps(yp, point5);
      __m128 fyp = _mm_floor_ps(yp);
      __m128i iyp = _mm_cvtps_epi32(fyp);
      __m128 vy0 = _mm_sub_ps(yp, fyp);
      __m128 vy1 = _mm_sub_ps(one, vy0);
      v = _mm_sqrt_ps(v);

      float vs[4];
      _mm_storeu_ps(vs, v);

      int best_os[4];
      _mm_storeu_si128((__m128i *)best_os, best_o);

      int iyps[4];
      _mm_storeu_si128((__m128i *)iyps, iyp);
      
      float vy0s[4];
      _mm_storeu_ps(vy0s, vy0);
      
      float vy1s[4];
      _mm_storeu_ps(vy1s, vy1);

      // Finish updating histograms sequentially.  This is a scatter.
      for (int yoff = 0; yoff < 4; yoff ++) {
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
      float *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
      float dy = *(s+1) - *(s-1);
      float dx = *(s+dims[0]) - *(s-dims[0]);
      float v = dx*dx + dy*dy;

      // second color channel
      s += dims[0]*dims[1];
      float dy2 = *(s+1) - *(s-1);
      float dx2 = *(s+dims[0]) - *(s-dims[0]);
      float v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += dims[0]*dims[1];
      float dy3 = *(s+1) - *(s-1);
      float dx3 = *(s+dims[0]) - *(s-dims[0]);
      float v3 = dx3*dx3 + dy3*dy3;

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
      float best_dot = 0;
      int best_o = 0;
      for (int o = 0; o < 9; o++) {
        float dot = uu[o]*dx + vv[o]*dy;
        if (dot > best_dot) {
          best_dot = dot;
          best_o = o;
        } else if (-dot > best_dot) {
          best_dot = -dot;
          best_o = o+9;
        }
      }

      // add to 4 histograms around pixel using linear interpolation
      float yp = ((float)y+0.5)/(float)sbin - 0.5;
      int iyp = (int)floor(yp);
      float vy0 = yp-iyp;
      float vy1 = 1.0-vy0;
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
    float *src1 = hist + o*blocks[0]*blocks[1];
    float *src2 = hist + (o+9)*blocks[0]*blocks[1];
    float *dst = norm;
    float *end = norm + blocks[1]*blocks[0];
    while (dst < end) {
      *(dst++) += (*src1 + *src2) * (*src1 + *src2);
      src1++;
      src2++;
    }
  }

  // compute features
  for (int x = 0; x < out[1]; x++) {
    for (int y = 0; y < out[0]; y++) {
      float *dst = feat + x*out[0] + y;      
      float *src, *p, n1, n2, n3, n4;

      p = norm + (x+1)*blocks[0] + y+1;
      n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + (x+1)*blocks[0] + y;
      n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + x*blocks[0] + y+1;
      n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + x*blocks[0] + y;      
      n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

      float t1 = 0;
      float t2 = 0;
      float t3 = 0;
      float t4 = 0;

      // contrast-sensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 18; o++) {
        float h1 = min(*src * n1, 0.2);
        float h2 = min(*src * n2, 0.2);
        float h3 = min(*src * n3, 0.2);
        float h4 = min(*src * n4, 0.2);
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
        float sum = *src + *(src + 9*blocks[0]*blocks[1]);
        float h1 = min(sum * n1, 0.2);
        float h2 = min(sum * n2, 0.2);
        float h3 = min(sum * n3, 0.2);
        float h4 = min(sum * n4, 0.2);
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
// image should be color with float values
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 2)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");
  plhs[0] = process(prhs[0], prhs[1]);
}



