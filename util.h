
#ifdef DEBUG
#undef NDEBUG
#include <assert.h>
#define ASSERT(b) assert((b))
#define dbg_printf(...) printf(__VA_ARGS__)
inline unsigned long rdtsc(){
  unsigned int lo,hi;
  __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
  return ((unsigned long)hi << 32) | lo;
}
unsigned long ___cycle___counter___;
#define cycle_begin do {___cycle___counter___ = rdtsc();} while (0);
#define cycle_end printf("Cycles elapsed = %ul\n", rdtsc()-___cycle___counter___)
#else
#define ASSERT(b) (void)(b);
#define dbg_printf(...) do {} while (0);
#define cycle_begin
#define cycle_end
#endif  // DEBUG

