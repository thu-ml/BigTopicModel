#ifndef __XORSHIFT
#define __XORSHIFT

#include <stdint.h>

struct xorshift {
    typedef uint_fast32_t result_type;
    /* The state must be seeded so that it is not everywhere zero. */
    uint64_t s[2];

    xorshift() { s[0] = 1234567; s[1] = 7654321; }
    void seed(uint64_t s1, uint64_t s2) {
        s[0] = s1; s[1] = s2;
    }

    uint32_t min() { return 0; }
    uint32_t max() { return 4294967295; }

    uint32_t operator () () {
        return sample();
    }

    uint32_t sample() {
        uint64_t x = s[0];
        uint64_t const y = s[1];
        s[0] = y;
        x ^= x << 23; // a
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
        return s[1] + y;
    }

    void discard() {
        sample();
    }
};

#endif
