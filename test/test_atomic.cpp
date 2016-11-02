#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "atomic_vector.h"

TEST(AtomicVector, EmplaceBack) {
    AtomicVector<int> a;
    a.EmplaceBack(1);
    a.EmplaceBack(2);
    a.EmplaceBack(3);
    a.EmplaceBack(4);

    for (size_t i = 0; i < a.Size(); i++)
        EXPECT_EQ(a.Get(i), i+1);
}
