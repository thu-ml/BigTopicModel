#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "atomic_vector.h"
#include "atomic_matrix.h"

TEST(AtomicVector, EmplaceBack) {
    AtomicVector<int> a;
    a.EmplaceBack(1);
    a.EmplaceBack(2);
    a.EmplaceBack(3);
    a.EmplaceBack(4);

    for (size_t i = 0; i < a.Size(); i++)
        EXPECT_EQ(a.Get(i), i+1);
}

TEST(AtomicMatrix, Basic) {
    AtomicMatrix<int> b;
    b.SetR(2); b.SetC(2);
    b.Inc(0, 0);
    b.Inc(1, 1);
    b.Inc(1, 1);

    EXPECT_EQ(b.Get(0, 0), 1);
    EXPECT_EQ(b.Get(0, 1), 0);
    EXPECT_EQ(b.Get(1, 0), 0);
    EXPECT_EQ(b.Get(1, 1), 2);

    b.SetR(10);
    EXPECT_EQ(b.Get(0, 0), 1);
    EXPECT_EQ(b.Get(0, 1), 0);
    EXPECT_EQ(b.Get(1, 0), 0);
    EXPECT_EQ(b.Get(1, 1), 2);

    b.PermuteColumns(std::vector<int>{1, 0});
    EXPECT_EQ(b.Get(0, 0), 0);
    EXPECT_EQ(b.Get(0, 1), 1);
    EXPECT_EQ(b.Get(1, 0), 2);
    EXPECT_EQ(b.Get(1, 1), 0);
}