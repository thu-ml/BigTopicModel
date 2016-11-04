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

    auto a_sess = a.GetSession();
    for (size_t i = 0; i < a_sess.Size(); i++)
        EXPECT_EQ(a_sess.Get(i), i+1);
}

TEST(AtomicMatrix, Basic) {
    AtomicMatrix<int> b;
    b.SetR(2); b.SetC(2);

    {
        auto b_sess = b.GetSession();
        b_sess.Inc(0, 0);
        b_sess.Inc(1, 1);
        b_sess.Inc(1, 1);

        EXPECT_EQ(b_sess.Get(0, 0), 1);
        EXPECT_EQ(b_sess.Get(0, 1), 0);
        EXPECT_EQ(b_sess.Get(1, 0), 0);
        EXPECT_EQ(b_sess.Get(1, 1), 2);
    }

    b.SetR(10);
    {
        auto b_sess = b.GetSession();
        EXPECT_EQ(b_sess.Get(0, 0), 1);
        EXPECT_EQ(b_sess.Get(0, 1), 0);
        EXPECT_EQ(b_sess.Get(1, 0), 0);
        EXPECT_EQ(b_sess.Get(1, 1), 2);
    }

    b.PermuteColumns(std::vector<int>{1, 0});
    {
        auto b_sess = b.GetSession();
        EXPECT_EQ(b_sess.Get(0, 0), 0);
        EXPECT_EQ(b_sess.Get(0, 1), 1);
        EXPECT_EQ(b_sess.Get(1, 0), 2);
        EXPECT_EQ(b_sess.Get(1, 1), 0);
    }
}