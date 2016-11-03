//
// Created by jianfei on 16-11-3.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "parallel_tree.h"

class ParallelTreeTest : public ::testing::Test {
protected:
    ParallelTree *tree;

    virtual void SetUp() {
        tree = new ParallelTree(3, std::vector<double>{1.0, 0.5});
        tree->SetThreshold(1);
    }

    virtual void TearDown() {
        delete tree;
    }
};

TEST_F(ParallelTreeTest, basic) {
    auto ret = tree->GetTree();
    EXPECT_EQ(ret.nodes.size(), 1);
    EXPECT_EQ(ret.num_nodes[0], 1);
    EXPECT_EQ(ret.num_instantiated[0], 0);

    auto inc_ret = tree->IncNumDocs(0);
    EXPECT_EQ(inc_ret.id, 2);
    EXPECT_EQ(inc_ret.pos[0], 0);
    EXPECT_EQ(inc_ret.pos[1], 0);
    EXPECT_EQ(inc_ret.pos[2], 0);

    inc_ret = tree->IncNumDocs(1);
    EXPECT_EQ(inc_ret.id, 3);
    EXPECT_EQ(inc_ret.pos[0], 0);
    EXPECT_EQ(inc_ret.pos[1], 0);
    EXPECT_EQ(inc_ret.pos[2], 1);

    inc_ret = tree->IncNumDocs(0);
    EXPECT_EQ(inc_ret.id, 5);
    EXPECT_EQ(inc_ret.pos[0], 0);
    EXPECT_EQ(inc_ret.pos[1], 1);
    EXPECT_EQ(inc_ret.pos[2], 2);

    ret = tree->GetTree();
    EXPECT_FLOAT_EQ(ret.nodes[0].log_path_weight, log(1.0/4));
    EXPECT_FLOAT_EQ(ret.nodes[1].log_path_weight, log(2.0/4 * 0.5/2.5));
    EXPECT_FLOAT_EQ(ret.nodes[2].log_path_weight, log(2.0/4 * 1.0/2.5));
    EXPECT_FLOAT_EQ(ret.nodes[3].log_path_weight, log(2.0/4 * 1.0/2.5));
    EXPECT_FLOAT_EQ(ret.nodes[4].log_path_weight, log(1.0/4 * 0.5/1.5));
    EXPECT_FLOAT_EQ(ret.nodes[5].log_path_weight, log(1.0/4 * 1.0/1.5));

    tree->IncNumDocs(3);
    tree->DecNumDocs(2);

    ret = tree->GetTree();
    EXPECT_EQ(ret.nodes.size(), 6);

    auto pos_map = tree->Compress();
    EXPECT_EQ(pos_map[1][0], 0);
    EXPECT_EQ(pos_map[1][1], 1);
    EXPECT_EQ(pos_map[2][0], 1);
    EXPECT_EQ(pos_map[2][1], 2);

    ret = tree->GetTree();
    EXPECT_EQ(ret.nodes.size(), 5);
    EXPECT_EQ(ret.nodes[0].id, 0);
    EXPECT_EQ(ret.nodes[1].id, 1);
    EXPECT_EQ(ret.nodes[2].id, 3);
    EXPECT_EQ(ret.nodes[3].id, 4);
    EXPECT_EQ(ret.nodes[4].id, 5);

    EXPECT_EQ(ret.num_instantiated[0], 1);
    EXPECT_EQ(ret.num_instantiated[1], 1);
    EXPECT_EQ(ret.num_instantiated[2], 1);
}