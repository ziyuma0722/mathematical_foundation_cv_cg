from graph_cut import GraphCut
import numpy as np


def main():
    # TODO: Implement Task 1
    graph = GraphCut(3, 10)
    
    # node 0 the top one, node 2 the bottom one
    # unary edges
    unaries = np.array([[9,4],[7,7],[5,8]])
    graph.set_unary(unaries)

    # pairwise edges
    pairwise = np.array([[0,1,3,2],[1,2,5,1]])
    graph.set_pairwise(pairwise)

    minCut = graph.minimize()
    labels = graph.get_labeling()

    print(minCut)
    print(labels)
    pass


if __name__ == '__main__':
    main()


# The result is 19.0 for min-cut value and [False, False, False] for segmentation, 
# meaning all nodes belong to the source.

# manual verification: in total 2^3 combinations
# [false, false, false] all three nodes belong to the source, 4+7+8 = 19
# [true, true, true] all three nodes belong to the target, 9+7+5 = 21
# [false, true, true] only node 0 belongs to the source, 4+3+2+7+5 = 21
# [true, false, true] only node 1 belongs to the source, 9+3+2+1+5+7+5 = 32
# [true, true, false] only node 2 belongs to the source, 9+7+1+5+8 = 30
# [true, false, false] only node 0 not belongs to the source, 9+3+2+7+8 = 29
# [false, true, false] only node 1 not belongs to the source, 4+3+2+7+1+5+8 = 30
# [false, false, true] only node 2 not belongs to the source, 4+7+1+5+5 = 22

# [false, false, false] is indeed the mincut segmentation