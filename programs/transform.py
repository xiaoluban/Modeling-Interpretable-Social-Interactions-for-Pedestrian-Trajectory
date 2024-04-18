"""
find "neighbors" according to pre-defined neighbor_size
"""

import torch


def rela_transform(nodes, list_of_nodes, neighbor_size, use_cuda):
    numNodes = nodes.size()[0]

    vepo_grid = torch.zeros(numNodes, numNodes, 1)
    if use_cuda:
        vepo_grid = vepo_grid.cuda()

    if len(list_of_nodes) == 1 or len(list_of_nodes) == 0:
        return vepo_grid

    agent_nodes = []
    others = []
    for idx_node in range(numNodes):
        agent = nodes[idx_node, 2:4]
        agent = agent.expand(nodes.size()[0], 2)
        agent_nodes.append(agent)
        others.append(nodes[:, 2:4])

    agent_nodes = torch.cat(agent_nodes, 0)

    others = torch.cat(others, 0)
    dist_neighbor = torch.sqrt(
        (agent_nodes[:, 0] - others[:, 0]) * (agent_nodes[:, 0] - others[:, 0]) + (agent_nodes[:, 1] - others[:, 1]) * (
                agent_nodes[:, 1] - others[:, 1]))
    dist_neighbor_list = dist_neighbor < neighbor_size
    dist_neighbor_list = dist_neighbor_list.view(numNodes, numNodes, 1)
    vepo_grid = dist_neighbor_list

    for idx_node in list_of_nodes:
        vepo_grid[idx_node, idx_node, 0] = 0
        for idx_node_other in range(numNodes):
            if idx_node_other not in list_of_nodes:
                vepo_grid[idx_node, idx_node_other, 0] = 0

    return vepo_grid.type(torch.FloatTensor)
