"""
Vanilla_lstm based model without social pooling
introduce getSocialTensorMat()
"""

import torch
import torch.nn as nn
from criterion import mdn_loss, adefde, mdn_sample
from transform import rela_transform


class Interp_SocialLSTM(nn.Module):

    def __init__(self, args, infer=False):
        super(Interp_SocialLSTM, self).__init__()
        self.args = args
        self.num_gaussians = args.num_gaussians
        self.use_cuda = args.use_cuda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.infer = infer

        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length

        self.rnn_size = args.rnn_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.neighbor_size = args.neighbor_size
        self.lstm = nn.LSTMCell(self.rnn_size, self.rnn_size)
        self.embedding_input_layer = nn.Linear(4, self.rnn_size)
        self.embed_score = nn.Linear(self.rnn_size, 1)

        self.k_head = args.k_head

        self.embedding_mode = nn.Linear(self.k_head, self.rnn_size)
        self.embedding_rela_loc = nn.Linear(6, self.k_head * self.rnn_size)
        self.embedding_rela_loc_v = nn.Linear(6, self.k_head * self.rnn_size)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.softmax1 = torch.nn.Softmax(dim=0)

        # MDN
        self.pi = nn.Sequential(
            nn.Linear(self.rnn_size, self.num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(
            self.rnn_size,
            self.num_gaussians *
            self.output_size)
        self.mu = nn.Linear(
            self.rnn_size,
            self.num_gaussians *
            self.output_size)

    def getSocialTensorMat(
            self,
            nodes_current,
            vepo_grids,
            h_nodes_vv1,
            list_of_nodes):

        # Number of peds
        h_nodes_vv = h_nodes_vv1.clone()
        numNodes = nodes_current.size()[0]

        # Construct the variable
        social_tensor = torch.zeros(numNodes, self.rnn_size).to(self.device)
        for node in range(numNodes):
            if node not in list_of_nodes:
                continue
            agent_grid = vepo_grids[node]
            agent_loc = nodes_current[node:(node + 1), :]
            t_thres = 0

            while t_thres < 1:

                if numNodes == 1:
                    social_tensor[node] = torch.zeros([1, self.rnn_size])
                    t_thres += 1
                    continue
                else:
                    list_visible_peds = [idx for idx, i in enumerate(
                        agent_grid[:, 0].tolist()) if i == 1]
                    if len(list_visible_peds) == 0:
                        social_tensor[node] = torch.zeros([1, self.rnn_size])
                        t_thres += 1
                        continue

                    num_neighbors = len(list_visible_peds)

                    if t_thres == 0:
                        agent = h_nodes_vv[node].expand(
                            num_neighbors, self.rnn_size)
                    else:
                        agent = (
                                h_nodes_vv[node] +
                                social_tensor[node]).expand(
                            num_neighbors,
                            self.rnn_size)

                    agent = torch.unsqueeze(agent, 2)
                    agent = agent.expand(
                        num_neighbors, self.rnn_size, self.k_head)
                    agent = agent.reshape(
                        num_neighbors, self.k_head * self.rnn_size)

                    rela_loc = agent_loc - nodes_current[list_visible_peds]

                    input_interact_loc = self.relu(
                        self.embedding_rela_loc(rela_loc))
                    input_interact_loc_v = self.relu(
                        self.embedding_rela_loc_v(rela_loc))

                    score_rela = torch.mul(input_interact_loc, agent)
                    score_rela = score_rela.reshape(
                        num_neighbors, self.rnn_size, self.k_head)
                    score_rela = score_rela.permute(0, 2, 1)

                    score_rela = self.embed_score(score_rela)
                    score_rela = torch.squeeze(score_rela, 2)
                    att_rela = self.softmax(score_rela)

                    prob, pi = torch.max(att_rela, 1)
                    pi = pi.tolist()
                    t_id = []
                    for idx_pi in pi:
                        t_id.append(torch.tensor(idx_pi))

                    pi_end = torch.zeros(
                        (att_rela.size()[0], att_rela.size()[1])).to(
                        self.device)
                    for idx_p in range(pi_end.size()[0]):
                        pi_end[idx_p, t_id[idx_p]] = 1.0

                    att_rela = (pi_end - att_rela).detach() + att_rela

                    fea_mode = self.embedding_mode(att_rela)

                    idx_att_rela = []
                    for att_k in range(self.k_head):
                        idx_att_rela.append(att_rela[:, att_k] == 1)

                    att_rela = torch.unsqueeze(att_rela, 2)
                    att_rela = att_rela.expand(
                        num_neighbors, self.k_head, self.rnn_size)

                    input_interact_loc_v = input_interact_loc.reshape(
                        num_neighbors, self.rnn_size, self.k_head)
                    input_interact_loc_v = input_interact_loc_v.permute(
                        0, 2, 1)

                    fea_interact = torch.mul(att_rela, input_interact_loc_v)

                    for att_k in range(self.k_head):
                        if torch.sum(idx_att_rela[att_k]) == 0:
                            continue
                        att_fea_interact = fea_interact[:, att_k, :]
                        score_rela_tar = att_fea_interact[idx_att_rela[att_k]
                                         ] + fea_mode[idx_att_rela[att_k]]
                        score_rela_tar = torch.sum(score_rela_tar, 1)
                        score_rela_tar = torch.unsqueeze(score_rela_tar, 1)

                        att_rela_tar = self.softmax1(score_rela_tar)
                        att_rela_tar = att_rela_tar.expand(
                            att_rela_tar.size()[0], self.rnn_size)

                        social_tensor[node] = torch.sum(torch.mul(att_rela_tar,
                                                                  att_fea_interact[idx_att_rela[att_k]] + fea_mode[
                                                                      idx_att_rela[att_k]]), 0) + social_tensor[node]
                    t_thres += 1

        return social_tensor

    def train_one(
            self,
            nodes,
            nodesPresent,
            nodes_neighbors,
            hidden_states,
            cell_states, ):

        numNodes = hidden_states.size()[0]

        pi = torch.zeros(numNodes, self.num_gaussians).to(self.device)
        mu = torch.zeros(
            numNodes,
            self.num_gaussians,
            self.output_size).to(
            self.device)
        sigma = torch.zeros(
            numNodes,
            self.num_gaussians,
            self.output_size).to(
            self.device)

        list_of_nodes = torch.LongTensor(nodesPresent).to(self.device)

        hidden_states_current = torch.index_select(
            hidden_states, 0, list_of_nodes)
        cell_states_current = torch.index_select(cell_states, 0, list_of_nodes)

        nodes_current = torch.index_select(nodes, 0, list_of_nodes)

        social_tensor = self.getSocialTensorMat(
            nodes, nodes_neighbors, hidden_states, list_of_nodes)
        social_tensor = torch.index_select(social_tensor, 0, list_of_nodes)

        input_embedded = self.relu(self.embedding_input_layer(
            nodes_current[:, 0:4])) + social_tensor
        h_nodes, c_nodes = self.lstm(
            input_embedded, (hidden_states_current, cell_states_current))

        pi[list_of_nodes.data] = self.pi(h_nodes).view(-1, self.num_gaussians)
        mu[list_of_nodes] = self.mu(
            h_nodes).view(-1, self.num_gaussians, self.output_size)
        sigma[list_of_nodes] = torch.exp(self.sigma(
            h_nodes)).view(-1, self.num_gaussians, self.output_size)

        hidden_states[list_of_nodes.data] = h_nodes
        cell_states[list_of_nodes.data] = c_nodes

        return pi, mu, sigma, hidden_states, cell_states

    def forward(self, nodes_temp, nodesPresent, args, obs_len, pred_eln):

        nodes = nodes_temp[:, :, 1:]
        nodes = torch.from_numpy(nodes).float().to(self.device)
        numNodes = nodes.size()[1]

        ret_nodes_list = []
        ret_nodes = torch.zeros(
            obs_len + pred_eln,
            numNodes,
            5).to(
            self.device)
        ret_nodes[:obs_len, :, :] = torch.from_numpy(nodes_temp).float().to(self.device)[
                                    :obs_len, :, [0, 1, 2, 5, 6]].clone()

        loss_back = 0

        ade = []

        fde_pred = 0.0
        fde_gt = 0.0

        nodesLast1 = nodesPresent[0]
        nodesCurr1 = nodesPresent[obs_len + pred_eln - 1]
        nodeIDs1 = [value for value in nodesCurr1 if value in nodesLast1]

        list_of_nodes1 = torch.LongTensor(nodeIDs1).to(self.device)
        if len(list_of_nodes1) == 0:
            return 0, torch.Tensor([0]), torch.Tensor([0]), 0

        hidden_states_ende = torch.zeros(
            numNodes, args.rnn_size).to(
            self.device)
        cell_states_ende = torch.zeros(numNodes, args.rnn_size).to(self.device)

        for framenum in range(obs_len):
            nodes_neighbors = rela_transform(nodes[framenum, :, [0, 1, 4, 5]], nodesPresent[framenum],
                                             self.neighbor_size, self.use_cuda)
            pi, ol_mu, sigma, hidden_states_ende, cell_states_ende = self.train_one(
                nodes[framenum], nodesPresent[framenum], nodes_neighbors, hidden_states_ende, cell_states_ende, True)

            nodesLast = nodesPresent[framenum]
            nodesCurr = nodesPresent[framenum + 1]
            nodeIDs = [value for value in nodesCurr if value in nodesLast]

            list_of_nodes = torch.LongTensor(nodeIDs).to(self.device)
            if not len(list_of_nodes) == 0:
                loss = mdn_loss(pi, sigma, ol_mu,
                                nodes[framenum + 1, :, 0:2], list_of_nodes)
                loss_back += loss

        output, _ = mdn_sample(pi, sigma, ol_mu, list_of_nodes, self.infer)

        next_x = output[:, 0].data
        next_y = output[:, 1].data

        ret_nodes[framenum + 1,
        :,
        0] = torch.from_numpy(nodes_temp).float().to(self.device)[framenum,
             :,
             0]
        ret_nodes[framenum + 1, :, 1] = next_x
        ret_nodes[framenum + 1, :, 2] = next_y
        ret_nodes[framenum + 1, :, 3] = next_x + ret_nodes[framenum, :, 3]
        ret_nodes[framenum + 1, :, 4] = next_y + ret_nodes[framenum, :, 4]

        mu1 = output
        ade_err = adefde(mu1, nodes[framenum + 1, :, 0:2], list_of_nodes1)
        ade.append(ade_err)
        fde_pred += mu1
        fde_gt += nodes[framenum + 1, :, 0:2]

        last_location = nodes[framenum, :, 4:6]
        for framenum in range(obs_len, obs_len + pred_eln - 1):
            curr_nodes = torch.zeros(nodes_temp.shape[1], 6).to(self.device)

            curr_nodes[:, 0:2] = output
            curr_nodes[:, 2:4] = output / 0.4
            curr_nodes[:, 4:6] = output + last_location

            nodes_neighbors = rela_transform(
                curr_nodes[:, [0, 1, 4, 5]], nodesPresent[framenum], self.neighbor_size, self.use_cuda)
            pi, ol_mu, sigma, hidden_states_ende, cell_states_ende = self.train_one(
                curr_nodes, nodesPresent[framenum], nodes_neighbors, hidden_states_ende, cell_states_ende)

            nodesLast = nodesPresent[framenum]
            nodesCurr = nodesPresent[framenum + 1]
            nodeIDs = [value for value in nodesCurr if value in nodesLast]
            if len(nodeIDs) == 0:
                break
            list_of_nodes = torch.LongTensor(nodeIDs).to(self.device)

            output, _ = mdn_sample(pi, sigma, ol_mu, list_of_nodes, self.infer)

            next_x = output[:, 0].data
            next_y = output[:, 1].data

            ret_nodes[framenum + 1, :, 0] = ret_nodes[framenum, :, 0]
            ret_nodes[framenum + 1, :, 1] = next_x
            ret_nodes[framenum + 1, :, 2] = next_y
            ret_nodes[framenum + 1, :, 3] = next_x + ret_nodes[framenum, :, 3]
            ret_nodes[framenum + 1, :, 4] = next_y + ret_nodes[framenum, :, 4]

            if not len(list_of_nodes) == 0:
                loss = mdn_loss(pi, sigma, ol_mu,
                                nodes[framenum + 1, :, 0:2], list_of_nodes)
                loss_back += loss

            ade_err = adefde(output +
                             curr_nodes[:, 4:6], nodes[framenum +
                                                       1, :, 4:6], list_of_nodes1)
            ade.append(ade_err)
            if framenum == obs_len + pred_eln - 2:
                fde = ade_err
            fde_pred += output
            fde_gt += nodes[framenum + 1, :, 0:2]
            last_location = curr_nodes[:, 4:6]

        nodesLast = nodesPresent[0]
        nodesCurr = nodesPresent[obs_len + pred_eln - 1]
        nodeIDs = [value for value in nodesCurr if value in nodesLast]
        if len(nodeIDs) == 0:
            fde = 0.0
            ade = [torch.Tensor([0.0])].to(self.device)
        else:
            fde = adefde(fde_pred, fde_gt, list_of_nodes1)
        ret_nodes_list.append(ret_nodes)
        return loss_back, torch.mean(torch.stack(ade, 0)), fde, ret_nodes_list
