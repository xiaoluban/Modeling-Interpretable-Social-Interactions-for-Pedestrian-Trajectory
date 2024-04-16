"""
Criterion for the structural RNN model
"""

import torch
import numpy as np
from torch.distributions import Categorical


def mdn_loss(pi1, sigma1, mu1, data1, list_of_nodes):
    """
    Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG parameters.
    """
    pi = torch.index_select(pi1, 0, list_of_nodes)
    sigma = torch.index_select(sigma1, 0, list_of_nodes)
    mu = torch.index_select(mu1, 0, list_of_nodes)
    data = torch.index_select(data1, 0, list_of_nodes)

    prob = pi * mdn_gaussian_2d_likelihood(sigma, mu, data)

    epsilon = 1e-20
    nll = -torch.log(torch.clamp(torch.mean(prob, 1), min=epsilon))
    return torch.mean(nll)


def mdn_sample(pi1, sigma1, mu1, list_of_nodes, infer):
    """
    Draw samples from a MoG during test
    """
    numNodes = pi1.size()[0]
    out_feature = sigma1.size()[2]

    pi = torch.index_select(pi1, 0, list_of_nodes)
    sigma = torch.index_select(sigma1, 0, list_of_nodes)
    mu = torch.index_select(mu1, 0, list_of_nodes)

    if infer:
        prob, pi = torch.max(pi, 1)
        pi = pi.tolist()
        pis = []
        for idx_pi in pi:
            pis.append(torch.tensor(idx_pi))
    else:
        categorical = Categorical(pi)
        pis = list(categorical.sample().data)

    sample = sigma.data.new(sigma.size()[0], sigma.size()[2]).normal_()
    sample_temp = sigma.data.new(sigma.size()[0], sigma.size()[2]).normal_()

    for i, idx in enumerate(pis):
        sample_temp[i] = sample[i].mul(sigma[i, idx]).add(mu[i, idx])

    sample1 = torch.zeros(numNodes, out_feature)
    sample1[list_of_nodes] = sample_temp

    return sample1, sample_temp


def adefde(predict, targets, nodesPresent):
    """
    Calculate the ade/fde error
    """
    predict = torch.index_select(predict, 0, nodesPresent)
    targets = torch.index_select(targets, 0, nodesPresent)

    diff = predict - targets
    ade = torch.sum(
        torch.sqrt(
            torch.sum(
                torch.mul(
                    diff,
                    diff),
                1)),
        0) / diff.size()[0]

    return ade


def mdn_gaussian_2d_likelihood(sigma, mu, target):
    """
    """
    num_gaussians = sigma.size()[1]
    num_nodes = sigma.size()[0]
    gaussian_prob = torch.zeros(num_nodes, num_gaussians)

    for idx_gaussian in range(num_gaussians):
        mux = mu[:, idx_gaussian, 0]
        muy = mu[:, idx_gaussian, 1]
        sx = sigma[:, idx_gaussian, 0] + torch.exp(torch.tensor(-6.0))
        sy = sigma[:, idx_gaussian, 1] + torch.exp(torch.tensor(-6.0))
        corr = 0.0

        normx = target[:, 0] - mux
        normy = target[:, 1] - muy
        sxsy = sx * sy

        for i_sxy in range(sxsy.size()[0]):
            if torch.isnan(sxsy[i_sxy]):
                sxsy[i_sxy] = 1.0

        z = (normx / sx) ** 2 + (normy / sy) ** 2
        negRho = torch.tensor(1 - corr ** 2)

        result = torch.exp(-z / (2 * negRho))
        denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

        result = result / denom
        gaussian_prob[:, idx_gaussian] = result

    return gaussian_prob
