import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Based on https://github.com/codymlewis/hale/


class STD_DAGMM(nn.Module):
    """
    Based on https://github.com/danieltan07/dagmm
    and https://github.com/datamllab/pyodds
    """

    def __init__(self, in_len, device, n_gmm=2, latent_dim=4):
        super().__init__()
        # AC encode
        self.encoder = nn.ModuleList(
            [
                nn.Linear(in_len, 60),
                nn.ReLU(),
                nn.Linear(60, 30),
                nn.ReLU(),
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            ]
        ).eval()
        # AC decode
        self.decoder = nn.ModuleList(
            [
                nn.Linear(1, 10),
                nn.Tanh(),
                nn.Linear(10, 30),
                nn.Tanh(),
                nn.Linear(30, 60),
                nn.Tanh(),
                nn.Linear(60, in_len),
            ]
        ).eval()
        # GMM
        self.estimator = nn.ModuleList(
            [
                nn.Linear(latent_dim, 10),
                nn.Tanh(),
                nn.Dropout(p=0.5),
                nn.Linear(10, n_gmm),
                nn.Softmax(dim=1),
            ]
        ).eval()
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))
        # Other configuration
        self.device = device
        self.to(device)

    def to_var(self, x):
        return Variable(x).to(self.device)

    def relative_euclidean_distance(self, a, b, dim=1):
        return (a - b).norm(2, dim=dim) / torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def encode(self, x):
        for f in self.encoder:
            x = f(x)
        return x

    def decode(self, x):
        for f in self.decoder:
            x = f(x)
        return x

    def estimate(self, x):
        for f in self.estimator:
            x = f(x)
        return x

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        rec_cosine = F.cosine_similarity(
            x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1
        )
        rec_euclidean = self.relative_euclidean_distance(
            x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1
        )
        rec_std = torch.std(x.view(x.shape[0], -1), dim=1)
        z = torch.cat(
            [
                enc,
                rec_euclidean.unsqueeze(-1),
                rec_cosine.unsqueeze(-1),
                rec_std.unsqueeze(-1),
            ],
            dim=1,
        )
        gamma = self.estimate(z)
        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / N
        self.phi = phi.data
        mu = torch.sum(
            gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0
        ) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0
        ) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = Variable(self.phi)
        if mu is None:
            mu = Variable(self.mu)
        if cov is None:
            cov = Variable(self.cov)
        k, d, _ = cov.size()
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            cov_k = cov[i] + self.to_var(torch.eye(d) * eps)
            pinv = np.linalg.pinv(cov_k.data.cpu().numpy())
            cov_inverse.append(Variable(torch.from_numpy(pinv)).unsqueeze(0))
            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            determinant = np.prod(
                np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None)
            )
            det_cov.append(determinant)
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())
        cov_inverse = torch.cat(cov_inverse, dim=0).to(self.device)
        det_cov = Variable(torch.from_numpy(np.float32(np.array(det_cov))))
        exp_term_tmp = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu,
            dim=-1,
        )
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(
                self.to_var(phi.unsqueeze(0))
                * exp_term
                / (torch.sqrt(self.to_var(det_cov)) + eps).unsqueeze(0),
                dim=1,
            )
            + eps
        )
        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag

    def predict(self, X):
        E = torch.tensor([], device=self.device)
        for x in X:
            _, _, z, _ = self(x.unsqueeze(0))
            e, _ = self.compute_energy(z, size_average=False)
            E = torch.cat((E, e))
        return E

    def fit(self, x, epochs=1, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.0001)
        for i in range(epochs):
            self.train()
            enc, dec, z, gamma = self(x)
            loss, sample_energy, recon_error, cov_diag = self.loss_function(
                x, dec, z, gamma, 0.1, 0.005
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            optimizer.step()

            logging.info(f"STD-DAGMM: epoch {i} - {loss.item()}")
