import torch


class FootPrinter:
    def __init__(self, device="cpu", encoder=None):
        self.device = device
        self.encoder = encoder

    def update_encoder(self, encoder):
        self.encoder = encoder

    def culc_footprint(self, local_data, dataloader=True):
        if dataloader is True:
            latent_representation = []
            for batch_idx, (x, labels) in enumerate(local_data):
                x, labels = x.to(self.device), labels.to(self.device)
                output = self.encoder(x)
                latent_representation.append(output)
            latent_representation = torch.cat(latent_representation)
        else:
            latent_representation = self.encoder(local_data)

        u = torch.mean(latent_representation, axis=0)
        sigma = torch.std(latent_representation, axis=0)
        footprint = (u, sigma)
        return footprint

    def kldiv_between_server_and_client(self, server_footprint, client_footprint):
        server_u, server_sigma = server_footprint
        client_u, client_sigma = client_footprint
        kl = torch.log(server_sigma / client_sigma) + (
            (client_sigma ** 2) + (client_u - server_u) ** 2
        ) / (2 * (server_sigma ** 2))

        return torch.mean(kl).item()
