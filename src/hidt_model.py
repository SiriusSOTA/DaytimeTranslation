import torch
import torch.nn as nn
import torch.optim as optim

from src.hidt_components import (
    ConditionalDiscriminator,
    ContentEncoder,
    Decoder,
    StyleEncoder,
    UnconditionalDiscriminator,
    MetricCalculator,
)


class HiDTModel(nn.Module):

    def __init__(self,
                 config: dict,
                 device: torch.device = torch.device("cpu"),
                 learning_rate: float = 3e-4,
                 verbose: bool = True,
                 ):
        super().__init__()

        self.config = config
        self.device = device
        self.verbose = verbose

        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.generator = Decoder()
        self.cond_discriminator = ConditionalDiscriminator()
        self.uncond_discriminator = UnconditionalDiscriminator()

        self.criterion_dist = MetricCalculator.criterion_dist
        self.criterion_rec = nn.L1Loss()
        self.criterion_seg = nn.CrossEntropyLoss()
        self.criterion_c = nn.L1Loss()
        self.criterion_s = nn.L1Loss()
        self.criterion_cyc = nn.L1Loss()
        self.criterion_seg_r = nn.CrossEntropyLoss()
        self.criterion_c_r = nn.L1Loss()
        self.criterion_s_r = nn.L1Loss()
        self.criterion_rec_r = nn.L1Loss()

        self.lambdas = config["lambdas"]

    def forward(self,
                x: torch.Tensor
                ):
        raise NotImplementedError("forward not implemented for this model")

    def generator_step(self,
                       x: torch.Tensor,
                       x_prime: torch.Tensor
                       ):
        # autoencoding branch

        c, h = self.content_encoder(x)
        s = self.style_encoder(x)
        loss_dist = MetricCalculator.criterion_dist(s)
        x_tilde, _ = self.generator(content=c, style=s, hooks=h)
        loss_rec = self.criterion_rec(x_tilde, x)

        # swapping branch
        s_prime = self.style_encoder(x_prime)
        x_hat, m_hat = self.generator(content=c, style=s_prime, hooks=h)

        loss_seg = 0  # TODO: self.criterion_seg(m_hat, m)
        c_hat, h_hat = self.content_encoder(x_hat)
        s_hat = self.style_encoder(x_hat)
        loss_c = self.criterion_c(c_hat, c)
        loss_s = self.criterion_s(s_hat, s_prime)

        c_prime, h_prime = self.content_encoder(x_prime)
        x_prime_hat, _ = self.generator(content=c_prime, style=s,
                                        hooks=h_prime)
        s_prime_hat = self.style_encoder(x_prime_hat)
        x_hat_tilde, _ = self.generator(content=c_hat, style=s_prime_hat,
                                        hooks=h_hat)
        loss_cyc = self.criterion_cyc(x_hat_tilde, x)

        # noise branch
        s_r = torch.randn(len(x), 3).to(self.device)
        x_r, m_r = self.generator(content=c, style=s_r, hooks=h)
        loss_seg_r = 0  # TODO: self.criterion_seg_r(m_r, m)
        c_r_tilde, h_r_tilde, = self.content_encoder(x_r)
        s_r_tilde = self.style_encoder(x_r)

        loss_c_r = self.criterion_c_r(c_r_tilde, c)
        loss_s_r = self.criterion_s_r(s_r_tilde, s_r)

        x_r_tilde, _ = self.generator(content=c_r_tilde, style=s_r_tilde,
                                      hooks=h_r_tilde)
        loss_rec_r = self.criterion_rec_r(x_r_tilde, x_r)

        # all discriminators
        du_x_hat = self.uncond_discriminator(x_hat)
        dc_x_hat = self.cond_discriminator(x_hat, s_prime.clone().detach())

        loss_adv = (
                MetricCalculator.criterion_adv(
                    du_x_hat,
                    torch.ones_like(du_x_hat)
                ) +
                MetricCalculator.criterion_adv(
                    dc_x_hat,
                    torch.ones_like(dc_x_hat),
                )
        )

        du_x_r = self.uncond_discriminator(x_r)
        dc_x_r = self.cond_discriminator(x_r, s_r.clone().detach())

        loss_adv_r = (
                MetricCalculator.criterion_adv(du_x_r,
                                               torch.ones_like(du_x_r)) +
                MetricCalculator.criterion_adv(dc_x_r, torch.ones_like(dc_x_r))
        )

        loss_terms = [
            loss_adv + loss_adv_r,
            loss_rec + loss_rec_r + loss_cyc,
            loss_c + loss_c_r,
            loss_s,
            loss_s_r,
            loss_dist,
        ]

        loss = 0
        for i, _ in enumerate(loss_terms):
            loss += self.lambdas[i] * loss_terms[i]

        info = {"loss": loss,
                "adversarial loss": loss_adv + loss_adv_r,
                "image reconstruction loss": loss_rec + loss_rec_r + loss_cyc,
                "content reconstruction loss": loss_c + loss_c_r,
                "style reconstruction loss": loss_s,
                "random style reconstruction loss": loss_s_r,
                "style distribution loss": loss_dist}
        return {"generator": info}

    def discriminator_step(self,
                           x: torch.Tensor,
                           x_prime: torch.Tensor
                           ):
        c, h = self.content_encoder(x)
        s = self.style_encoder(x)
        x_tilde, m = self.generator(content=c, style=s, hooks=h)

        # swapping branch
        s_prime = self.style_encoder(x_prime)
        x_hat, m_hat = self.generator(content=c, style=s_prime, hooks=h)

        c_hat, h_hat = self.content_encoder(x_hat)
        s_hat = self.style_encoder(x_hat)

        c_prime, h_prime = self.content_encoder(x_prime)
        x_prime_hat, _ = self.generator(content=c_prime, style=s,
                                        hooks=h_prime)
        s_prime_hat = self.style_encoder(x_prime_hat)
        x_hat_tilde, _ = self.generator(content=c_hat, style=s_prime_hat,
                                        hooks=h_hat)

        # noise branch
        s_r = torch.randn(len(x), 3).to(self.device)
        x_r, m_r = self.generator(content=c, style=s_r, hooks=h)
        c_r_tilde, h_r_tilde, = self.content_encoder(x_r)
        s_r_tilde = self.style_encoder(x_r)
        x_r_tilde, _ = self.generator(content=c_r_tilde, style=s_r_tilde,
                                      hooks=h_r_tilde)

        # all discriminators
        du_x_hat = self.uncond_discriminator(x_hat)
        dc_x_hat = self.cond_discriminator(x_hat, s_prime.clone().detach())
        loss_adv_hat = (
                MetricCalculator.criterion_adv(
                    du_x_hat,
                    torch.zeros_like(du_x_hat)
                )
                + MetricCalculator.criterion_adv(
            dc_x_hat,
            torch.zeros_like(dc_x_hat),
        )
        )

        du_x_r = self.uncond_discriminator(x_r)
        dc_x_r = self.cond_discriminator(x_r, s_r.clone().detach())

        loss_adv_r = (
                MetricCalculator.criterion_adv(
                    du_x_r,
                    torch.zeros_like(du_x_r),
                ) +
                MetricCalculator.criterion_adv(
                    dc_x_r,
                    torch.zeros_like(dc_x_r),
                )
        )
        du_x = self.uncond_discriminator(x)
        dc_x = self.cond_discriminator(x, s.clone().detach())
        loss_adv_real = (
                MetricCalculator.criterion_adv(du_x, torch.ones_like(du_x)) +
                MetricCalculator.criterion_adv(dc_x, torch.ones_like(dc_x))
        )

        loss = loss_adv_hat + loss_adv_r + loss_adv_real

        info = {
            "loss": loss,
            "loss_adv_hat": loss_adv_hat,
            "loss_adv_r": loss_adv_r,
            "loss_adv_real": loss_adv_real
        }

        return {"discriminator": info}

    def training_step(self,
                      batch,
                      step: str
                      ):
        self.train()
        x, x_prime = batch
        x = x.to(self.device)
        x_prime = x_prime.to(self.device)

        if step == "generator":  # generator step
            return self.generator_step(x, x_prime)
        elif step == "discriminator":
            return self.discriminator_step(x, x_prime)

        raise ValueError("step should be generator or discriminator"
                         ", received: " + str(step))

    def validation_step(self, batch):
        self.eval()
        loss = self.training_step(batch=batch, step="generator")

        return loss

    @torch.no_grad()
    def sample(self, batch):
        self.eval()

        x, x_prime = batch
        x = x.to(self.device)
        x_prime = x_prime.to(self.device)
        c, h = self.content_encoder(x)
        s = self.style_encoder(x)
        x_tilde, _ = self.generator(content=c, style=s, hooks=h)
        s_prime = self.style_encoder(x_prime)
        x_hat, _ = self.generator(content=c, style=s_prime, hooks=h)
        s_r = torch.randn(len(x), 3).to(self.device)
        x_r, _ = self.generator(content=c, style=s_r, hooks=h)

        return torch.cat([x, x_hat, x_prime, x_tilde, x_r])

    def configure_optimizers(self):
        params_g = list(self.generator.parameters()) + \
                   list(self.content_encoder.parameters()) + \
                   list(self.style_encoder.parameters())

        optimizer_g = optim.Adam(
            params=params_g,
            lr=self.config["learning_rate"],
        )
        params_d = list(self.cond_discriminator.parameters()) + \
                   list(self.uncond_discriminator.parameters())

        optimizer_d = optim.Adam(
            params=params_d,
            lr=self.config["learning_rate"],
        )

        optimizers = [
            {"label": "generator", "value": optimizer_g},
            {"label": "discriminator", "value": optimizer_d},
        ]
        schedulers = []

        return optimizers, schedulers
