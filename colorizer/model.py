import torch
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn, optim
from torchvision.models.resnet import resnet34


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator
    """

    def __init__(self):
        """
        Constructor for PatchGAN discriminator
        """
        super().__init__()

        filters = 64
        downsampling_layers = 3

        # First layer: Convolutional layer to map RGB image to feature space
        layers = [nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=4, stride=2, padding=1, bias=True)]
        layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        model = [nn.Sequential(*layers)]

        # Downsampling layers: Increase the number of filters while reducing spatial dimensions
        for index in range(downsampling_layers):
            layers = [
                nn.Conv2d(
                    in_channels=filters * 2**index,
                    out_channels=filters * 2 ** (index + 1),
                    kernel_size=4,
                    stride=1 if index == (downsampling_layers - 1) else 2,
                    padding=1,
                    bias=False,
                )
            ]
            layers += [nn.BatchNorm2d(num_features=filters * 2 ** (index + 1))]
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            model += [nn.Sequential(*layers)]

        # Final layer: Convolution to map to a single channel output
        model += [
            nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels=filters * 2**downsampling_layers,
                        out_channels=1,
                        kernel_size=4,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                ]
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass of PatchGAN discriminator
        """
        return self.model(x)


class GANLoss(nn.Module):
    """
    Class to calculate GAN loss
    """

    def __init__(self):
        """
        Constructor for GANLoss class
        """
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, preds, target_is_real):
        """
        Create labels for discriminator
        """
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def forward(self, preds, target_is_real):
        """
        Calculate loss
        """
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


def init_model(patch_discriminator: PatchDiscriminator, device) -> PatchDiscriminator:
    """
    Initialize weights of network
    """

    def init_layer(layer):
        """
        Initialize weights of layer depending on layer type
        """
        classname = layer.__class__.__name__
        if hasattr(layer, "weight") and "Conv" in classname:
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)

            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0.0)

    patch_discriminator = patch_discriminator.to(device)
    patch_discriminator.apply(init_layer)
    return patch_discriminator


class ColorizationModel(nn.Module):
    """
    Class for colorization model
    """

    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.l1_loss = 100.0

        # Generator
        # cut last two layers of resnet34 to remove classification head
        self.body = create_body(resnet34(), pretrained=True, n_in=1, cut=-2)
        self.generator = DynamicUnet(self.body, 2, (256, 256)).to(self.device)

        # Discriminator
        self.discriminator = init_model(PatchDiscriminator(), self.device)

        # define loss functions
        self.gan_criterion = GANLoss().to(self.device)
        self.l1_criterion = nn.L1Loss()

        # define optimizers for generator and discriminator
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # setup loss meters
        self.loss_discriminator_fake = 0.0
        self.loss_discriminator_real = 0.0
        self.loss_discriminator = 0.0
        self.loss_generator_gan = 0.0
        self.loss_generator_l1 = 0.0
        self.loss_generator = 0.0

        # setup input
        self.l = None
        self.ab = None

    def get_device(self):
        """
        Get device of model
        """
        return self.device

    def set_requires_grad(self, model, requires_grad) -> None:
        """
        Enable or disable gradient calculation for model

        Parameters:
        model (nn.Module): Model to enable/disable gradient calculation
        requires_grad (bool): Enable or disable gradient calculation
        """
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        """
        Prepare input data

        Parameters:
        data (dict): Input data
        """
        self.l = data["l"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def optimize(self):
        fake_colorized_image = self.generator(self.l)

        self.discriminator.train()
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_discriminator.zero_grad()

        # backward of discriminator
        fake_image = torch.cat([self.l, fake_colorized_image], dim=1)
        fake_preds = self.discriminator(fake_image.detach())
        self.loss_discriminator_fake = self.gan_criterion(fake_preds, False)
        real_image = torch.cat([self.l, self.ab], dim=1)
        real_preds = self.discriminator(real_image)
        self.loss_discriminator_real = self.gan_criterion(real_preds, True)
        self.loss_discriminator = (self.loss_discriminator_fake + self.loss_discriminator_real) * 0.5
        self.loss_discriminator.backward()
        self.optimizer_discriminator.step()

        # forward of generator
        self.generator.train()
        self.set_requires_grad(self.discriminator, False)
        self.optimizer_generator.zero_grad()
        fake_image = torch.cat([self.l, fake_colorized_image], dim=1)
        fake_preds = self.discriminator(fake_image)
        self.loss_generator_gan = self.gan_criterion(fake_preds, True)
        self.loss_generator_l1 = self.l1_criterion(fake_colorized_image, self.ab) * self.l1_loss
        self.loss_generator = self.loss_generator_gan + self.loss_generator_l1
        self.loss_generator.backward()
        self.optimizer_generator.step()
