from colorizer.model import ColorizationModel


class LossStatistic:
    """
    Class for storing loss statistic
    """

    def __init__(self):
        """
        Constructor for Trainstat class
        """
        self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, count=1) -> None:
        """
        Add new value to statistics

        Parameters:
        val (float): Value
        count (int): Number of values
        """
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


class LossStatistics:
    """
    Class for storing loss statistics
    """

    def __init__(self):
        """
        Constructor for Trainstat class
        """
        self.loss_d_fake = LossStatistic()
        self.loss_d_real = LossStatistic()
        self.loss_d = LossStatistic()
        self.loss_g_gan = LossStatistic()
        self.loss_g_l1 = LossStatistic()
        self.loss_g = LossStatistic()

    def update(self, model: ColorizationModel, count: int) -> None:
        """
        Update loss statistics

        Parameters:
        model (ColorizationModel): Model
        count (int): Number of values
        """
        self.loss_d_fake.update(model.loss_discriminator_fake.item(), count=count)
        self.loss_d_real.update(model.loss_discriminator_real.item(), count=count)
        self.loss_d.update(model.loss_discriminator.item(), count=count)
        self.loss_g_gan.update(model.loss_generator_gan.item(), count=count)
        self.loss_g_l1.update(model.loss_generator_l1.item(), count=count)
        self.loss_g.update(model.loss_generator.item(), count=count)

    def print(self):
        """
        Print loss statistics
        """
        print(f"loss_d_fake: {self.loss_d_fake.avg:.5f}")
        print(f"loss_d_real: {self.loss_d_real.avg:.5f}")
        print(f"loss_d: {self.loss_d.avg:.5f}")
        print(f"loss_g_gan: {self.loss_g_gan.avg:.5f}")
        print(f"loss_g_l1: {self.loss_g_l1.avg:.5f}")
        print(f"loss_g: {self.loss_g.avg:.5f}")
