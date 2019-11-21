from torch.utils.tensorboard import SummaryWriter


class DoubleSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', light_log_dir=None, mode='full', **kwargs):
        super(DoubleSummaryWriter, self).__init__(log_dir=log_dir, comment=comment, **kwargs)
        self.light_writer = SummaryWriter(log_dir=light_log_dir, comment=comment, **kwargs)
        self.mode = mode

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super(DoubleSummaryWriter, self).add_scalar(tag, scalar_value, global_step, walltime)
        if self.mode == 'full':
            self.light_writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        super(DoubleSummaryWriter, self).add_text(tag, text_string, global_step, walltime)
        if self.mode == 'full':
            self.light_writer.add_text(tag, text_string, global_step, walltime)

    def change_mode(self, mode):
        if self.mode != mode:
            self.mode = mode
