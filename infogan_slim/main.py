import trainer as t


class Config(object):
    def __init__(self, **kwargs):
        # configuration for building the network
        self.x_dim = kwargs.get("x_dim", 784)
        self.x_height = kwargs.get("x_height", 28)
        self.x_weight = kwargs.get("x_weight", 28)
        self.x_depth = kwargs.get("x_depth", 1)

        self.c_dim = kwargs.get("c_dim", 12)
        self.z_dim = kwargs.get("z_dim", 62)

        self.batch_size = kwargs.get("batch_size", 128)
        self.lr = kwargs.get("lr", 0.0002)
        self.beta1 = kwargs.get("beta1", 0.5)  # recommend value in dcgan paper

        self.logdir = kwargs.get("logdir", "./logdir")
        self.sampledir = kwargs.get("sampledir", "./sample_image")

        self.max_steps = kwargs.get("max_steps", 100000)
        self.sample_every_n_steps = kwargs.get("sample_every_n_steps", 500)
        self.summary_every_n_steps = kwargs.get("summary_every_n_steps", 10)
        self.savemodel_every_n_steps = kwargs.get("savemodel_every_n_steps", 2000)
        self.save_model_secs = kwargs.get("save_model_secs", 1200)
        self.checkpoint_basename = kwargs.get("checkpoint_basename", "./infogan/"+str(self.lr)+"/model.ckpt")

        # configuration for the dataset queue
        self.min_after_dequeue = kwargs.get("min_after_dequeue", 5000)


def main():
    config = Config()
    trainer = t.Trainer(config)
    trainer.fit()

if __name__ == '__main__':
    main()