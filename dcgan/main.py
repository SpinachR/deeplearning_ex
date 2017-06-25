import trainer as t


class Config(object):
    def __init__(self, **kwargs):
        # configuration for building the network
        self.x_dim = kwargs.get("x_dim", 784)
        self.x_height = kwargs.get("x_height", 28)
        self.x_weight = kwargs.get("x_weight", 28)
        self.x_depth = kwargs.get("x_depth", 1)

        self.y_dim = kwargs.get("y_dim", 5)
        self.z_dim = kwargs.get("z_dim", 100)

        self.batch_size = kwargs.get("batch_size", 100)
        self.lr = kwargs.get("lr", 0.0002)
        self.beta1 = kwargs.get("beta1", 0.5)  # recommend value in dcgan paper

        # configuration for the supervisor
        self.logdir = kwargs.get("logdir", "./logdir")
        self.sampledir = kwargs.get("sampledir", "./sample_image")

        self.max_steps = kwargs.get("max_steps", 100000)
        self.sample_every_n_steps = kwargs.get("sample_every_n_steps", 500)
        self.summary_every_n_steps = kwargs.get("summary_every_n_steps", 10)
        self.save_model_secs = kwargs.get("save_model_secs", 1200)
        self.checkpoint_basename = kwargs.get("checkpoint_basename", "./dcgan/"+str(self.lr))

        # configuration for the dataset queue
        self.min_after_dequeue = kwargs.get("min_after_dequeue", 5000)


def main():
    config = Config()
    trainer = t.Trainer(config)
    trainer.fit()

if __name__ == '__main__':
    main()