'''
Implementation for variational auto-encoder
'''

import os.path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST')
INPUT_DIM = 784
HIDDEN_ENCODER_DIM = 400
HIDDEN_DECODER_DIM = 400
LATENT_DIM = 20
LAM = 0


class VaeModel:
    # initialize parameter
    def __init__(self, input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lam):
        self.input_dim = input_dim
        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.latent_dim = latent_dim
        self.lam = lam

    # create placeholders for input data
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim])

    # create variables for weights (w, b)
    def _create_variables(self):
        with tf.name_scope("encoder"):
            self.W_encoder_input_hidden = tf.Variable(tf.truncated_normal([self.input_dim, self.hidden_encoder_dim], stddev=0.001))
            self.b_encoder_input_hidden = tf.Variable(tf.constant(0., shape=[self.hidden_encoder_dim]))

            self.W_encoder_hidden_mu = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], stddev=0.001))
            self.b_encoder_hidden_mu = tf.Variable(tf.constant(0., shape=[self.latent_dim]))

            self.W_encoder_hidden_logvar = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], stddev=0.001))
            self.b_encoder_hidden_logvar = tf.Variable(tf.constant(0., shape=[self.latent_dim]))

        with tf.name_scope("decoder"):
            self.W_decoder_z_hidden = tf.Variable(tf.truncated_normal([self.latent_dim, self.hidden_decoder_dim], stddev=0.001))
            self.b_decoder_z_hidden = tf.Variable(tf.constant(0., shape=[self.hidden_decoder_dim]))

            self.W_decoder_hidden_reconstruction = tf.Variable(tf.truncated_normal([self.hidden_decoder_dim, self.input_dim], stddev=0.001))
            self.b_decoder_hidden_reconstruction = tf.Variable(tf.constant(0., shape=[self.input_dim]))

    # define the graph model and create loss func
    def _create_loss(self):
        with tf.name_scope("loss"):
            l2_loss = tf.nn.l2_loss(self.W_encoder_input_hidden) + tf.nn.l2_loss(self.W_encoder_hidden_mu) + \
                      tf.nn.l2_loss(self.W_encoder_hidden_logvar) + tf.nn.l2_loss(self.W_decoder_z_hidden) + \
                      tf.nn.l2_loss(self.W_decoder_hidden_reconstruction)

            '''Encoder'''
            hidden_encoder = tf.nn.relu(tf.matmul(self.x, self.W_encoder_input_hidden) + self.b_encoder_input_hidden)

            # Mu encoder
            mu_encoder = tf.matmul(hidden_encoder, self.W_encoder_hidden_mu) + self.b_encoder_hidden_mu
            # Sigma encoder
            logvar_encoder = tf.matmul(hidden_encoder, self.W_encoder_hidden_logvar) + self.b_encoder_hidden_logvar
            # Sample epsilon
            epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

            '''Sample latent variable'''
            std_encoder = tf.exp(0.5 * logvar_encoder)
            z = mu_encoder + tf.multiply(std_encoder, epsilon)

            '''Decoder'''
            hidden_decoder = tf.nn.relu(tf.matmul(z, self.W_decoder_z_hidden) + self.b_decoder_z_hidden)
            x_hat = tf.matmul(hidden_decoder, self.W_decoder_hidden_reconstruction) + self.b_decoder_hidden_reconstruction

            '''Loss func'''
            KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder),
                                       reduction_indices=1)
            BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=self.x), reduction_indices=1)
            self.loss = tf.reduce_mean(BCE + KLD, name='loss')
            self.regularized_loss = self.loss + self.lam * l2_loss

    # create optimizer
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.regularized_loss)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


def train_model(model, batch_size, num_train_steps):
    saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring saved parameters")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initializing parameters")
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('experiment_graph', graph=sess.graph)
        for step in range(1, num_train_steps):
            batch = mnist.train.next_batch(batch_size)
            _, cur_loss, summary_op = sess.run([model.optimizer, model.loss, model.summary_op], feed_dict={model.x: batch[0]})
            summary_writer.add_summary(summary_op, global_step=step)

        if step % 100 == 0:
            save_path = saver.save(sess, "checkpoints/model.ckpt", global_step=step)
            print("Step {0} | Loss: {1}".format(step, cur_loss))



def main():
    model = VaeModel(INPUT_DIM, HIDDEN_ENCODER_DIM, HIDDEN_DECODER_DIM, LATENT_DIM, LAM)
    model.build_graph()
    train_model(model, 100, int(1e6))


if __name__ == '__main__':
    main()
