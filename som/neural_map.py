import numpy as np
import tensorflow as tf
import utils


class NeuralMap(object):

    def __init__(self, input_shape, n_rows, n_cols, name, state=None, weights=None, forward_prop=None):
        self._nrows = n_rows
        self._ncols = n_cols
        self._input_shape = input_shape
        self._name = name

        self._state = state if state is not None else tf.Variable(tf.zeros((self._nrows, self._ncols)),
                                                                  name=self._name + "_state")
        self._weights = weights if weights is not None else tf.Variable(tf.random_uniform(
            shape=[self._nrows * self._ncols, np.prod(self._input_shape)],
            maxval=1.0, seed=0), name=self._name + "_weights")

        if forward_prop is None:
            self._build_forward_prop_graph()
        else:
            self._forward_prop = forward_prop
        return

    @classmethod
    def from_meta_graph(cls, meta_path, ckpnt_dir):
        sess = tf.get_default_session()
        assert(sess is not None)
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckpnt_dir))
        graph = tf.get_default_graph()

        list_tensors = [(i.name, i.values()) for i in graph.get_operations()]
        tensors = dict(list_tensors)

        weights_key = [key for key in tensors.keys() if key.endswith('weights')]
        assert(len(weights_key)is not 0)
        weights_key = weights_key[0]

        map_name = weights_key[:-8]

        input_shape = graph.get_tensor_by_name(map_name + "_input:0").shape.as_list()
        n_rows = graph.get_tensor_by_name(map_name + "_state:0").shape[0]
        n_cols = graph.get_tensor_by_name(map_name + "_state:0").shape[1]
        state = graph.get_tensor_by_name(map_name + "_state:0")
        weights = graph.get_tensor_by_name(map_name + "_weights:0")
        forward_prop = graph.get_tensor_by_name(map_name + "_forward_prop:0")
        return cls(input_shape=input_shape, n_rows=n_rows, n_cols=n_cols, name=map_name, state=state, weights=weights,
                   forward_prop=forward_prop)

    def get_state(self):
        return self._state

    def set_state(self, state):
        if self._state is None:
            raise Exception("Neural map need to be initialized")
        if state.shape != self._state.shape:
            raise Exception("Dimensions of the neural map and state should be compatible {}; {}".format(
                state.shape, self._state.shape))
        self._state = state

    # def forward_prop(self, input):
    #     inp = tf.constant(input, dtype=tf.float32)
    #     input_flat = tf.reshape(inp, [tf.size(inp), 1])
    #     output_flat = tf.matmul(self._weights, input_flat)
    #     self._state = tf.reshape(output_flat, [self._nrows, self._ncols])
    #     return self._state

    def _build_forward_prop_graph(self):
        input = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name=self._name + "_input")
        input_flat = tf.reshape(input, [tf.size(input), 1], name=self._name + "_input_flat")
        output_flat = tf.matmul(self._weights, input_flat)
        new_state = tf.reshape(output_flat, [self._nrows, self._ncols])
        self._forward_prop = tf.assign(self._state, new_state, name=self._name+"_forward_prop")
        return self._forward_prop

    def get_forward_prop_graph(self):
        assert(self._forward_prop is not None)
        return self._forward_prop


class NeuralSOM(NeuralMap):

    def __init__(self, input_shape, n_rows, n_cols, name, state=None, weights=None,
                 forward_prop=None, bmu=None, update_weights=None):
        NeuralMap.__init__(self, input_shape, n_rows, n_cols, name, state, weights, forward_prop)
        self._bmu = None
        if bmu is None:
            self._build_bmu_graph()
        else:
            self._bmu_graph = bmu

        if update_weights is None:
            self._build_train_graph()
        else:
            self._update_weights = update_weights
        return

    @classmethod
    def from_meta_graph(cls, meta_path, ckpnt_dir):
        sess = tf.get_default_session()
        assert (sess is not None)
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckpnt_dir))
        graph = tf.get_default_graph()

        list_tensors = [(i.name, i.values()) for i in graph.get_operations()]
        tensors = dict(list_tensors)

        weights_key = [key for key in tensors.keys() if key.endswith('weights')]
        assert (len(weights_key) is not 0)
        weights_key = weights_key[0]

        map_name = weights_key[:-8]

        input_shape = graph.get_tensor_by_name(map_name + "_input:0").shape.as_list()
        n_rows = graph.get_tensor_by_name(map_name + "_state:0").shape[0]
        n_cols = graph.get_tensor_by_name(map_name + "_state:0").shape[1]
        state = graph.get_tensor_by_name(map_name + "_state:0")
        weights = graph.get_tensor_by_name(map_name + "_weights:0")
        forward_prop = graph.get_tensor_by_name(map_name + "_forward_prop:0")
        bmu = graph.get_tensor_by_name(map_name + "_bmu:0")
        update_weights = graph.get_tensor_by_name(map_name + "_update_weights:0")
        return cls(input_shape=input_shape, n_rows=n_rows, n_cols=n_cols, name=map_name, state=state, weights=weights,
                   forward_prop=forward_prop, bmu=bmu, update_weights=update_weights)

    def calc_distance_to_bmu_matrix(self, bmu):
        assert(bmu is not None)
        unit_coords = utils.matrix_indices_2d(self._nrows, self._ncols)

        coord_diff = tf.subtract(unit_coords, tf.to_float(bmu))
        dist = tf.norm(coord_diff, axis=-1)
        return dist

    def calc_neighbourhood_coeff(self, bmu, sigma):
        """
        returns neighbour coefficient for the map units with declines with the increase of the distance to the bmu
        :param bmu: 1-D list of indices of the best matching unit
        :param sigma: if distance exceeds sigma then coeff falls below 0.6, if exceeds 2*sigma then coefficient drops below 0.14
        :return: 2-D tensor with neighbour coefficients for each unit in the map
        """
        dist = self.calc_distance_to_bmu_matrix(bmu)
        h = tf.exp(tf.divide(tf.multiply(-0.5, dist**2), sigma**2))
        return h

    def _build_bmu_graph(self):
        f_prop = self.get_forward_prop_graph()
        self._bmu_graph = tf.identity(utils.argmax_2d(f_prop), name=self._name + "_bmu")
        return

    def get_bmu_graph(self):
        assert(self._bmu_graph is not None)
        return self._bmu_graph

    def get_bmu(self, input):
        sess = tf.get_default_session()
        assert (sess is not None)
        op = self.get_bmu_graph()
        output = sess.run(op, feed_dict={self._name + "_input:0": input})
        return output

    def get_bmus(self, input):
        sess = tf.get_default_session()
        assert (sess is not None)
        op = self.get_bmu_graph()
        bmus = []
        for i in range(input.shape[0]):
            input_sample = input[i]
            if input_sample.ndim is 1:
                input_sample = input_sample[:, np.newaxis]
            bmu = sess.run(op, feed_dict={self._name + "_input:0": input_sample})
            bmus.append(bmu)
        return np.array(bmus)

    def _build_train_graph(self):
        neighbour_sigma = tf.placeholder(dtype=tf.float32, name=self._name + "_sigma")
        learning_rate = tf.placeholder(dtype=tf.float32, name=self._name + "_lr")
        input_flat = tf.get_default_graph().get_tensor_by_name("{}_input_flat:0".format(self._name))
        bmu = self.get_bmu_graph()
        h = self.calc_neighbourhood_coeff(bmu, neighbour_sigma)
        delta = tf.multiply(tf.expand_dims(h, -1), tf.subtract(self._weights, tf.transpose(input_flat))) * learning_rate
        new_weights = tf.subtract(self._weights, delta)
        self._update_weights = tf.assign(self._weights, new_weights, name=self._name + "_update_weights")

    def get_train_graph(self):
        assert(self._update_weights is not None)
        return self._update_weights

    def train_one_sample(self, input, neighbour_sigma, learning_rate):
        sess = tf.get_default_session()
        assert(sess is not None)
        op = self.get_train_graph()
        new_weights = sess.run(op, feed_dict={self._name + "_input:0": input,
                                                     self._name + "_sigma:0": neighbour_sigma,
                                                     self._name + "_lr:0": learning_rate})
        return new_weights

    def train(self, input, num_epochs=10,
              neighbour_sigma_start=None, neighbour_sigma_end=None, neighbourhood_decay="geometrical",
              learning_rate_start=0.1, learning_rate_end=0.01, learning_rate_decay="linear",
              verbose=0):
        sess = tf.get_default_session()
        assert(sess is not None)
        new_weights = None
        op = self.get_train_graph()
        if neighbour_sigma_start is None:
            neighbour_sigma_start = min(self._input_shape)/2
        if neighbour_sigma_end is None:
            neighbour_sigma_end = 1

        if learning_rate_decay == "logarithmic":
            learning_rates = np.logspace(learning_rate_start, learning_rate_end, num=num_epochs*input.shape[0])
        elif learning_rate_decay == "linear":
            learning_rates = np.linspace(learning_rate_start, learning_rate_end, num=num_epochs * input.shape[0])
        elif learning_rate_decay == "geometrical":
            learning_rates = np.geomspace(learning_rate_start, learning_rate_end, num=num_epochs * input.shape[0])
        assert(learning_rates is not None)

        if neighbourhood_decay == "logarithmic":
            neighbour_sigmas = np.logspace(neighbour_sigma_start, neighbour_sigma_end, num=num_epochs*input.shape[0])
        elif neighbourhood_decay == "linear":
            neighbour_sigmas = np.linspace(neighbour_sigma_start, neighbour_sigma_end, num=num_epochs*input.shape[0])
        elif neighbourhood_decay == "geometrical":
            neighbour_sigmas = np.geomspace(neighbour_sigma_start, neighbour_sigma_end, num=num_epochs*input.shape[0])
        assert(neighbour_sigmas is not None)


        for i in range(num_epochs):
            if verbose > 0:
                print("\nepoch: {} out of {}".format(i+1, num_epochs))
            for j in range(input.shape[0]):
                if verbose > 1 and j % 100 == 0:
                    print("\r...processing {} out of {}".format(j, input.shape[0]), end="")
                input_sample = input[j]
                if input_sample.ndim is 1:
                    input_sample = input_sample[:, np.newaxis]
                new_weights = sess.run(op, feed_dict={self._name + "_input:0": input_sample,
                                                      self._name + "_sigma:0": neighbour_sigmas[i*input.shape[0] + j],
                                                      self._name + "_lr:0": learning_rates[i*input.shape[0] + j]})
        if verbose > 0:
            print("\nlearning finished...")
        return new_weights


def test_1():
    np.set_printoptions(precision=3)
    input_shape = (3, 2)
    som = NeuralSOM(input_shape, 5, 4, "test_som")
    np.random.seed(0)
    input = np.random.rand(*input_shape)
    print(input)
    print(input.shape)
    print(som._input_shape)
    # out = som.train_one_sample(input, 25, 0.01)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("/logs/...", sess.graph)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        weights = sess.run(som._weights)
        print(weights)
        for i in range(200):
            print(i)
            new_weights = som.train_one_sample(input, 2, 0.1)
        weights1 = sess.run(som._weights)
        print(weights1)
    print("delta: ", np.array(weights1 - weights))
    print("diff with input: ", np.array(input.flatten() - weights1))
    return


def test_2():
    np.set_printoptions(precision=3)
    input_shape = (3, 2)
    som_name = "test_som"

    np.random.seed(0)
    num_samples = 100
    input = np.random.rand(num_samples, *input_shape)
    print("input shape:", input.shape)

    with tf.Session() as sess:
        som = NeuralSOM(input_shape, 5, 4, som_name)
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        som.train(input, num_epochs=2, verbose=True)
        print(sess.run(som._state))
        print(sess.run(som._weights))
        saver.save(sess, "./logs/my_test_model")


def test_get_bmu():
    np.set_printoptions(precision=3)
    input_shape = (3, 2)
    som_name = "test_som"

    np.random.seed(0)
    num_samples = 100
    input = np.random.rand(num_samples, *input_shape)
    print("input shape:", input.shape)

    with tf.Session() as sess:
        path = './logs/my_test_model.meta'

        map = NeuralSOM.from_meta_graph(path)
        # saver = tf.train.import_meta_graph('./logs/my_test_model.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        # graph = tf.get_default_graph()


        som = NeuralSOM(input_shape, 5, 4, som_name)
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        som.train(input, verbose=True)
        writer = tf.summary.FileWriter("./logs/", sess.graph)

        writer.flush()
        # saver.save(sess, '')


def test_load_from_graph():
    np.set_printoptions(precision=3)
    input_shape = (3, 2)

    np.random.seed(0)
    num_samples = 100
    input = np.random.rand(num_samples, *input_shape)
    print("input shape:", input.shape)

    with tf.Session() as sess:
        meta_path = './logs/my_test_model.meta'
        ckpt_dir = './logs/'
        map = NeuralSOM.from_meta_graph(meta_path, ckpt_dir)
        # map.train(input, verbose=True)
        print(map.get_bmus(input))


# test_2()
# test_load_from_graph()