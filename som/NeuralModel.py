class NeuralModel(object):

    def __init__(self):
        self._layers = {}
        self._num_layers = 0
        return

    def add_layer(self, neural_map):
        nm_name = neural_map._name
        if nm_name is None:
            nm_name = "NeuralMap{}".format(self._num_layers)
            neural_map.name = nm_name

        self._layers[nm_name] = neural_map
        self._num_layers += 1

