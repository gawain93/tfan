import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
import tensorflow_federated as tff
import collections
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

NUM_CLIENTS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])
    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]

# change to randomly sampled client!
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)

MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

def create_mnist_variables():
  return MnistVariables(
      weights = tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias = tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples = tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum = tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum = tf.Variable(0.0, name='accuracy_sum', trainable=False))

def mnist_forward_pass(variables, batch):
    y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32))

    num_examples = tf.cast(tf.size(batch['y']), tf.float32)

    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions

def get_local_mnist_metrics(variables):
    return collections.OrderedDict([
        ('num_examples', variables.num_examples),
        ('loss', variables.loss_sum / variables.num_examples),
        ('accuracy', variables.accuracy_sum / variables.num_examples)
    ])

@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
    return {
        'num_examples': tff.federated_sum(metrics.num_examples),
        'loss': tff.federated_mean(metrics.loss, metrics.num_examples),
        'accuracy': tff.federated_mean(metrics.accuracy, metrics.num_examples)
    }

class MnistModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                              tf.float32)),
                                        ('y', tf.TensorSpec([None, 1], tf.int32))])

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        num_exmaples = tf.shape(batch['x'])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_exmaples)

    @tf.function
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients

class MnistTrainableModel(MnistModel, tff.learning.TrainableModel):

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape() as tape:
            output = self.forward_pass(batch)
        grads = tape.gradient(output.loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.SGD(0.02)
        optimizer.apply_gradients(
            zip(tf.nest.flatten(grads), tf.nest.flatten(self.trainable_variables)))
        return output

iterative_process = tff.learning.build_federated_averaging_process(
    MnistTrainableModel)
state = iterative_process.initialize()
for round_num in range(2, 11):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

evaluation = tff.learning.build_federated_evaluation(MnistModel)
train_metrics = evaluation(state.model, federated_train_data)
federated_test_data = make_federated_data(emnist_test, sample_clients)
test_metrics = evaluation(state.model, federated_test_data)
print(str(test_metrics))
print(str(train_metrics))