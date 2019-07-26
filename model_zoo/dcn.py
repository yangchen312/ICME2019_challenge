
import tensorflow as tf


class DCNParams(object):
    """ class for initializing weights """
    def __init__(self, feature_size, embedding_size, field_size, num_cross_layer, deep_units):
        self._feature_size = feature_size
        self._embedding_size = embedding_size
        self._field_size = field_size
        self._num_cross_layer = num_cross_layer
        self._deep_units = deep_units

    def initialize_weights(self):
        """ initialize DCN weights
        Returns
        weights:
        feature_embeddings:  vi, vj second order params
        weights_first_order: wi first order params
        bias: b  bias
        hidden_layer{i}: fnn hidden layer params
        """
        # embedding weights
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        weights["feature_embedding"] = tf.get_variable(
            name='weights',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[self._feature_size, self._embedding_size])
        # cross net weights
        for i in range(self._num_cross_layer):
            weights['w_cross_layer{}'.format(i)] = tf.get_variable(
                name='cross_layer{}_weights'.format(i),
                dtype=tf.float32,
                initializer=weights_initializer,
                shape=[self._field_size * self._embedding_size, 1])
            weights['b_cross_layer{}'.format(i)] = tf.get_variable(
                name='cross_layer{}_bias'.format(i),
                dtype=tf.float32,
                initializer=bias_initializer,
                shape=[self._field_size * self._embedding_size, 1])
        # fnn weights
        for i, v in enumerate(self._deep_units):
            if i == 0:
                weights['fnn_hidden_layer{}'.format(i)] = tf.get_variable(
                    name='fnn_hidden_layer{}'.format(i),
                    dtype=tf.float32,
                    initializer=weights_initializer,
                    shape=[self._field_size * self._embedding_size, v])
            else:
                weights['fnn_hidden_layer{}'.format(i)] = tf.get_variable(
                    name='fnn_hidden_layer{}'.format(i),
                    dtype=tf.float32,
                    initializer=weights_initializer,
                    shape=[self._deep_units[i-1], v])
        weights['combination_weights'] = tf.get_variable(
            name='combination_weights',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[2, 1]
        )
        return weights


class DCN(object):

    """DCN implementation"""

    @staticmethod
    def deepfm_model_fn(features, labels, mode, params):

        # parse params
        embedding_size = params['embedding_size']
        feature_size = params['feature_size']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        field_size = params['field_size']
        optimizer_used = params['optimizer']
        num_cross_layer = params['num_cross_layer']
        deep_units = params['deep_units']

        # parse features
        feature_idx = features["feature_idx"]
        feature_idx = tf.reshape(feature_idx, shape=[-1, field_size])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[-1, field_size, 1])

        # tf fm weights
        tf_model_params = DCNParams(feature_size, embedding_size, field_size, num_cross_layer, deep_units)
        weights = tf_model_params.initialize_weights()
        embeddings = tf.nn.embedding_lookup(
            weights["feature_embeddings"],
            feature_idx
        )

        # build function
        f_e_m = tf.multiply(feature_values, embeddings)
        inputs = tf.keras.layers.Flatten()(f_e_m)
        # cross layers
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(num_cross_layer):
            xl_w = tf.tensordot(x_l, weights['w_cross_layer{}'.format(i)], axes=(1, 0))
            dot_ = tf.matmul(x_0, xl_w)
            x_l = dot_ + weights['b_cross_layer{}'.format(i)] + x_l
        cross_output = tf.squeeze(x_l, axis=2)
        # fnn
        fnn = inputs
        for i in range(len(deep_units)):
            fnn = tf.nn.relu(tf.matmul(fnn, weights['fnn_hidden_layer{}'.format(i)]))
        # Combination
        logits = tf.matmul(tf.concat([cross_output, fnn], axis=1), weights['combination_weights'])

        # final objective function
        predicts = tf.sigmoid(logits)
        predictions = {"prob": predicts}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            labels = tf.reshape(labels, shape=[batch_size, 1])
            # loss function
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            sigmoid_loss = tf.reduce_mean(sigmoid_loss)
            loss = sigmoid_loss

            # train op
            if optimizer_used == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
            elif optimizer_used == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            else:
                raise Exception("unknown optimizer", optimizer_used)

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            # metric
            eval_metric_ops = {"auc": tf.metrics.auc(labels, predicts)}

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
            elif mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
            else:
                raise NotImplementedError('Unknown mode {}'.format(mode))


