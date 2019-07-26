
import tensorflow as tf


class DeepFMModelParams(object):
    """ class for initializing weights """
    def __init__(self, feature_size, embedding_size, field_size, hidden_units):
        self._feature_size = feature_size
        self._embedding_size = embedding_size
        self._field_size = field_size
        self._hidden_units = hidden_units

    def initialize_weights(self):
        """ init DeepFM  weights
        Returns
        weights:
        feature_embeddings:  vi, vj second order params
        weights_first_order: wi first order params
        bias: b  bias
        hidden_layer{i}: fnn hidden layer params
        """
        # embedding & fm weights
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        weights["feature_embeddings"] = tf.get_variable(
            name='weights',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[self._feature_size, self._embedding_size])
        weights["weights_first_order"] = tf.get_variable(
            name='vectors',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[self._feature_size, 1])
        weights["fm_bias"] = tf.get_variable(
            name='bias',
            dtype=tf.float32,
            initializer=bias_initializer,
            shape=[1])
        # fnn weights
        for i, v in enumerate(self._hidden_units):
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
                    shape=[self._hidden_units[i-1], v])
        weights['fnn_output_layer'] = tf.get_variable(
            name='fnn_output_layer',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[self._hidden_units[-1], 1])
        weights['combination_weights'] = tf.get_variable(
            name='combination_weights',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[2, 1]
        )
        return weights


class DeepFMModel(object):

    """DeepFM implementation"""

    @staticmethod
    def deepfm_model_fn(features, labels, mode, params):

        # parse params
        embedding_size = params['embedding_size']
        feature_size = params['feature_size']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        field_size = params['field_size']
        optimizer_used = params['optimizer']
        hidden_units = params['hidden_units']

        # parse features
        feature_idx = features["feature_idx"]
        feature_idx = tf.reshape(feature_idx, shape=[-1, field_size])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[-1, field_size, 1])

        # tf fm weights
        tf_model_params = DeepFMModelParams(feature_size, embedding_size, field_size, hidden_units)
        weights = tf_model_params.initialize_weights()
        embeddings = tf.nn.embedding_lookup(
            weights["feature_embeddings"],
            feature_idx
        )
        weights_first_order = tf.nn.embedding_lookup(
            weights["weights_first_order"],
            feature_idx
        )
        bias = weights['fm_bias']

        # build function
        # fm first order
        first_order = tf.multiply(feature_values, weights_first_order)
        first_order = tf.reduce_sum(first_order, 2)
        first_order = tf.reduce_sum(first_order, 1, keep_dims=True)

        # fm second order
        # feature * embeddings
        f_e_m = tf.multiply(feature_values, embeddings)
        # square(sum(feature * embedding))
        f_e_m_sum = tf.reduce_sum(f_e_m, 1)
        f_e_m_sum_square = tf.square(f_e_m_sum)
        # sum(square(feature * embedding))
        f_e_m_square = tf.square(f_e_m)
        f_e_m_square_sum = tf.reduce_sum(f_e_m_square, 1)
        second_order = f_e_m_sum_square - f_e_m_square_sum
        second_order = tf.reduce_sum(second_order, 1, keep_dims=True)
        #
        # fm final objective function
        fm_logits = second_order + first_order + bias

        # fnn hidden layers
        fnn = tf.keras.layers.Flatten()(f_e_m)
        for i in range(len(hidden_units)):
            fnn = tf.nn.relu(tf.matmul(fnn, weights['fnn_hidden_layer{}'.format(i)]))
        fnn_logits = tf.nn.relu(tf.matmul(fnn, weights['fnn_output_layer']))
        logits = tf.matmul(tf.concat([fm_logits, fnn_logits], axis=1), weights['combination_weights'])
        # logits = fm_logits + fnn_logits

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

