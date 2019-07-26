
import tensorflow as tf
import itertools


class AFMModelParams(object):
    """ class for initializing weights """
    def __init__(self, feature_size, embedding_size, field_size, attention_factor):
        self._feature_size = feature_size
        self._embedding_size = embedding_size
        self._field_size = field_size
        self._attention_factor = attention_factor

    def initialize_weights(self):
        """ init AFM  weights
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
        # attention network weights
        weights['attention_w'] = tf.get_variable(
            name='attention_w',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[self._embedding_size, self._attention_factor])
        weights['attention_b'] = tf.get_variable(
            name='attention_b',
            dtype=tf.float32,
            initializer=bias_initializer,
            shape=[self._attention_factor,])
        weights['projection_h'] = tf.get_variable(
            name='projection_h',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[self._attention_factor, 1])
        # last projection layer weights
        weights['projection_p'] = tf.get_variable(
            name='projection_p',
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[self._embedding_size, 1])
        return weights


class AFMModel(object):

    """AFM implementation"""

    @staticmethod
    def afm_model_fn(features, labels, mode, params):

        # parse params
        embedding_size = params['embedding_size']
        feature_size = params['feature_size']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        field_size = params['field_size']
        optimizer_used = params['optimizer']
        attention_factor = params['attention_factor']
        keep_prob = params['keep_prob']

        # parse features
        feature_idx = features["feature_idx"]
        feature_idx = tf.reshape(feature_idx, shape=[-1, field_size])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[-1, field_size, 1])

        # tf fm weights
        tf_model_params = AFMModelParams(feature_size, embedding_size, field_size, attention_factor)
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
        first_order = tf.reduce_sum(first_order, 1, keepdims=True)

        # AFM - attentional bi-interaction
        # feature embeddings
        f_e_m = tf.multiply(feature_values, embeddings)
        f_e_m_list = tf.split(f_e_m, field_size, axis=1)
        row = list()
        col = list()
        for r, c in itertools.combinations(f_e_m_list, 2):
            row.append(r)
            col.append(c)
        p = tf.concat(row, axis=1)
        q = tf.concat(col, axis=1)

        bi_interaction = tf.multiply(p, q)
        attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
            bi_interaction, weights['attention_w'], axes=(-1, 0)), weights['attention_b']))
        normalized_att_score = tf.nn.softmax(tf.tensordot(attention_temp, weights['projection_h'], axes=(-1, 0)))
        attention_output = tf.reduce_sum(tf.multiply(normalized_att_score, bi_interaction), axis=1)
        attention_output = tf.nn.dropout(attention_output, keep_prob, seed=2019)

        afm_out = tf.tensordot(attention_output, weights['projection_p'], axes=(-1, 0))

        # final objective function
        logits = afm_out + first_order + bias

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

