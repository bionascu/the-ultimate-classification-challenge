import numpy as np
import tensorflow as tf


# Custom loss function
def negative_sampling(labels, logits, counts, ns_size):
    # Create sampling probability as a tensor
    base_prob = tf.constant((counts.values ** .75) / (counts.values ** .75).sum())
    base_prob = tf.ones((5270,))
    sampling_prob = tf.reshape(tf.tile(base_prob, tf.shape(labels)), tf.shape(logits))
    # sampling_prob = tf.Variable(tf.reshape(tf.tile(base_prob, tf.shape(labels)), tf.shape(logits)))

    # Set true label probabilities to 0 TODO comment line 9, uncomment line 10 and below
    # indices = tf.transpose(tf.stack([tf.to_int64(tf.range(tf.shape(logits)[0])), labels]))
    # updates = tf.zeros(tf.shape(labels), dtype=tf.float64)
    # tf.scatter_nd_update(ref=sampling_prob, indices=indices, updates=updates)

    # Create indexes based on sampling probability
    # TODO figure out why we need to convert to log probabilities
    ns_col_idxs = tf.multinomial(tf.log(sampling_prob), ns_size)

    ns_idxs = tf.concat([tf.reshape(tf.tile(tf.to_int64(tf.range(tf.shape(logits)[0])), tf.convert_to_tensor([ns_size])), (-1, 1)),
                         tf.reshape(tf.transpose(ns_col_idxs), (-1, 1))], axis=1)

    # Sample negative logits
    ns_logits = tf.reshape(tf.gather_nd(logits, ns_idxs), (-1, ns_size))
    # Sample true logits
    ps_logits = tf.gather_nd(logits, tf.transpose(tf.stack([tf.to_int64(tf.range(tf.shape(logits)[0])), labels])))
    # Concatenate
    sample_logits = tf.concat([tf.reshape(ps_logits, (-1, 1)), ns_logits], axis=1)
    # Sample true labels
    sample_labels = tf.reshape(tf.zeros(tf.shape(ps_logits), dtype=tf.int64), (-1,))

    # Compute cross-entropy
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sample_labels, logits=sample_logits, name="ns_cross_entropy"))

    return loss


# Custom loss function
def negative_sampling2(y_true, y_pred, counts, ns_size):
    # TODO try sampled_softmax (this is not working)
    # Set true category count to 0
    #counts.set_value(y_true.value_index, 0) #TODO
    # Compute sampling probability
    sampling_prob = np.power(counts.values, 0.75) / np.sum(
        np.power(counts.values, 0.75))
    num_classes = counts.size
    # compute cross-entropy
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf.reshape(tf.constant(sampling_prob), (-1, 1)),
                                                     biases=tf.zeros([num_classes]),
                                                     labels=y_true,
                                                     inputs=y_pred,
                                                     name="ns_loss",
                                                     num_sampled=ns_size,
                                                     num_classes=num_classes))
    return loss
