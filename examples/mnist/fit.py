from functools import partial

import gin
import haiku as hk
import jax
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from huf import callbacks, metrics, models, module_ops, ops


@gin.configurable
def net_fun(inputs, is_training: bool, *, dropout_rate=0.5):
    assert isinstance(is_training, bool)
    x = hk.Conv2D(16, 3, 2)(inputs)
    del inputs
    x = hk.BatchNorm(True, True, 0.99)(x, is_training=is_training)
    x = jax.nn.relu(x)
    x = hk.Conv2D(64, 3, 2)(x)
    x = hk.BatchNorm(True, True, 0.99)(x, is_training=is_training)
    x = jax.nn.relu(x)
    x = x.reshape(x.shape[0], -1)
    x = module_ops.dropout(x, dropout_rate, is_training)
    x = hk.Linear(10)(x)
    return x


@gin.configurable
def get_dataset(split: str, batch_size: int):
    def map_fun(image, labels):
        return tf.cast(image, tf.float32) / 255, labels

    dataset = tfds.load("mnist", split=split, as_supervised=True)
    return dataset.batch(batch_size, True).map(map_fun)


if __name__ == "__main__":
    loss = ops.weighted_mean_fun(
        partial(ops.sparse_categorical_crossentropy, from_logits=True)
    )
    metrics = dict(acc=metrics.SparseCategoricalAccuracy)
    optimizer = optax.adamw(1e-3)
    model = models.Model(net_fun, loss, optimizer, metrics)

    batch_size = 64
    train_data, validation_data = tfds.load(
        "mnist", split=("train", "test"), as_supervised=True
    )
    train_data = get_dataset("train", batch_size)
    validation_data = get_dataset("test", batch_size)

    model.fit(
        jax.random.PRNGKey(0),
        train_data,
        epochs=10,
        validation_data=validation_data,
        callbacks=[callbacks.ProgbarLogger()],
    )

    # # The below demonstrates how training can be split across two runs
    # res = model.fit(
    #     jax.random.PRNGKey(0),
    #     train_data,
    #     epochs=5,
    #     validation_data=validation_data,
    #     callbacks=[callbacks.ProgbarLogger()],
    # )
    # model.fit(
    #     res.rng,
    #     train_data,
    #     epochs=10,
    #     initial_epoch=res.epochs,
    #     initial_state=res.model_state,
    #     validation_data=validation_data,
    #     callbacks=[callbacks.ProgbarLogger()],
    # )
