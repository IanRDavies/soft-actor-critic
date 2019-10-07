import tensorflow as tf
import gym


def mlp(x, hidden_sizes, output_dim, activation=tf.nn.relu, output_activation=None):
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=output_dim, activation=output_activation)


def clip(x, lower, upper):
    adj = tf.stop_gradient(
        (upper - x)*tf.cast(x > upper, tf.float32)
        + (lower - x)*tf.cast(x < lower, tf.float32)
    )
    clipped = x + adj
    return clipped


def polyak_update(variables, target_variables, polyak):
    operations = []
    for variable, target_variable in zip(sorted(variables, key=lambda v: v.name), sorted(target_variables, key=lambda v: v.name)):
        operations.append(target_variable.assign(
            polyak * target_variable + (1. - polyak) * variable))
    op = tf.group(*operations)
    return op


def get_scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.

    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


def initialise_environment(name='LunarLanderContinuous-v2'):
    env = gym.make(name)
    return env
