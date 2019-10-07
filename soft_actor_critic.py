import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from replay_buffer import ReplayBuffer
import utils


# Define constants.
# Their purpose is explained when they are called upon.
LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6


class SAC(object):
    '''
    Soft Actor-Critic implementation following the paper referenced below.

    Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
    Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
    with a Stochastic Actor," Deep Learning Symposium, NIPS 2017.

    Soft Actor-Critic takes a maximum entropy approach to the common
    actor-critic reinforcement learning framework.

    We implement soft Actor-Critic for continuous action spaces.

    Our implementation has a separate value function and uses a clipped
    double Q approach to stabilise training.

    Arguments
    env - gym.Env instance
        The environment in which the agent is acting (from Open AI Gym)
    gamma - float (default: 0.999)
        The discount factor applied to future rewards
    alpha - float (default: 0.2)
        The weighting of the entropy term in the policy objective
    policy_lr - float (default: 1e-3)
        The learning rate used in an Adam optimiser for policy training
    critic_lr - float (default: 1e-3):
        The learning rate used in an Adam optimiser for Q and value
        function training
    polyak - float (default: 0.995)
        The constant used in polyak averaging:
        target_var[k+1] = polyak * target_var[k] + (1-polyak) * var[k+1]
    policy_net_shape - tuple of int (default: (400, 300))
        The number of units in the hidden layers of the policy network
    q_net_shape - tuple of int (default: (400, 300))
        The number of units in the hidden layers of Q and value networks
    batch_size - int (default: 1024)
        The number of time steps of experience used in each update iteration
    rb_capacity - int (default: 1e6)
        The capacity of the replay buffer
    logdir - str (default: None)
        The directory in which to place tensorboard log files.
        If None provided logging will not work.
    '''

    def __init__(
        self,
        env,
        gamma=0.999,
        alpha=0.2,
        policy_lr=1e-3,
        critic_lr=1e-3,
        polyak=0.995,
        policy_net_shape=(400, 300),
        q_net_shape=(400, 300),
        value_net_shape=(400, 300),
        batch_size=1024,
        rb_capacity=1000000,
        logdir=None
    ):
        # Attain the necessary dimensions etc from the environment
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # We assume that action spaces are symmetric about zero.
        self.action_scale = env.action_space.high[0]
        self.env = env
        # Save the network shapes
        self.policy_net_shape = policy_net_shape
        self.q_net_shape = q_net_shape
        self.value_net_shape = value_net_shape

        # Store hyperparameters in attributes
        self.gamma = gamma
        self.alpha = alpha
        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
        self.batch_size = batch_size

        # Build the tensorflow computation graph for SAC
        self._build()

        # Initialise a replay buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, rb_capacity)

        # Set up tensorboard infrastructure
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        if logdir is not None:
            self.summary_writer = tf.summary.FileWriter(logdir, flush_secs=30, max_queue=2)

        # Initialise variables
        self.sess.run(tf.global_variables_initializer())
        # Ensure that value network and target value network have the same
        # initialisation
        self._align_value_net_initialisations()

    def _build(self):
        '''
        Build the full computation graph for soft actor-critic
        '''
        # We split out the building piece by piece to avoid an overly
        # long method. However we place the methods in a sensible
        # order below to aid readability.

        # Instantiate placeholders
        self._build_placeholders()
        # Build the core of the model. The policy and the Q and Value
        # functions.
        self._build_actor_critic()
        # Finally, set up the losses and then the optimisation
        # operations.
        self._build_losses()
        self._build_optimisation()
        # Build the logging operations (tensorflow summaries)
        self._build_summaries()

    def _build_placeholders(self):
        '''Initialise tensorboard placeholders for SAC'''
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'action_ph')
        self.next_obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'next_obs_ph')
        self.reward_ph = tf.placeholder(tf.float32, (None,), 'reward_ph')
        self.terminal_ph = tf.placeholder(tf.float32, (None,), 'terminal_ph')

    def _build_actor_critic(self):
        '''Build both the actor (policy) and critic for soft actor-critic'''
        with tf.variable_scope('actor'):
            self.mu, self.pi, self.logp = self._make_policy(
                self.obs_ph, self.policy_net_shape
            )

        # Main outputs from computation graph
        with tf.variable_scope('critic'):
            # Set up two separate Q networks to allow for clipping.
            # This requires careful use of the reuse argument of
            # tensorflow variable scopes.
            # The value function has a separate method call.
            self.q1 = self.q_func(self.obs_ph, self.act_ph, self.q_net_shape, scope='q1')
            self.q2 = self.q_func(self.obs_ph, self.act_ph, self.q_net_shape, scope='q2')
            self.q1_pi = self.q_func(self.obs_ph, self.pi, self.q_net_shape, scope='q1', reuse=True)
            self.q2_pi = self.q_func(self.obs_ph, self.pi, self.q_net_shape, scope='q2', reuse=True)
            self.v = self.value_func(self.obs_ph, self.value_net_shape, scope='v')
        # A different variable scope is used for the target value function
        # to ensure a separate set of parameters are maintained.
        with tf.variable_scope('target'):
            self.v_next = self.value_func(self.next_obs_ph, self.value_net_shape, scope='v')

    def _make_policy(self, obs, hidden_sizes, activation_fn=tf.nn.relu):
        '''
        Build a policy for soft actor critic

        Arguments
        obs - tf.Tensor or tf.Placeholder
            The observations in a tensor of size (batch_size, obs_dim)
        hidden_sizes - tuple of int
            The number of units in the hidden layers of the the policy network
        activation_fn - function
            The activation function to apply to all but the finallayer of the
            policy networks

        Returns
        mu - tf.Tensor
            The mean of the squashed Gaussian policy
        act - tf.Tensor
            Sampled actions for the observations provided
        logp - tf.Tensor
            The log probability of the actions under the current
            policy
        '''
        # Run the feed forward network to process the observation
        # to attain both a mean and a log standard deviation.
        mu = utils.mlp(obs, hidden_sizes, self.act_dim, activation_fn)

        # The mean of the distribution is the output of a final dense layer
        # mu = tf.layers.dense(h, units=self.act_dim, activation=None)

        # The log standard deviation is the output of a different network
        # this has tanh activation to keep log_sd within limits.
        log_sd = utils.mlp(obs, hidden_sizes, self.act_dim, activation_fn, tf.tanh)

        # Using Open AI style processing to rescale log_sd
        # This allows gradients to propagate through the clipping and avoids
        # numerical instability that can result in the initial stages of training
        # when neural network weights are small and random.
        log_sd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sd + 1)

        # Use tensorflow probability to handle the likelihood
        # calculations and the random sampling of actions.
        # A neater solution to explicitly applying the
        # reparameterisation trick.
        pi = tfp.distributions.MultivariateNormalDiag(
            loc=mu,
            scale_diag=tf.exp(log_sd),
            validate_args=True,
            allow_nan_stats=False
        )
        act = pi.sample()
        logp = pi.log_prob(act)

        # Ensure that the actions are in the bounds of the action
        # space. We assume that the action space is symmetric
        # about zero and the same size in all dimensions.
        # An assumption that OpenAI themselves make.
        # First account for the tanh function
        mu, act, logp = self._apply_tanh_squashing(mu, act, logp)
        # Then scale to the action space
        action_scale = float(self.action_scale)
        mu *= action_scale
        act *= action_scale
        return mu, act, logp

    @staticmethod
    def _apply_tanh_squashing(act, mu, logp):
        '''
        Squash policy output for bounded action spaces

        Arguments
        act - tf.Tensor
            Actions sampled from the Gaussian policy
        mu - tf.Tensor
            The mean of the Gaussian policy
        logp - tf.Tensor
            The log likelihood of actions under the Gaussian
            policy

        Returns
        Inputs adjusted for the tanh squashing function required by
        having a bounded action space.

        mu - tf.Tensor
        act - tf.Tensor
        log_prob - tf.Tensor
        '''
        # Use tanh to force mean and actions to be in [-1, 1]
        mu = tf.tanh(mu)
        act = tf.tanh(act)
        # Apply the likelihood adjustment as laid out in Appendix C
        # of the soft actor-critic paper (Haarnoja et al., 2017)
        # Add epsilon to avoid log(0)
        log_prob = logp - tf.reduce_sum(tf.log(
            utils.clip(1-tf.square(act), 0, 1)+EPSILON), axis=1)

        # Return the mean (for deterministic play), sampled actions and
        # the adjusted log likelihood
        return mu, act, log_prob

    def _build_losses(self):
        '''Build the loss calculations for actor and critic'''
        # Build target values based on Bellman equations
        # Note that the value target includes an entropy term
        # Entropy is included through the fact that we sample
        # actions according to the policy which therefore means
        # that minibatching provides a monte carlo approximation
        # to the integral needed for entropy calculation.

        # First perform the double-Q clipping then calculate
        # target values.
        q_min = tf.minimum(self.q1_pi, self.q2_pi)
        q_target = tf.stop_gradient(self.reward_ph +
                                    self.gamma * (1 - self.terminal_ph) * self.v_next)
        v_target = tf.stop_gradient(q_min - self.alpha * self.logp)

        # The policy is trained to maximise entropy and the Q value from
        # the first Q network
        self.policy_loss = tf.reduce_mean(self.alpha * self.logp - self.q1_pi)
        # The Q networks are trained on the mean squared Bellman error
        self.q1_loss = 0.5 * tf.reduce_mean(tf.squared_difference(q_target, self.q1))
        self.q2_loss = 0.5 * tf.reduce_mean(tf.squared_difference(q_target, self.q2))
        # The value network is trained on t
        self.v_loss = 0.5 * tf.reduce_mean(tf.squared_difference(v_target, self.v))
        self.critic_loss = self.q1_loss + self.q2_loss + self.v_loss

        # Collect Losses in one list
        self.losses = [
            self.policy_loss,
            self.q1_loss,
            self.q2_loss,
            self.v_loss,
            self.critic_loss
        ]

    def _build_optimisation(self):
        '''Builds graph for parameter updates'''
        # Collect parameters
        policy_params = utils.get_scope_vars('actor')
        q1_params = utils.get_scope_vars('critic/q1')
        q2_params = utils.get_scope_vars('critic/q2')
        # The parameters of the value functions are stored as
        # attributed to enable them to be set as equal after
        # initialisation
        self.v_params = utils.get_scope_vars('critic/v')
        self.target_v_params = utils.get_scope_vars('target/v')

        # Build separate optimisers to allow for separate learning
        # rates if required.
        self.policy_optimiser = tf.train.AdamOptimizer(self.policy_lr)
        self.critic_optimiser = tf.train.AdamOptimizer(self.critic_lr)

        # First update the critic. Note that we can update
        # both Q functions and the value function at the
        # same time. This follows from the different functions
        # forming separate computation subgraphs.
        self.critic_train_op = self.critic_optimiser.minimize(
            self.critic_loss, var_list=self.v_params + q1_params + q2_params)

        # Once the critic has been updated, update the policy
        # parameters. This is not strictly required but aids in
        # consistency and avoiding inconsistency via a race
        # condition
        with tf.control_dependencies([self.critic_train_op]):
            self.policy_train_op = self.policy_optimiser.minimize(
                self.policy_loss, var_list=policy_params
            )

        # Once the actor and critic have been updated, update
        # the target value function using polyak averaging.
        # This is done last as the main value function must be
        # updated first. Furthermore, this ensures that the
        # Q target values will be calculated correctly.
        with tf.control_dependencies([self.policy_train_op]):
            self.update_target_v_params = utils.polyak_update(
                self.v_params, self.target_v_params, self.polyak
            )

        # Collect all training operations in one place
        self.training_ops = [
            self.policy_train_op,
            self.critic_train_op,
            self.update_target_v_params
        ]

    def _build_summaries(self):
        '''Build tensorboard summaries'''
        # Set up scalar plots for all losses
        summaries = [
            tf.summary.scalar('policy_loss', self.policy_loss),
            tf.summary.scalar('q1_loss', self.q1_loss),
            tf.summary.scalar('q2_loss', self.q2_loss),
            tf.summary.scalar('v_loss', self.v_loss),
            tf.summary.scalar('critic_loss', self.critic_loss)
        ]
        # Merge these loss functions into one summary
        self.summary_op = tf.summary.merge(summaries)

        # Reward is from the environment and passed in externally
        # We handle these logs separately and simply set up a
        # method to pass values into placeholders which are then
        # passed into the summaries.
        self.avg_reward_summary_ph = tf.placeholder(tf.float32, (), 'reward_for_logging')
        self.max_reward_summary_ph = tf.placeholder(tf.float32, (), 'max_reward_for_logging')
        self.avg_reward_summary = tf.summary.scalar('Episode Reward (Average)', self.avg_reward_summary_ph)
        self.max_reward_summary = tf.summary.scalar('Max Episode Reward - Per Period', self.max_reward_summary_ph)
        # Merge the reward summaries
        self.reward_summary = tf.summary.merge([self.avg_reward_summary, self.max_reward_summary])

    def q_func(self, obs, actions, hidden_layer_sizes, activation=tf.nn.relu,
               output_activation=None, scope=None, reuse=False):
        '''
        A Q function mapping observations and actions to values.

        Arguments
        obs - tf.Tensor
            The agent's observations of the environment
        actions - tf.Tensor
            The actions the agent took
        hidden_layer_sizes - tuple of int
            The number of units in each hidden layer of the
            Q network (excluding the output layer which
            produces a scalar)
        activation - function (default: tf.nn.relu)
            The non-linearity applied at each layer of
            the Q network
        output_activation - function (default: None)
            The activation applied to the output of the
            Q network
        scope - string (default: None)
            The tensorflow scope to define variables in
        reuse - boolean (default: False)
            Whether or not to reuse variables in the scope

        Returns
        q - tf.Tensor
            The q values associated with the observations and
            actions provided
        '''
        # Set up the variable scope
        with tf.variable_scope(scope, reuse=reuse):
            # Use the multi-layered perceptron from the utils
            # file to calculate a Q value
            q = tf.squeeze(utils.mlp(tf.concat([obs, actions], axis=-1),
                                     hidden_layer_sizes, 1, activation, output_activation))
        return q

    def value_func(self, obs, hidden_layer_sizes, activation=tf.nn.relu,
                   output_activation=None, scope=None, reuse=False):
        '''
        A value function mapping observations to (state) values.

        Arguments
        obs - tf.Tensor
            The agent's observations of the environment
        hidden_layer_sizes - tuple of int
            The number of units in each hidden layer of the
            value network (excluding the output layer which
            produces a scalar)
        activation - function (default: tf.nn.relu)
            The non-linearity applied at each layer of
            the value network
        output_activation - function (default: None)
            The activation applied to the output of the
            value network
        scope - string (default: None)
            The tensorflow scope to define variables in
        reuse - boolean (default: False)
            Whether or not to reuse variables in the scope

        Returns
        v - tf.Tensor
            The values associated with the observations
            provided
        '''
        # Set up the variable scope
        with tf.variable_scope(scope, reuse=reuse):
            # Calculate the value using the multi-layered
            # perceptron function defined in the utils file.
            v = tf.squeeze(utils.mlp(obs, hidden_layer_sizes, 1, activation, output_activation))
        return v

    def action(self, obs, deterministic=False):
        '''
        Attain an action for a (batch of) observations.

        Arguments
        obs - np.array
            The observation(s) for which agent actions are desired
        deterministic - bool (default: False)
            Boolean determining whether actions are sampled or deterministic

        Returns
        np.array
            Action(s) for the observation(s) provided.
        '''
        # Run the tensorflow graph for the mean of the policy if acting
        # deterministically or for sampled actions otherwise
        # np.squeeze is applied to remove unnecessary dimensions when passing
        # in a single observation as id the case in play.
        return np.squeeze(self.sess.run(self.mu if deterministic else self.pi, {self.obs_ph: obs[None]}))

    def experience(self, obs, action, reward, next_obs, terminal):
        '''
        Process a time step of experience and add it to the replay buffer.

        Arguments
        obs - np.array
            Observation from the environment
        action - np.array
            Action taken
        reward - float
            Reward received
        next_obs - np.array
            Observation received after action executed
        terminal - bool or float
            End of episode indicator
        '''
        # Simply add experience to the replay buffer
        self.replay_buffer.add(obs, action, reward, next_obs, terminal)

    def train(self):
        '''
        Run one set of parameter updates for soft actor critic
        '''
        # Collect a minibatch of data for the update
        obs, act, rew, next_obs, terminal = self.replay_buffer.sample(self.batch_size)
        # Build a placeholder: value dictionary as usual for tensorflow
        feed_dict = {
            self.obs_ph: obs,
            self.act_ph: act,
            self.reward_ph: rew,
            self.next_obs_ph: next_obs,
            self.terminal_ph: terminal
        }
        # Run all training operations which will update paramaters
        # for the policy, q functions, value function and target value function
        self.sess.run(self.training_ops, feed_dict)

    def run_summary(self, step, episode_reward=None, max_ep_reward=None):
        '''
        Run tensorboard logging of losses and agent performance metrics

        Arguments
        step - int
            The time step or episode count at the point of logging
        episode_reward - float (optional, default: None)
            The reward from the current episode or average over several
            episodes. If not supplied reward logging is not performed
        max_ep_reward - float (optional, default: None)
            The maximum of the rewards over (recent) episodes.
            If not supplied then reward logging is not performed.
        '''
        # Ensure that the a summary writer is initialised.
        # If no log directory passed to initialisation there will be no
        # summary writer and hence we cannot do logging.
        assert hasattr(self, 'summary_writer'), 'Logging not set up, no log path provided at initialisation'
        # Sample a batch of experience and run summary operations
        obs, act, rew, next_obs, terminal = self.replay_buffer.sample(self.batch_size)
        feed_dict = {
            self.obs_ph: obs,
            self.act_ph: act,
            self.reward_ph: rew,
            self.next_obs_ph: next_obs,
            self.terminal_ph: terminal
        }
        s = self.sess.run(self.summary_op, feed_dict)
        self.summary_writer.add_summary(s, step)
        # If reward data provided the logging operations simply
        # pass this through palceholders to tensorboard.
        if episode_reward is not None and max_ep_reward is not None:
            reward_summary = self.sess.run(
                self.reward_summary,
                {self.avg_reward_summary_ph: episode_reward,
                 self.max_reward_summary_ph: max_ep_reward}
            )
            self.summary_writer.add_summary(reward_summary, step)

    def _align_value_net_initialisations(self):
        '''Ensure that value network paramaters are the same at initialisation'''
        # Build one operation to set target parameters equal to their
        # non-target equals.
        align_op = tf.group(*[
            v_t.assign(v) for v_t, v in zip(
                sorted(self.target_v_params, key=lambda v: v.name), sorted(self.v_params, key=lambda v: v.name))
        ])
        # Run the operation.
        self.sess.run(align_op)

    def save(self, path):
        '''
        Save the model parameters to a folder

        Arguments
        path - str
            The destination folder and filename prefix determining
            where to save paramater files.
        '''
        # Save Model
        self.saver.save(self.sess, path)

    def load(self, path):
        '''
        Load model parameters from disk

        Arguments
        path - str
            The folder and filename prefix where parameters have
            been saved.
        '''
        # Load Model
        self.saver.restore(self.sess, path)
