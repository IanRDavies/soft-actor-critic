# Soft Actor-Critic (Haarnoja et al., 2017)

In this repo we implement soft actor-critic and test it using the [continuous-actions implementation of Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) from [Open AI Gym](https://gym.openai.com). The Lunar Lander problem is considered solved by algorithms that attain a reward of 200 or higher. Our algorithm comfortably reaches this level and attains similar performance to the spinning up SAC implementation.

A set of Tensorboard log files and a final saved policy are included in this directory.

Our approach is object orientated and defined in `soft_actor_critic.py` with utility functions defined in `utils.py`. Use `train.py` to train a soft actor-critic agent and `test.py` to see the results in testing and watch the trained agent play.

We use hyperparameters borrowed from [Open AI's SAC implementation](https://spinningup.openai.com/en/latest/algorithms/sac.html). In general, our approach is similar to that of Open AI in that we choose not to learn the weighting of the entropy in the policy objective preferring to fix this as a hyperparameter for simplicity. Furthermore, we borrow a couple of tricks to stabilise the implementation. Our implementation is however more widely applicable as we separate the agent's computation from that of the environment etc.

Our implementation is build on [TensorFlow](https://www.tensorflow.org/) (v 1.14) and [NumPy](https://numpy.org/) (v 1.16.4).

See the comments in the code for more implementation details.

Included is a response to the original paper which summarises and critiques soft actor-critic. This is provided in `Soft Actor-Critic Response.pdf`.

### Original Paper
[Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," Deep Learning Symposium, NIPS 2017](https://arxiv.org/pdf/1801.01290.pdf)