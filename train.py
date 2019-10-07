from utils import initialise_environment
import argparse
from soft_actor_critic import SAC


def train(agent, env, num_episodes, exploration_eps, max_episode_length, log_frequency, save_frequency, save_path):
    '''
    Train an agent using Soft Actor-Critic

    Arguments
    agent - SAC instance
        The agent to be trained
    env - gym.Env
        The environment in which the agent is to act
    num_episodes - int
        The number of episodes to run
    exploration_eps - int
        The number of episodes of random exploration to run
    max_episode_length - int
        The maximal number of timesteps to allow an episode to run for.
        If the episode is ongoing at this point terminate it.
    log_frequency - int
        The number of episodes between tensorboard log entries
    save_frequency - int
        The number of episodes between the model paramaters being saved
    save_path - str
        The path in which to save the model parameters
    '''
    # Initialise counts and reward tracking measures
    num_steps = 0
    av_ep_reward = 0
    max_ep_reward = -1e6
    # Outer loop over the number of episodes for training purposes
    for i in range(num_episodes):
        # Attain initial observation
        obs = env.reset()
        # Reset end of episode boolean
        terminal = False
        # Rest episode reward sum and time step tracking
        t = 0
        ep_reward = 0

        # Run a full episode
        while not terminal:
            # In the initial exploration phase actions are sampled
            # uniformly at random
            if i < exploration_eps:
                action = env.action_space.sample()
            # Once exploration is complete we query the agent's
            # policy for actions.
            else:
                action = agent.action(obs)
            # Pass action to the environment and receive information
            next_obs, reward, terminal, info = env.step(action)
            # Update tracked in-episode and across-episode measures
            ep_reward += reward
            num_steps += 1
            t += 1
            # The enforced end at the max_episode length
            if t == max_episode_length:
                terminal = True
            # Let the agent process and store the current time step of experience
            agent.experience(obs, action, reward, next_obs, terminal)
            # Let the agent train
            if i >= exploration_eps:
                agent.train()
            # Update the observation and continue the current episode
            obs = next_obs
        # Update a running average and maximum of episode rewards
        k = i % log_frequency
        av_ep_reward = k / (k+1) * av_ep_reward + ep_reward / (k+1)
        max_ep_reward = max(max_ep_reward, ep_reward)

        # If appropriate do logging
        if i > 0 and (i+1) % log_frequency == 0:
            # Run logging operations and reset average and maximum
            # counts
            agent.run_summary(i+1, av_ep_reward, max_ep_reward)
            av_ep_reward = 0
            max_ep_reward = -1e6
        # Save the model parameters at appropriate intervals
        if i > 0 and (i+1) % save_frequency == 0:
            agent.save(save_path)


def parse_args():
    '''Parse arguments passed from the command line'''
    parser = argparse.ArgumentParser('Test Arguments for Soft Actor-Critic.')
    # Choice of environment
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')

    # Training infrastructure options
    parser.add_argument('-s', '--save_path', type=str, default='./saved_model/model')
    parser.add_argument('-n', '--num_episodes', type=int, default=1000)
    parser.add_argument('-T', '--max_ep_len', type=int, default=1000)
    parser.add_argument('-lf', '--log_freq', type=int, default=100)
    parser.add_argument('-sf', '--save_freq', type=int, default=250)
    parser.add_argument('-ld', '--logdir', type=str, default='./logs')
    parser.add_argument('-ie', '--initial_exploration_eps', type=int, default=100)

    # Soft Actor-Critic Hyperparameters
    # See soft_actor_critic.py for details regarding what each hyperparameter does.
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('-plr', '--policy_lr', type=float, default=1e-3)
    parser.add_argument('-clr', '--critic_lr', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('-rc', '--replay_buffer_capacity', type=int, default=250000)
    parser.add_argument('-pns', '--policy_net_shape', type=tuple, default=(100, 30))
    parser.add_argument('-qns', '--q_net_shape', type=tuple, default=(100, 30))
    parser.add_argument('-vns', '--value_net_shape', type=tuple, default=(100, 30))

    # Process and return arguments
    return parser.parse_args()


def main(args):
    '''Initialise and train a Soft Actor-Critic Agent'''
    # Attain the environment
    env = initialise_environment(args.env)
    # Initialise the agent using the arguments provided
    agent = SAC(env, args.gamma, args.alpha, args.policy_lr, args.critic_lr,
                args.polyak, args.policy_net_shape, args.q_net_shape,
                args.value_net_shape, args.batch_size,
                args.replay_buffer_capacity, args.logdir)
    # Run the training function which handles logging and model saving
    train(agent, env, args.num_episodes, args.initial_exploration_eps,
          args.max_ep_len, args.log_freq, args.save_freq, args.save_path)


if __name__ == '__main__':
    # Parse the arguments and then run the main training function.
    args = parse_args()
    main(args)
