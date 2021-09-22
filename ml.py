import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gym

from tensorflow.keras import regularizers
from tf_agents.trajectories import time_step as ts

import base64
import IPython
import matplotlib

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.utils import common



class MNISTLearnEnv(py_environment.PyEnvironment):

    hiddenLayerAmount = 2
    nodeAmount = 2 ** 5             # 5 = 32    7 = 128    9 = 512
    batchSize = 32
    startingLearningRate = 0.01

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batchSize)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"))
    for layerIndex in range(hiddenLayerAmount):
        name = "dense{}".format(layerIndex)
        model.add(tf.keras.layers.Dense(nodeAmount, activation='relu', name=name))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name="denseLast"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=startingLearningRate)
    lossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    startingWeights = model.get_weights()


    def batchStep(self):
        self.optimizer.learning_rate.assign(self._state[0])

        for step, (xBatchTrain, yBatchTrain) in enumerate(self.dataset.take(2)):
            with tf.GradientTape() as tape:
                # (xBatchTrain, yBatchTrain) = self.dataset.take(1) # TODO: jotai viisaampaa tähä
                logits = self.model(xBatchTrain, training=True)
                lossValue = self.lossFunction(yBatchTrain, logits)
            grads = tape.gradient(lossValue, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return lossValue.numpy()



    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0, maximum=4,
            name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,),
            dtype=np.float64,
            minimum=[0, 0, 0, 0, 0, 0], maximum=[10, 100, 100, 100, 100, 100],
            name='observation')
        self._state = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print("reset")
        self.model.set_weights(self.startingWeights)    # set weights back to starting weights

        self._state = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float64))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if action == 0:
            pass
        elif action == 1:
            self._state[0] *= 1.01
        elif action == 2:
            self._state[0] *= 1.1
        elif action == 3:
            self._state[0] *= 0.99
        elif action == 4:
            self._state[0] *= 0.9
        else:
            raise ValueError('`action` should be 0, 1 or 2.')

        self._state[1] = self.batchStep()
        self._state[2] = self.batchStep()
        self._state[3] = self.batchStep()
        self._state[4] = self.batchStep()
        self._state[5] = self.batchStep()

        averageLoss = sum(self._state[1:])/5
        reward = (1 / averageLoss)

        if self._episode_ended or averageLoss < 0.1 or averageLoss > 5:
            self._episode_ended = True
            print("done: reward = %.5f" % reward, ", learningRate = %.5f" % self._state[0], ", loss = %.5f" % averageLoss)

            return ts.termination(np.array(self._state, dtype=np.float64), reward)
        else:
            print("done: reward = %.5f" % reward, ", learningRate = %.5f" % self._state[0], ", loss = %.5f" % averageLoss)

            return ts.transition(
                np.array(self._state, dtype=np.float64), reward=reward, discount=1)




def main():
    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    pyEnv = MNISTLearnEnv()
    # pyEnv = pyEnv.reset()
    env = tf_py_environment.TFPyEnvironment(pyEnv)

    print('action_spec:', env.action_spec())
    print('time_step_spec.observation:', env.time_step_spec().observation)
    print('time_step_spec.step_type:', env.time_step_spec().step_type)
    print('time_step_spec.discount:', env.time_step_spec().discount)
    print('time_step_spec.reward:', env.time_step_spec().reward)

    action = np.array(1, dtype=np.int32)
    time_step = env.reset()
    print(time_step)
    while not time_step.is_last():
        time_step = env.step(action)
        print(time_step)

    # print(env.observation_spec())
    utils.validate_py_environment(pyEnv, episodes=5)

    # @test {"skip": true}
    # env = env.reset()
    print(env)

    train_env = env
    eval_env = env

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(pyEnv.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))

    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)


    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()


    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    print("agentti tehty")

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    example_environment = train_env
    time_step = example_environment.reset()
    random_policy.action(time_step)

    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # See also the metrics module for standard implementations of different metrics.
    # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    agent.collect_data_spec
    agent.collect_data_spec._fields

    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer)

    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    # This loop is so common in RL, that we provide standard implementations.
    # For more details see tutorial 4 or the drivers module.
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/4_drivers_tutorial.ipynb
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    print("done")

