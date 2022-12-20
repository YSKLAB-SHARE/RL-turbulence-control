##################################
#
#
#
##################################

import tensorflow as tf # ver. 2.4.1
from tensorflow.keras import layers
import numpy as np # ver. 1.21.5
#import matplotlib.pyplot as plt
import TCF_Minimal_channel2 as TCF # solver
import time
import shutil

M1=24     # number of grid points in the streamwise direction
N1=16     # number of Fourier modes in the streamwise direction
M2=96     # number of grid points in the wall normal direction
N2=64     # number of Chebyshev modes in the wall normal direction
M3=24     # number of grid points in the spanwise direction
N3=16     # number of Fourier modes in the streamwise direction
NH33=7    # N3/2 - 1
MH33=11   # M3/2 - 1


ITL =  10000 # number of time steps in one episode
IYD = 14 # index of the detection plan height

total_episodes = 200 # number of episodes

RL_step = 10 # number of steps for linear interpolation
w = 0.0 # weight in reward for control inputs

num_states = 2 # number of states (u', v')
num_actions = 1 # number of action

X = int(ITL/RL_step)

# upper and lower bound for the control input
upper_bound = 1.0
lower_bound = -1.0

# for plotting the policy
u_range = 201
v_range = 201

# statistics at y^+ = 15
u_mean = 10.55
u_rms = 2.70
v_rms = 0.40
w_rms = 0.82

# Initial_field
def Initial_field():
    fil1 = open('U1_ini.DAT','rb')
    u1_ini = np.fromfile(fil1, np.float64, -1).reshape((N1+1, M2+1, NH33+1,3), order='F')

    fil2 = open('U2_ini.DAT','rb')
    u2_ini = np.fromfile(fil2, np.float64, -1).reshape((N1+1, M2+1, NH33+1,3), order='F')

    fil1.close()
    fil2.close()
    fil3 = open('N1_ini.DAT','rb')
    n1_ini = np.fromfile(fil3, np.float64, -1).reshape((N1+1, N2+1, NH33+1,3,2), order='F')

    fil4 = open('N2_ini.DAT','rb')
    n2_ini = np.fromfile(fil4, np.float64, -1).reshape((N1+1, N2+1, NH33+1,3,2), order='F')

    fil3.close()
    fil4.close()

    return u1_ini,u2_ini,n1_ini,n2_ini

# Noise
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# Buffer and training
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

        return self.state_buffer,self.action_buffer,self.reward_buffer,self.next_state_buffer

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)


# Actor network
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states))
    out = layers.Dense(8, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)

    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    # Our upper bound is 2.0
    outputs = lower_bound + (upper_bound - lower_bound)*(1+outputs)/2.0
    model = tf.keras.Model(inputs, outputs)
    return model


# Critic network
def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(8, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(16, activation="relu")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(16, activation="relu")(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

# Noise
std_dev = 0.1
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# create networks
actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(5000000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

ep_CF = []
ep_V_E = []

CF_array = np.array(np.zeros((total_episodes, ITL)))
V_E_array = np.array(np.zeros((total_episodes, ITL)))

#last_CF = np.array(np.zeros(last_ITL))
policy_array = np.array(np.zeros((total_episodes, u_range,v_range)))
policy_state = np.array(np.zeros((u_range,v_range,num_states)))
for i in range(u_range):
    for j in range(v_range):
        policy_state[i,j,0] = (-7+0.07*i)/u_rms
        policy_state[i,j,1] = (-1+0.01*j)/v_rms


input11 = np.array(np.zeros((M1+1, MH33+1)))
input12 = np.array(np.zeros((M1+1, MH33+1)))
input21 = np.array(np.zeros((M1+1, MH33+1)))
input22 = np.array(np.zeros((M1+1, MH33+1)))

u1_ini = np.array(np.zeros((N1+1, M2+1, NH33+1,3)))
u2_ini = np.array(np.zeros((N1+1, M2+1, NH33+1,3)))
n1_ini = np.array(np.zeros((N1+1, N2+1, NH33+1,3,2)))
n2_ini = np.array(np.zeros((N1+1, N2+1, NH33+1,3,2)))

reward_array1 = np.array(np.zeros((M1+1, M3)))
reward_array2 = np.array(np.zeros((M1+1, M3)))
prev_state_array1 = np.array(np.zeros((M1+1,M3,num_states)))
prev_state_array2 = np.array(np.zeros((M1+1,M3,num_states)))
state_array1 = np.array(np.zeros((M1+1,M3,num_states)))
state_array2 = np.array(np.zeros((M1+1,M3,num_states)))
action_array1 = np.array(np.zeros((M1+1,M3)))
action_array2 = np.array(np.zeros((M1+1,M3)))
new_input11 = np.array(np.zeros((M1+1, MH33+1)))
new_input12 = np.array(np.zeros((M1+1, MH33+1)))
new_input21 = np.array(np.zeros((M1+1, MH33+1)))
new_input22 = np.array(np.zeros((M1+1, MH33+1)))
old_input11 = np.array(np.zeros((M1+1, MH33+1)))
old_input12 = np.array(np.zeros((M1+1, MH33+1)))
old_input21 = np.array(np.zeros((M1+1, MH33+1)))
old_input22 = np.array(np.zeros((M1+1, MH33+1)))
noise_array1 = np.array(np.zeros((M1+1,M3)))
noise_array2 = np.array(np.zeros((M1+1,M3)))


#actor_model.load_weights('./my_checkpoint1')
#actor_model.load_weights('./my_checkpoint%d' %policy_number)
#actor_model.save_weights('./result/NN/Actor/my_checkpoint%d'%(total_episodes))
#critic_model.save_weights('./result/NN/Critic/my_checkpoint%d'%(total_episodes))

for ep in range(total_episodes):
    u1_ini,u2_ini,n1_ini,n2_ini = Initial_field()

    CF = []
    V_E = []


    old_input11[0:M1+1,0:MH33+1] = 0
    old_input12[0:M1+1,0:MH33+1] = 0
    old_input21[0:M1+1,0:MH33+1] = 0
    old_input22[0:M1+1,0:MH33+1] = 0
    new_input11[0:M1+1,0:MH33+1] = 0
    new_input12[0:M1+1,0:MH33+1] = 0
    new_input21[0:M1+1,0:MH33+1] = 0
    new_input22[0:M1+1,0:MH33+1] = 0
    input11[0:M1+1,0:MH33+1] = 0
    input12[0:M1+1,0:MH33+1] = 0
    input21[0:M1+1,0:MH33+1] = 0
    input22[0:M1+1,0:MH33+1] = 0

    # call the flow solver
    u, CF_i, V_E_i,ONL1,ONL2 = TCF.main(0,u1_ini,u2_ini,input11,input12,input21,input22,n1_ini,n2_ini)

    #############
    # INPUT of the solver:
    #   JMAX : number of time steps
    #   u1_ini, u2_ini : initial conditions
    #   input11, input12, input21, input22 : control input for up and bottom walls
    #   n1_ini, n2_ini : non-linear terms for Adams-Bashforth
    #
    # OUTPUT of the solver:
    #   u : velocity (and pressure) fields
    #   CF_i : skin friction coefficients
    #   V_E_i : mean control input ^2
    #   ONL1, ONL2 : non-linear terms for Adams-Bashforth
    #############

    for i in range(M1+1):
        for k in range(M3):
            prev_state_array1[i,k,0] = (u[i,IYD,k,0]-np.average(u[0:,IYD,0:,0]))/u_rms
            prev_state_array1[i,k,1] = u[i,IYD,k,1]/v_rms

            prev_state_array2[i,k,0] = (u[i,M2-IYD,k,0]-np.average(u[0:,M2-IYD,0:,0]))/u_rms
            prev_state_array2[i,k,1] = u[i,M2-IYD,k,1]/v_rms

    action_ave = []
    episodic_reward = 0

    for p in range(int(ITL/RL_step)):
#        start = time.time()

        action_average_array = []
        action_weight = []
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()
        old_input11[0:M1+1,0:MH33+1] = new_input11[0:M1+1,0:MH33+1]
        old_input12[0:M1+1,0:MH33+1] = new_input12[0:M1+1,0:MH33+1]
        old_input21[0:M1+1,0:MH33+1] = new_input21[0:M1+1,0:MH33+1]
        old_input22[0:M1+1,0:MH33+1] = new_input22[0:M1+1,0:MH33+1]

        prev_state_array1[0:M1+1,0:M3,1] = -prev_state_array1[0:M1+1,0:M3,1]

        prev_state1 = prev_state_array1.reshape(((M1+1)*M3, num_states))
        prev_state2 = prev_state_array2.reshape(((M1+1)*M3, num_states))

        tf.squeeze(prev_state1)
        tf.squeeze(prev_state2)
        sampled_actions1 = tf.squeeze(actor_model(prev_state1))
        sampled_actions2 = tf.squeeze(actor_model(prev_state2))

        for k in range(M3):
            for i in range(M1+1):
#                ou_noise.reset()
                noise_array1[i,k] = ou_noise()
#                ou_noise.reset()
                noise_array2[i,k] = ou_noise()

        legal_action1 = np.clip(sampled_actions1, lower_bound, upper_bound)
        legal_action2 = np.clip(sampled_actions2, lower_bound, upper_bound)

        action1 = np.squeeze(legal_action1)
        action2 = np.squeeze(legal_action2)
        action1 = np.array([action1],dtype = 'float64')
        action2 = np.array([action2],dtype = 'float64')
        action_array1 = action1.reshape((M1+1, M3))+noise_array1
        action_array2 = action2.reshape((M1+1, M3))+noise_array2

        prev_state_array1[0:M1+1,0:M3,1] = -prev_state_array1[0:M1+1,0:M3,1]

        new_input11[0:M1+1,0:MH33+1] = -action_array1[0:M1+1, 0:MH33+1]
        new_input12[0:M1+1,0:MH33+1] = -action_array1[0:M1+1, MH33+1:M3]
        new_input21[0:M1+1,0:MH33+1] = action_array2[0:M1+1, 0:MH33+1]
        new_input22[0:M1+1,0:MH33+1] = action_array2[0:M1+1, MH33+1:M3]

        for y in range(RL_step):
            input11[0:M1+1,0:MH33+1] = old_input11[0:M1+1,0:MH33+1]+(new_input11[0:M1+1,0:MH33+1]-old_input11[0:M1+1,0:MH33+1])*(y)/RL_step
            input12[0:M1+1,0:MH33+1] = old_input12[0:M1+1,0:MH33+1]+(new_input12[0:M1+1,0:MH33+1]-old_input12[0:M1+1,0:MH33+1])*(y)/RL_step
            input21[0:M1+1,0:MH33+1] = old_input21[0:M1+1,0:MH33+1]+(new_input21[0:M1+1,0:MH33+1]-old_input21[0:M1+1,0:MH33+1])*(y)/RL_step
            input22[0:M1+1,0:MH33+1] = old_input22[0:M1+1,0:MH33+1]+(new_input22[0:M1+1,0:MH33+1]-old_input22[0:M1+1,0:MH33+1])*(y)/RL_step

            # call the flow solver
            u, CF_i, V_E_i,ONL1,ONL2 = TCF.main(1,VF1,VF2,input11,input12,input21,input22,ONL1,ONL2)

            CF.append(CF_i)
            V_E.append(V_E_i)


        reward = -CF_i-w*V_E_i

        for i in range(M1):
            for k in range(M3):

                state_array1[i,k,0] = (u[i,IYD,k,0]-np.average(u[0:,IYD,0:,0]))/u_rms
                state_array1[i,k,1] = u[i,IYD,k,1]/v_rms
                state_array2[i,k,0] = (u[i,M2-IYD,k,0]-np.average(u[0:,M2-IYD,0:,0]))/u_rms
                state_array2[i,k,1] = u[i,M2-IYD,k,1]/v_rms

        prev_state_array1[0:M1+1,0:M3,1] = -prev_state_array1[0:M1+1,0:M3,1]
        state_array1[0:M1+1,0:M3,1] = -state_array1[0:M1+1,0:M3,1]
        for k in range(M3):
            for i in range(M1):
                state_buffer,action_buffer,reward_buffer,next_state_buffer=buffer.record((prev_state_array1[i,k],-u[i,0,k,1],reward,state_array1[i,k]))
                state_buffer,action_buffer,reward_buffer,next_state_buffer=buffer.record((prev_state_array2[i,k],u[i,M2,k,1],reward,state_array2[i,k]))

        prev_state_array1[0:M1+1,0:M3,1] = -prev_state_array1[0:M1+1,0:M3,1]
        state_array1[0:M1+1,0:M3,1] = -state_array1[0:M1+1,0:M3,1]
        episodic_reward += reward


        buffer.learn()
        update_target(tau)
        prev_state_array1[0:M1+1,0:M3,0:2] = state_array1[0:M1+1,0:M3,0:2]
        prev_state_array2[0:M1+1,0:M3,0:2] = state_array2[0:M1+1,0:M3,0:2]

    ep_reward_list.append(episodic_reward)
    ep_CF.append(np.average(CF))
    ep_V_E.append(np.average(V_E))

    CF_array[ep,0:ITL] = CF[0:ITL]
    V_E_array[ep,0:ITL] = V_E[0:ITL]

    policy_state1 = policy_state.reshape((u_range*v_range, num_states))

    policy_action = tf.squeeze(actor_model(policy_state1))
    policy_action = np.clip(policy_action, lower_bound, upper_bound)
    policy_action = np.squeeze(policy_action)
    policy_action = policy_action.reshape((u_range,v_range))

    for i in range(u_range):
        for j in range(v_range):
            policy_array[ep,i,j] = policy_action[i,j]


    actor_model.save_weights('./result/NN/Actor/my_checkpoint%d'%(ep))
    critic_model.save_weights('./result/NN/Critic/my_checkpoint%d'%(ep))



actor_model.save_weights('./continue_learning/NN/Actor/my_checkpoint')
critic_model.save_weights('./continue_learning/NN/Critic/my_checkpoint')
target_actor.save_weights('./continue_learning/NN/Target_Actor/my_checkpoint')
target_critic.save_weights('./continue_learning/NN/Target_Critic/my_checkpoint')
np.save('./continue_learning/Buffer/state_buffer.npy', state_buffer)
np.save('./continue_learning/Buffer/action_buffer.npy', action_buffer)
np.save('./continue_learning/Buffer/reward_buffer.npy', reward_buffer)
np.save('./continue_learning/Buffer/next_state_buffer.npy', next_state_buffer)
np.save('./result/Policy_array.npy', policy_array) # policy
np.save('./result/CF_array.npy', CF_array) # CF values
np.save('./result/V_E_array.npy', V_E_array) # control input ^2
shutil.make_archive('./result/continue_learning', 'zip', root_dir='./continue_learning')
