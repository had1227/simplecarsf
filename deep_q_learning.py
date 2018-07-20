import random
import numpy as np
import tensorflow as tf
import pycar
import spmax
import scipy

from collections import deque

seed = 0
np.random.seed(seed)
random.seed(seed)

class ReplayMemory:
    def __init__(self, memory_size=10000, per_alpha=0.2, per_beta0=0.4):       
        self.memory = SumTree(capacity=memory_size)
        self.memory_size = memory_size
        self.per_alpha = per_alpha
        self.per_beta0 = per_beta0
        self.per_beta = per_beta0
        self.per_epsilon = 1E-6
        self.prio_max = 0
    
    def anneal_per_importance_sampling(self, step, max_step):
        self.per_beta = self.per_beta0 + step*(1-self.per_beta0)/max_step

    def error2priority(self, errors):
        return np.power(np.abs(errors) + self.per_epsilon, self.per_alpha)

    def save_experience(self, state, action, reward, state_next, done):
        experience = (state, action, reward, state_next, done)
        self.memory.add(np.max([self.prio_max, self.per_epsilon]), experience)
        
    def retrieve_experience(self, batch_size):
        idx = None
        priorities = None
        w = None

        idx, priorities, experience = self.memory.sample(batch_size)
        sampling_probabilities = priorities / self.memory.total()
        w = np.power(self.memory.n_entries * sampling_probabilities, -self.per_beta)
        w = w / w.max()
        return idx, priorities, w, experience
    
    def update_experience_weight(self, idx, errors ):
        priorities = self.error2priority(errors)
        for i in range(len(idx)):
            self.memory.update(idx[i], priorities[i])
        self.prio_max = max(priorities.max(), self.prio_max)
        
class SumTree:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

        self.write = 0
        self.n_entries = 0

        self.tree_len = len(self.tree)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1

        if left >= self.tree_len:
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            right = left + 1
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]

    def sample(self, batch_size):
        batch_idx = [None] * batch_size
        batch_priorities = [None] * batch_size
        batch = [None] * batch_size
        segment = self.total() / batch_size

        a = [segment*i for i in range(batch_size)]
        b = [segment * (i+1) for i in range(batch_size)]
        s = np.random.uniform(a, b)

        for i in range(batch_size):
            (batch_idx[i], batch_priorities[i], batch[i]) = self.get(s[i])

        return batch_idx, batch_priorities, batch

class DQNAgent:
    def __init__(self, obs_dim, n_action, seed=0,
                 discount_factor = 0.99, epsilon_decay = 0.999995, epsilon_min = 0.1,
                 learning_rate = 1e-3, # STEP SIZE
                 batch_size = 256, 
                 memory_size = 50000, hidden_unit_size = 64,
                 target_mode = 'DDQN', memory_mode = 'PER', policy_mode='argmax', restore=False, net_dir=''):
        self.seed = seed 
        self.obs_dim = obs_dim
        self.n_action = n_action

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = 1000

        self.target_mode = target_mode
        self.memory_mode = memory_mode
        self.policy_mode = policy_mode
        self.q_alpha = 1

        if memory_mode == 'PER':
            self.memory = ReplayMemory(memory_size=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
            
        self.hidden_unit_size = hidden_unit_size
        self.restore = restore
        self.net_dir = net_dir

        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.build_update_operation()
            self.saver = tf.train.Saver()
            self.init_session()
    
    def build_placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.target_ph = tf.placeholder(tf.float32, (None, self.n_action), 'target')
        self.batch_weights_ph = tf.placeholder(tf.float32,(None, self.n_action), name="batch_weights")
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr')        
    
    def build_model(self):
        hid1_size = self.hidden_unit_size  # 10 empirically determined
        hid2_size = self.hidden_unit_size
        
        with tf.variable_scope('q_func'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            if self.policy_mode=='argmax':
                self.q_predict = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
            elif self.policy_mode=='spmax':
                out = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_out')
                self.q_predict = tf.contrib.sparsemax.sparsemax(out/self.q_alpha, name='q_predict')
                self.q_dist = tf.distributions.Multinomial([1.], probs=self.q_predict)
                self.q_sample = self.q_dist.sample()
            elif self.policy_mode=='softmax':
                out = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_out')
                self.q_predict = tf.nn.softmax(out/self.q_alpha, name='q_predict')
                self.q_dist = tf.distributions.Multinomial([1.], probs=self.q_predict)
                self.q_sample = self.q_dist.sample()
                        
        with tf.variable_scope('q_func_old'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            if self.policy_mode=='argmax':
                self.q_predict_old = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
            elif self.policy_mode=='spmax':
                out = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_out')
                self.q_predict_old = tf.contrib.sparsemax.sparsemax(out/self.q_alpha, name='q_predict')
                self.q_dist_old = tf.distributions.Multinomial([1.], probs=self.q_predict_old)
            elif self.policy_mode=='softmax':
                out = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_out')
                self.q_predict_old = tf.nn.softmax(out/self.q_alpha, name='q_predict')
                self.q_dist = tf.distributions.Multinomial([1.], probs=self.q_predict_old)
        
        self.weights = tf.trainable_variables(scope='q_func')
        self.weights_old = tf.trainable_variables(scope='q_func_old')

    def build_loss(self):
        self.errors = self.target_ph - self.q_predict
        self.loss = 0.5*tf.reduce_mean(tf.square(self.target_ph - self.q_predict))
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.loss)
            
    def build_update_operation(self):
        update_ops = []
        for var, var_old in zip(self.weights, self.weights_old):
            update_ops.append(var_old.assign(var))
        self.update_ops = update_ops
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        if self.restore:
            self.saver.restore(self.sess, "./net/" + self.net_dir)
        else:
            self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_ops)
        
    def update_target(self):
        self.sess.run(self.update_ops)
    
    def update_memory(self, step, max_step):
        if self.memory_mode == 'PER':
            self.memory.anneal_per_importance_sampling(step,max_step)
        
    def update_policy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def get_prediction_old(self, obs): 
        q_value_old = self.sess.run(self.q_predict_old,feed_dict={self.obs_ph:obs})        
        return q_value_old
        
    def get_prediction(self, obs):        
        q_value = self.sess.run(self.q_predict,feed_dict={self.obs_ph:obs})        
        return q_value

    def get_argmax_action(self,obs):
        q_value = self.get_prediction([obs])
        a = np.argmax(q_value[0])
        return a

    def get_action(self, obs):
        if np.random.rand() <= self.epsilon:
            a = random.randrange(self.n_action)
            return a
        else:
            q_value = self.get_prediction([obs])
            return np.argmax(q_value[0])

    def get_max_action(self,obs):
        if self.policy_mode=='argmax':
            q_value = self.sess.run(self.q_predict,feed_dict={self.obs_ph:[obs]})
            return np.argmax(q_value[0])
        elif self.policy_mode=='spmax':
            q_sample_val = self.sess.run(self.q_sample,feed_dict={self.obs_ph:[obs]})
            return np.argmax(q_sample_val[0])
        elif self.policy_mode=='softmax':
            q_sample_val = self.sess.run(self.q_sample,feed_dict={self.obs_ph:[obs]})
            return np.argmax(q_sample_val[0])

    def add_experience(self, obs, action, reward, next_obs, done):
        if self.memory_mode == 'PER':
            self.memory.save_experience(obs, action, reward, next_obs, done)
        else:
            self.memory.append((obs, action, reward, next_obs, done))

    def train_model(self):
        loss = np.nan
        
        if self.memory_mode == 'PER':
            n_entries = self.memory.memory.n_entries
        else:
            n_entries = len(self.memory)
            
        if n_entries > self.train_start:
            
            if self.memory_mode == 'PER':
                # PRIORITIZED EXPERIENCE REPLAY
                idx, priorities, w, mini_batch = self.memory.retrieve_experience(self.batch_size)
                batch_weights = np.transpose(np.tile(w, (self.n_action, 1)))
            else:
                mini_batch = random.sample(self.memory, self.batch_size)
                batch_weights = np.ones((self.batch_size,self.n_action))

            observations = np.zeros((self.batch_size, self.obs_dim))
            next_observations = np.zeros((self.batch_size, self.obs_dim))
            actions, rewards, dones = [], [], []

            for i in range(self.batch_size):
                observations[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_observations[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            target = self.get_prediction(observations)
            if self.target_mode == 'DDQN':
                bast_a = np.argmax(self.get_prediction(next_observations),axis=1)

            next_q_value = self.get_prediction_old(next_observations)

            # BELLMAN UPDATE RULE 
            for i in range(self.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    if self.target_mode == 'DDQN':
                        target[i][actions[i]] = rewards[i] + self.discount_factor * next_q_value[i][bast_a[i]]
                    else:
                        if self.policy_mode=='argmax':
                            target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(next_q_value[i]))
                        elif self.policy_mode=='spmax':
                            target[i][actions[i]] = rewards[i] + self.discount_factor * self.alpha * spmax.spmax(next_q_value[i])
                        elif self.policy_mode=='softmax':
                            target[i][actions[i]] = rewards[i] + self.discount_factor * self.alpha * scipy.misc.logsumexp(next_q_value[i])

            loss, errors, _ = self.sess.run([self.loss, self.errors, self.optim], 
                                 feed_dict={self.obs_ph:observations,self.target_ph:target,self.learning_rate_ph:self.learning_rate,self.batch_weights_ph:batch_weights})
            errors = errors[np.arange(len(errors)), actions]
            
            if self.memory_mode == 'PER':
                # PRIORITIZED EXPERIENCE REPLAY
                self.memory.update_experience_weight(idx, errors)
            
        return loss


env = pycar.env(visualize=False)
obs_dim = env.observation_space
act_dim = env.action_space

mode = ['train','test']
cur_mode = 'train'

max_t = env.time_limit
agent = DQNAgent(env.observation_space,env.action_space,memory_mode='PER',target_mode='DDQN', policy_mode='argmax',
                restore=False, net_dir='q_learning_iter_9900.ckpt') # memory_mode='PER',target_mode='DDQN'

avg_return_list = deque(maxlen=100)
avg_loss_list = deque(maxlen=100)
avg_success_list = deque(maxlen=100)

result_saver = []

for i in range(10000):
    obs = env.reset()
    done = False
    total_reward = 0
    total_loss = 0
    total_success = 0
    for t in range(max_t):
        if cur_mode=='test':
            action = agent.get_max_action(obs)
            next_obs, reward, done, info = env.step(action)
        else:
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.add_experience(obs,action,reward,next_obs,done)
        
            loss = agent.train_model()
            agent.update_memory(t,max_t)
            agent.update_policy()
        
            obs = next_obs

        total_reward += reward
        total_loss += loss
        total_success += info
        
        if done:
            break
    
    if cur_mode != 'test':
        if (i%4)==0:
            agent.update_target()
        if (i%100)==0:
            agent.saver.save(agent.sess, "./net/q_learning_iter_{}.ckpt".format(i))

    avg_return_list.append(total_reward)
    avg_loss_list.append(total_loss)
    avg_success_list.append(total_success)
    
    if (i > 100 and np.mean(avg_success_list) > 0.95):
        print('{} loss : {:.3f}, return : {:.3f}, success : {:.3f}, eps : {:.3f}'.format(i, np.mean(avg_loss_list), np.mean(avg_return_list), np.mean(avg_success_list), agent.epsilon))
        print('The problem is solved with {} episodes'.format(i))
        if cur_mode != 'test':
            agent.saver.save(agent.sess, "./q_learning_iter_{}.ckpt".format(i))
        break
    
    if (i%100)==0:
        result_saver.append({'i':i,'loss':np.mean(avg_loss_list),'return':np.mean(avg_return_list),'success':np.mean(avg_success_list),'eps':agent.epsilon})
        np.save('result.npy',result_saver)
        print('{} loss : {:.3f}, return : {:.3f}, success : {:.3f}, eps : {:.3f}'.format(i, np.mean(avg_loss_list), np.mean(avg_return_list), np.mean(avg_success_list), agent.epsilon))

