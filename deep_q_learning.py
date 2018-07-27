import random
import numpy as np
import tensorflow as tf
import pycar
import spmax
import scipy
import timer
import gym

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

def env_initializer(env, state):
    env.env.state = state
    env.env.steps_beyond_done = None

class DQNAgent:
    def __init__(self, obs_dim, n_action, seed=0,
                 discount_factor = 0.99, epsilon_decay = 0.999, epsilon_min = 0.01,
                 learning_rate = 1e-3, # STEP SIZE
                 batch_size = 256, 
                 memory_size = 10000, hidden_unit_size = 64,
                 target_mode = 'DDQN', memory_mode = 'PER', policy_mode='argmax', restore=False, save_on=False, net_dir=''):
        self.seed = seed 
        self.obs_dim = obs_dim
        self.n_action = n_action

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = 500

        self.target_mode = target_mode
        self.memory_mode = memory_mode
        self.policy_mode = policy_mode
        self.q_alpha = 1
        self.gamma = 0.99

        if memory_mode == 'PER':
            self.memory = ReplayMemory(memory_size=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
            
        self.max_k = 5
        self.max_d = 5

        self.hidden_unit_size = hidden_unit_size
        self.restore = restore
        self.net_dir = net_dir
        self.save_on = save_on

        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.build_update_operation()
            if self.save_on:
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
        
        with tf.variable_scope('q_func_new'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            if self.policy_mode=='argmax':
                self.q_predict = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
            elif self.policy_mode=='spmax':
                self.q_predict = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
                self.q_dist = tf.contrib.sparsemax.sparsemax(self.q_predict/self.q_alpha, name='q_dist')
                self.q_dist_multi = tf.distributions.Multinomial([1.], probs=self.q_dist)
                self.q_sample = self.q_dist_multi.sample()
            elif self.policy_mode=='softmax':
                self.q_predict = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
                self.q_dist = tf.nn.softmax(self.q_predict/self.q_alpha, name='q_dist')
                self.q_dist_multi = tf.distributions.Multinomial([1.], probs=self.q_dist)
                self.q_sample = self.q_dist_multi.sample()
                        
        with tf.variable_scope('q_func_old'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            if self.policy_mode=='argmax':
                self.q_predict_old = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
            elif self.policy_mode=='spmax':
                self.q_predict_old = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
                self.q_dist_old = tf.contrib.sparsemax.sparsemax(self.q_predict_old/self.q_alpha, name='q_dist')
                self.q_dist_multi_old = tf.distributions.Multinomial([1.], probs=self.q_dist_old)
                self.q_sample_old = self.q_dist_multi_old.sample()
            elif self.policy_mode=='softmax':
                self.q_predict_old = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
                self.q_dist_old = tf.nn.softmax(self.q_predict_old/self.q_alpha, name='q_dist')
                self.q_dist_multi_old = tf.distributions.Multinomial([1.], probs=self.q_dist_old)
                self.q_sample_old = self.q_dist_multi_old.sample()
        
        self.weights = tf.trainable_variables(scope='q_func_new')
        self.weights_old = tf.trainable_variables(scope='q_func_old')

        self.grad_ph = [tf.placeholder('float32',shape=w.shape) for w in self.weights]
        self.grad_op = [tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).apply_gradients([g_v]) for g_v in zip(self.grad_ph,self.weights)]

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

    def get_action(self, obs):
        if self.policy_mode=='argmax':
            if np.random.rand() <= self.epsilon:
                a = random.randrange(self.n_action)
                return a
            else:
                q_value = self.get_prediction([obs])
                return np.argmax(q_value[0])
        elif self.policy_mode=='spmax':
            q_sample_val = self.sess.run(self.q_sample,feed_dict={self.obs_ph:[obs]}) # one hot
            return np.argmax(q_sample_val[0])  
        elif self.policy_mode=='softmax':
            q_sample_val,pred = self.sess.run([self.q_sample,self.q_predict],feed_dict={self.obs_ph:[obs]}) # one hot
            return np.argmax(q_sample_val[0])

    def get_max_action(self,obs):
        if self.policy_mode=='argmax':
            q_value = self.sess.run(self.q_predict,feed_dict={self.obs_ph:[obs]})
            return np.argmax(q_value[0])
        elif self.policy_mode=='spmax':
            q_sample_val = self.sess.run(self.q_sample,feed_dict={self.obs_ph:[obs]}) # one hot
            return np.argmax(q_sample_val[0])
        elif self.policy_mode=='softmax':
            q_sample_val = self.sess.run(self.q_sample,feed_dict={self.obs_ph:[obs]}) # one hot
            return np.argmax(q_sample_val[0])

    def add_experience(self, obs, action, reward, next_obs, done):
        if self.memory_mode == 'PER':
            self.memory.save_experience(obs, action, reward, next_obs, done)
        else:
            self.memory.append((obs, action, reward, next_obs, done))

    def get_mct_q_val_k_sample(self, env, obs, act, k, d):

        if d==1:
            env_initializer(env, obs)
            next_obs, rwd, done, info = env.step(act)
            visit=1
            if done:
                mse = tf.square(rwd - self.q_predict[0][act])
                grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                gradients = list(list(zip(*gradients))[0])
                return rwd, gradients, loss, visit, next_obs, rwd, done, info
            else:
                next_q_value = self.get_prediction_old(np.expand_dims(next_obs,0))
                q_val_est = rwd + self.gamma * self.q_alpha * spmax.spmax(next_q_value/self.q_alpha)
                mse = tf.square(q_val_est - self.q_predict[0][act])
                grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0)})
                gradients = list(list(zip(*gradients))[0])
                return q_val_est, gradients, loss, visit, next_obs, rwd, done, info
        else:
            env_initializer(env, obs)
            next_obs, rwd, done, info = env.step(act)
            visit=1

            if done:
                mse = tf.square(rwd - self.q_predict[0][act])
                grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                loss, gradients = self.sess.run([mse, grad], feed_dict={self.obs_ph:np.expand_dims(obs,0)})
                gradients = list(list(zip(*gradients))[0])
                return rwd, gradients, loss, visit, next_obs, rwd, done, info

            dist = self.sess.run(self.q_dist, feed_dict={self.obs_ph:np.expand_dims(obs,0)})

            n_of_branch = np.random.multinomial(k, dist[0])
            n_of_branch = np.maximum(n_of_branch - 1, 0) + 1

            actions = [[i]*t[i] for i in range(len(t))]
            actions = np.concatenate(actions)

            q_val_list = np.zeros(self.n_action)

            gradients = [np.zeros(w.shape) for w in self.weights]
            loss = 0

            for i in range(k):
                next_q_val, next_gradients, next_loss, v, _, _, _, _ = self.get_mct_q_val_k_sample(env,next_obs,actions[i],k,d-1)
                gradients = self.grad_sum(gradients,next_gradients)
                loss += next_loss
                q_val_list[actions[i]] += next_q_val
                visit += v

            q_val_list = q_val_list * dist[0]
            q_val_list = q_val_list / n_of_branch_for_each_act

            q_val_est = rwd + self.gamma * np.sum(q_val_list)
            mse = tf.square(q_val_est - self.q_predict[0][act])
            new_grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
            new_loss, new_gradients = self.sess.run([mse, new_grad], feed_dict={self.obs_ph:np.expand_dims(obs,0)})
            new_gradients = list(list(zip(*new_gradients))[0])

            gradients = self.grad_devide(gradients,k)
            gradients = self.grad_sum(gradients,new_gradients)
            loss = (1/k)*loss + new_loss

            return q_val_est, gradients, loss, visit, next_obs, rwd, done, info

    def get_mct_q_val(self, env, obs, act, d):

        if d==1:
            env_initializer(env, obs)
            next_obs, rwd, done, info = env.step(act)
            visit=1
            if done:
                mse = tf.square(rwd - self.q_predict[0][act])
                #grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                #loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                #gradients = list(list(zip(*gradients))[0])
                loss=0
                gradients = 0
                return rwd, gradients, loss, visit, next_obs, rwd, done, info
            else:
                next_q_value = self.get_prediction_old(np.expand_dims(next_obs,0))
                q_val_est = rwd + self.gamma * self.q_alpha * spmax.spmax(next_q_value/self.q_alpha)
                mse = tf.square(q_val_est - self.q_predict[0][act])
                #grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                #loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                #gradients = list(list(zip(*gradients))[0])
                gradients = 0
                loss = 0
                return q_val_est, gradients, loss, visit, next_obs, rwd, done, info
        else:
            env_initializer(env, obs)
            next_obs, rwd, done, info = env.step(act)
            visit=1

            if done and d==self.max_d:
                mse = tf.square(rwd - self.q_predict[0][act])
                grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                gradients = list(list(zip(*gradients))[0])
                loss=0
                self.grad_mean(gradients)
                return rwd, gradients, loss, visit, next_obs, rwd, done, info
            elif done:
                mse = tf.square(rwd - self.q_predict[0][act])
                #grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                #loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                #gradients = list(list(zip(*gradients))[0])
                loss=0
                gradients=0
                return rwd, gradients, loss, visit, next_obs, rwd, done, info

            dist = self.sess.run(self.q_dist, feed_dict={self.obs_ph:np.expand_dims(next_obs,0)})

            q_val_list = np.zeros(self.n_action)
            gradients = [np.zeros(w.shape) for w in self.weights]
            loss = 0
            node_num = 0

            for i in range(self.n_action):
                if dist[0][i]>0:
                    next_q_val, next_gradients, next_loss, v, _, _, _, _ = self.get_mct_q_val(env,next_obs,i,d-1)
                    #gradients = self.grad_sum(gradients,next_gradients)
                    #loss += next_loss
                    q_val_list[i] = next_q_val
                    node_num += 1
                    visit += v
                else:
                    q_val_list[i] = 0

            q_val_list = q_val_list * dist[0]

            q_val_est = rwd + self.gamma * np.sum(q_val_list)
            if d==self.max_d:
                mse = tf.square(q_val_est - self.q_predict[0][act])
                new_grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                new_loss, new_gradients = self.sess.run([mse,new_grad], feed_dict={self.obs_ph:np.expand_dims(obs,0)})
                new_gradients = list(list(zip(*gradients))[0])
                return q_val_est, new_gradients, new_loss, visit, next_obs, rwd, done, info

                #gradients = self.grad_devide(gradients,node_num)
                #gradients = self.grad_sum(gradients,new_gradients)
                #loss = (1.0/node_num)*loss + new_loss

            return q_val_est, gradients, loss, visit, next_obs, rwd, done, info

    def get_mct_q_val_all_grad(self, env, obs, act, d):

        if d==1:
            env_initializer(env, obs)
            next_obs, rwd, done, info = env.step(act)
            visit=1
            if done:
                mse = tf.square(rwd - self.q_predict[0][act])
                grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                gradients = list(list(zip(*gradients))[0])
                return rwd, gradients, loss, visit, next_obs, rwd, done, info
            else:
                next_q_value = self.get_prediction_old(np.expand_dims(next_obs,0))
                q_val_est = rwd + self.gamma * self.q_alpha * spmax.spmax(next_q_value/self.q_alpha)
                mse = tf.square(q_val_est - self.q_predict[0][act])
                grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                gradients = list(list(zip(*gradients))[0])
                print (gradients)
                return q_val_est, gradients, loss, visit, next_obs, rwd, done, info
        else:
            env_initializer(env, obs)
            next_obs, rwd, done, info = env.step(act)
            visit=1

            if done:
                mse = tf.square(rwd - self.q_predict[0][act])
                grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
                loss, gradients = self.sess.run([mse,grad], feed_dict={self.obs_ph:np.expand_dims(obs,0),self.learning_rate_ph:self.learning_rate})
                gradients = list(list(zip(*gradients))[0])
                print (gradients)
                return rwd, gradients, loss, visit, next_obs, rwd, done, info

            dist = self.sess.run(self.q_dist, feed_dict={self.obs_ph:np.expand_dims(next_obs,0)})

            q_val_list = np.zeros(self.n_action)
            gradients = [np.zeros(w.shape) for w in self.weights]
            loss = 0
            node_num = 0

            for i in range(self.n_action):
                if dist[0][i]>0:
                    next_q_val, next_gradients, next_loss, v, _, _, _, _ = self.get_mct_q_val(env,next_obs,i,d-1)
                    gradients = self.grad_sum(gradients,next_gradients)
                    loss += next_loss
                    q_val_list[i] = next_q_val
                    node_num += 1
                    visit += v
                else:
                    q_val_list[i] = 0

            q_val_list = q_val_list * dist[0]

            q_val_est = rwd + self.gamma * np.sum(q_val_list)
            mse = tf.square(q_val_est - self.q_predict[0][act])
            new_grad = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).compute_gradients(mse,self.weights)
            new_loss, new_gradients = self.sess.run([mse,new_grad], feed_dict={self.obs_ph:np.expand_dims(obs,0)})
            new_gradients = list(list(zip(*gradients))[0])

            gradients = self.grad_devide(gradients,node_num)
            gradients = self.grad_sum(gradients,new_gradients)
            loss = (1.0/node_num)*loss + new_loss
            print (gradients)

            return q_val_est, gradients, loss, visit, next_obs, rwd, done, info

    def grad_mean(self,grad):
        g_m = 0
        g_size=0
        for gg in grad:
            g_m += np.sum(gg)
            g_size += gg.size
        print(g_m/g_size)

    def grad_sum(self,grad1,grad2):
        return [a+b for a, b in zip(grad1,grad2)]

    def grad_devide(self,grad,val):
        return [g/val for g in grad]
    
    def mct_search(self,env,obs,act):
        q_val, grad, loss, visit, next_obs, reward, done, info = self.get_mct_q_val_all_grad(env, obs, act, self.max_d)

        return grad, loss, visit, next_obs, reward, done, info

    def train_with_grad(self, grad):
        feed_dict = {g_ph:g_v for g_ph,g_v in zip(self.grad_ph,grad)}
        feed_dict[self.learning_rate_ph]=self.learning_rate

        self.sess.run(self.grad_op, feed_dict=feed_dict)    

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
                            target[i][actions[i]] = rewards[i] + self.discount_factor * self.q_alpha * spmax.spmax(next_q_value[i])
                        elif self.policy_mode=='softmax':
                            target[i][actions[i]] = rewards[i] + self.discount_factor * scipy.misc.logsumexp(next_q_value[i])

            loss, errors, _ = self.sess.run([self.loss, self.errors, self.optim], 
                                 feed_dict={self.obs_ph:observations,self.target_ph:target,self.learning_rate_ph:self.learning_rate,self.batch_weights_ph:batch_weights})
            errors = errors[np.arange(len(errors)), actions]
            
            if self.memory_mode == 'PER':
                # PRIORITIZED EXPERIENCE REPLAY
                self.memory.update_experience_weight(idx, errors)
            
        return loss

#env = pycar.env(visualize=True)
env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

mode = ['train','test']
cur_mode = 'train'

policy_mode = ['argmax','spmax','softmax']
cur_policy = 'spmax'

target_mode = ['DQN', 'DDQN', 'A3C', 'MCT']
cur_target = 'MCT'

max_t = env.spec.max_episode_steps
#max_t = env.time_limit
agent = DQNAgent(obs_dim,act_dim,memory_mode='PER',target_mode=cur_target, policy_mode=cur_policy,
                restore=False, net_dir='argmax/iter_56000.ckpt') # memory_mode='PER',target_mode='DDQN'

avg_return_list = deque(maxlen=10)
avg_loss_list = deque(maxlen=10)
avg_success_list = deque(maxlen=10)

result_saver = []

timer.start_timer()

print(cur_mode)
print(cur_policy)
print(cur_target)


async_num=4
if cur_target=='MCT' or cur_target=='A3C':
    async_grad = [np.zeros(w.shape) for w in agent.weights]
    obs_set = []
    for n in range(async_num):
        obs_set.append(env.reset())

    rwd_set=[0,0,0,0]

vv=0
for i in range(10000):
    obs = env.reset()
    done = False
    total_reward = 0
    total_loss = 0
    total_success = 0
    #if(i%1000==0):
    #    cur_mode='test'
    #    print ("test")
    #elif (i%1000==100):
    #    cur_mode='train'
    #    print ("train")


    for t in range(max_t):
        if cur_mode=='test':
            action = agent.get_max_action(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            loss=0
            if done:
                break
        else:
            if cur_target=='MCT' or cur_target=='A3C':
                async_obs = obs_set[i % async_num]            
                env_initializer(env, async_obs)
                action = agent.get_action(async_obs)
                grad, loss, v, next_obs, reward, done, info = agent.mct_search(env,async_obs,action)

                async_grad = agent.grad_sum(async_grad,grad)
                rwd_set[i % async_num]+=reward
                vv += v

                if i % async_num == (async_num-1):
                    agent.train_with_grad(async_grad)
                    async_grad = [np.zeros(w.shape) for w in agent.weights]

                if done:
                    print("async : {}".format(i%async_num))
                    print("visits : {}".format(vv))
                    print("reward : {}".format(rwd_set[i % async_num]))
                    rwd_set[i % async_num]=0
                    next_obs = env.reset()
                obs_set[i % async_num] = next_obs


                total_reward += reward
                total_loss += loss
                #total_success += info
                break

            else:
                action = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.add_experience(obs,action,reward,next_obs,done)
                vv+=1
        
                loss = agent.train_model()
                agent.update_memory(t,max_t)
                agent.update_policy()
        
                obs = next_obs

                if done:
                    break

        total_reward += reward
        total_loss += loss
        #total_success += info
        
    
    if cur_mode != 'test':
        agent.update_target()
        #if (i%100)==0:
        #    agent.saver.save(agent.sess, "./net/"+cur_policy+"/iter_{}.ckpt".format(i))

    avg_return_list.append(total_reward)
    avg_loss_list.append(total_loss)
    avg_success_list.append(total_success)

    #print("vvvvvvvvvvvv")
    #print(vv)
    #timer.print_time()
    
    if (np.mean(avg_return_list) > 490):
        print('{} loss : {:.3f}, return : {:.3f}, success : {:.3f}, eps : {:.3f}'.format(i, np.mean(avg_loss_list), np.mean(avg_return_list), np.mean(avg_success_list), agent.epsilon))
        print('The problem is solved with {} episodes'.format(i))
        #if cur_mode != 'test':
        #    agent.saver.save(agent.sess, "./net/"+cur_policy+"/iter_{}.ckpt".format(i))
        break
    
    if (i%10)==0:
        #result_saver.append({'i':i,'loss':np.mean(avg_loss_list),'return':np.mean(avg_return_list),'success':np.mean(avg_success_list),'eps':agent.epsilon})
        #np.save(cur_policy + '_result.npy',result_saver)
        timer.print_time()
        print('{} loss : {:.3f}, return : {:.3f}, success : {:.3f}, eps : {:.3f}'.format(i, np.mean(avg_loss_list), np.mean(avg_return_list), np.mean(avg_success_list), agent.epsilon))

