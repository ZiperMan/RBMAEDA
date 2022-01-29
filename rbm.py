import numpy as np
import math

from PIL import Image
import matplotlib.pyplot as plt
import time

from numpy.core.fromnumeric import cumsum
from numpy.random.mtrand import sample

"""
######################### ALL HYPER PARAMETERS #######################
learning rate
s.d. of Gaussian for weight initialization
number of hidden units
numbers of steps to skip in gibbs chain when sampling
number of parallel gibbs chains during sampling
"""

# TODO: at least give better name
def not_one_not_zero(x):
    if x==1: return 0.99
    if x==0: return 0.01
    return x

class RBM:
    # TODO: myb remove n_visible from construstor arguments and calculate it from training_set

    """
    n_visible - number of visible units
    n_hidden - number of hidden units
    f_activation - vectorized function of activation
    training_set - list of binary lists of size n_visible
    """
    def __init__(self, n_visible, n_hidden, f_activation=None, unit_type="binary", softmax_size=None) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if f_activation is None:
            self.f_activation = self.sigmoid
        else:
            self.f_activation = np.vectorize(f_activation)

        self.unit_type = unit_type
        self.softmax_size = softmax_size

        self.rec_error_arr = [] # DEBUGG

    # TODO: probaj v biase na 0
    def init_visible_biases(self) -> np.ndarray:
        if self.unit_type == "softmax":
            return np.zeros((self.softmax_size, self.n_visible))

        return np.zeros(self.n_visible)
        """
        Let p be the proportion of training vectors where unit i is on.
        We calculate visible bias i as log(p / (1 - p)).
        We do this for every visible bias.
        """
        prop_vec = np.sum(self.training_set, axis=0) / len(self.training_set)
        prop_vec = np.vectorize(not_one_not_zero)(prop_vec) # Removing zeros and ones
        return np.log2(prop_vec / (1 - prop_vec))

    def init_hidden_biases(self) -> np.ndarray:
        return np.zeros(self.n_hidden)

    def init_weights(self) -> np.ndarray:
        if self.unit_type == 'softmax':
            return np.random.normal(0, 0.0001, (self.softmax_size, self.n_visible, self.n_hidden))

        return np.random.normal(0, 0.0001, (self.n_visible, self.n_hidden)) # TODO: try decreasing/increasing s.d.

    def init_network(self) -> None:
        self.v_biases = self.init_visible_biases()
        self.h_biases = self.init_hidden_biases()
        self.weights = self.init_weights()

    # if prob == True return raw probabilities aka dont sample
    def activate_hidden(self, visible, prob=False):

        if self.unit_type == 'softmax':
            prob_vec = np.zeros(self.n_hidden)

            for k in range(self.softmax_size):
                prob_vec += np.matmul(visible[k], self.weights[k])

            prob_vec = self.f_activation(prob_vec + self.h_biases)
        else:
            prob_vec = self.f_activation( np.matmul(visible, self.weights)  + self.h_biases) # TODO: check

        if prob: return prob_vec

        # print("visible shape")
        # print(visible.shape)
        # print("weights shape")
        # print(self.weights.shape)
        # print("PROB VEC SHAPE")
        # print(prob_vec.shape)

        # Sample from Bernoulli for each element of vector
        return np.random.binomial(1, prob_vec)

    # if prob == True return raw probabilities aka dont sample
    def activate_visible(self, hidden, prob=False):
        if self.unit_type == 'softmax':
            # actually this is probability matrix, not probability vector
            prob_vec = np.zeros((self.softmax_size, self.n_visible))
            normalization_vec = np.zeros(self.n_visible)

            for k in range(self.softmax_size):
                prob_vec[k] = np.exp(np.matmul(self.weights[k], hidden) + self.v_biases[k])
                normalization_vec += prob_vec[k]

            prob_vec /= normalization_vec
        else:
            prob_vec = self.f_activation(np.matmul(self.weights, hidden) + self.v_biases) # TODO: check

        if prob: return prob_vec

        if self.unit_type == 'softmax':
            # activation of visible units is different for softmax units
            activation_vec = []

            for row in prob_vec.T:
                idx = self.sample_roulette(row)
                new_arr = np.zeros(self.softmax_size)
                new_arr[idx] = 1
                activation_vec.append(new_arr)

            return np.array(activation_vec).T

        # print("hidden shape")
        # print(hidden.shape)
        # print("weights shape")
        # print(self.weights.shape)
        # print("PROB VEC SHAPE")
        # print(prob_vec.shape)
        # print("v_biases shape")
        # print(self.v_biases.shape)

        return np.random.binomial(1, prob_vec) # Sample from Bernoulli for each element of vector

    """
    We generate reconstruction (of training_example) by apllying alternative Gibbs sampling k times starting at the training_example
    """
    def generate_reconstruction(self, training_example, k=1):
        reconstruction = training_example

        for i in range(k):
            reconstruction = self.activate_visible(self.activate_hidden(reconstruction))

        return reconstruction

    """
    Contrastive Divergence algorithm. Aproximation of the gradient.
    Returns 3 gradients(vectors): for weigths, hidden and visible biases
    """
    def cd_k(self, training_example, persistant=False, index_in_mini_batch=None):

        if persistant:
            reconstruction = self.generate_reconstruction(self.persistant_chain[index_in_mini_batch], self.k)
        else:
            reconstruction = self.generate_reconstruction(training_example, self.k)

        self.curr_rec_error = np.square(np.sum(np.fabs(training_example - reconstruction))) # DEBUGG

        # get activation probabilities of hidden units
        hidden_activation_by_training = self.activate_hidden(training_example, True)
        hidden_activation_by_reconstruction = self.activate_hidden(reconstruction, True)

        hidden_bias_gradient = hidden_activation_by_training - hidden_activation_by_reconstruction
        visible_bias_gradient = training_example - reconstruction

        # TODO: myb do this globally
        # make sure all vectors are row vectors
        hidden_activation_by_training.shape = (1, self.n_hidden)
        hidden_activation_by_reconstruction.shape = (1, self.n_hidden)

        if self.unit_type == "binary":
            reconstruction.shape = (1, self.n_visible)
            training_example.shape = (1, self.n_visible)

            weights_gradient = np.matmul(training_example.T, hidden_activation_by_training) - np.matmul(reconstruction.T, hidden_activation_by_reconstruction)

        if self.unit_type == "softmax":
            weights_gradient = []

            for k in range(self.softmax_size):
                rec = reconstruction[k]
                rec.shape = (1, self.n_visible)
                train_exmp = training_example[k]
                train_exmp.shape = (1, self.n_visible)
                weights_gradient.append(np.matmul(train_exmp.T, hidden_activation_by_training) - np.matmul(rec.T, hidden_activation_by_reconstruction))

            weights_gradient = np.array(weights_gradient)

        return {"weigths": weights_gradient, "hidden_biases": hidden_bias_gradient, "visible_biases": visible_bias_gradient}

    """
    Stochastic gradient descent on negative average log likelihood for self.n_epochs epochs on training set self.training_set and with the mini batch size self.mini_batch_size

    Since calculating exact gradient is intractable we use some aproximation:
        self.alg="cd_k" - Use Contrastive Divergence algorithm to aproximate the gradient
        self.alg="pcd" - Use Persistant Contrastive Divergence algorithm to aproximate the gradient
    """
    def sgd(self, persistant=False):
        for i in range(self.n_epochs):
            s_time = time.time()

            # Shuffle training set. Simulation of randomly sampling mini batches
            np.random.shuffle(self.training_set)

            # debbug_plot_hidden_activation_probs = True # DEBUG

            avg_rec_error = 0 # DEBUG

            n_minibatches = len(self.training_set) / self.mini_batch_size # DEBUG
            cnt = 1 # DEBUG

            first_mini_batch = True

            for mini_batch in np.array_split(self.training_set, math.floor(len(self.training_set) / self.mini_batch_size)):
                # if debbug_plot_hidden_activation_probs: # DEBUG
                #     self.plot_hidden_units_activation_probabilities(mini_batch, i + 1) # DEBUG
                # debbug_plot_hidden_activation_probs = False  # DEBUG

                # Use first mini batch to initialize persistent chain for PCD
                if persistant and first_mini_batch:
                    self.persistant_chain = []

                    for exmp in mini_batch:
                        self.persistant_chain.append(exmp)

                    first_mini_batch = False

                # Calculate avarage gradient
                if self.unit_type == "binary":
                    avg_visible_biases_gradient = np.zeros(self.n_visible)
                    avg_hidden_biases_gradient = np.zeros(self.n_hidden)
                    avg_weights_gradient = np.zeros(self.weights.shape)

                if self.unit_type == "softmax":
                    avg_visible_biases_gradient = np.zeros((self.softmax_size, self.n_visible))
                    avg_hidden_biases_gradient = np.zeros(self.n_hidden)
                    avg_weights_gradient = np.zeros(self.weights.shape)

                for j in range(self.mini_batch_size):
                    training_example = mini_batch[j]
                    # for training_example in mini_batch:
                    # get estimate of the gradient for training_example using cd_k
                    gradients = self.cd_k(training_example, persistant, j)

                    avg_rec_error += self.curr_rec_error # DEBUGG

                    avg_visible_biases_gradient = avg_visible_biases_gradient + gradients["visible_biases"]
                    avg_hidden_biases_gradient = avg_hidden_biases_gradient + gradients["hidden_biases"]
                    avg_weights_gradient = avg_weights_gradient + gradients["weigths"]

                avg_visible_biases_gradient = avg_visible_biases_gradient / self.mini_batch_size
                avg_hidden_biases_gradient = avg_hidden_biases_gradient / self.mini_batch_size
                avg_weights_gradient = avg_weights_gradient / self.mini_batch_size

                # update weights and biases
                self.v_biases = self.v_biases + self.learning_rate * avg_visible_biases_gradient
                self.h_biases = self.h_biases + self.learning_rate * avg_hidden_biases_gradient
                self.weights = self.weights + self.learning_rate * avg_weights_gradient

                # update persistant chain for PCD
                if persistant:
                    for j in range(self.mini_batch_size):
                        self.persistant_chain[j] = self.generate_reconstruction(self.persistant_chain[j], self.k)


                # if cnt == n_minibatches: # DEBUG
                #     self.plot_hist(i, avg_weights_gradient, avg_hidden_biases_gradient, avg_visible_biases_gradient) # DEBUG

                cnt += 1 # DEBUG


            self.rec_error_arr.append(avg_rec_error / len(self.training_set)) # DEBUG

            # self.plot_filters(i) # DEBUG

            # print(">>>>>>>>>>>>>>>>>>>>>>> EPOCH " + str(i + 1) + " time to pass: " + str(time.time() - s_time))

        # self.plot_rec_err() # DEBUG

    # TODO: use some other stopping creteria rather than n_epochs
    # Train RBM using stochastic gradient descent on negative average log likelihood
    def train(self, training_set, learning_rate=0.1, mini_batch_size=10, n_epochs=100, alg="cd_k", k=1):
        # print("start training")
        self.training_set = np.array(training_set)
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.alg = alg
        self.k = k
        self.init_network()

        if self.unit_type == "softmax":
            self.vec_to_softmax()

        if alg == "cd_k":
            self.sgd()

        if alg == "pcd":
            self.sgd(persistant=True)

    def sample(self, n_samples):
        reconstruction = self.training_set[np.random.randint(len(self.training_set))]

        # skip first 1000 samples in the chain
        reconstruction = self.generate_reconstruction(reconstruction, 1000)

        # if self.unit_type == "binary": # DEBUG
        #     self.plot_sample(reconstruction) # DEBUG
        # else: # DEBUG
        #     print("SAMPLE", reconstruction) # DEBUG

        samples = []

        for i in range(n_samples):
            reconstruction = self.generate_reconstruction(reconstruction)
            samples.append(reconstruction)

        if self.unit_type == "softmax":
            return self.softmax_to_vec(samples)

        return np.array(samples)

    # TODO: implement with less time complexity
    def sample_roulette(self, prob_vec):
        dic = {}

        for i in range(len(prob_vec)): dic[prob_vec[i]] = i

        sorted_prob_vec = np.sort(prob_vec)
        cum_sum_vec = np.cumsum(sorted_prob_vec) # Sort vector and make cumulative sum
        rnd_num = np.random.uniform() # Sample random number from uniform distribution on (0,1)

        for i in range(len(cum_sum_vec)):
            if cum_sum_vec[i] >= rnd_num:
                return dic[sorted_prob_vec[i]]

    def vec_to_softmax(self):
        l = []

        for exmp in self.training_set:
            softmax_exmp = []

            for i in range(self.n_visible):
                kakogod = np.zeros(self.softmax_size)
                kakogod[int(exmp[i])] = 1
                softmax_exmp.append(kakogod)

            l.append(np.array(softmax_exmp).T)

        self.training_set = np.array(l)

    def softmax_to_vec(self, soft_sample):
        vec_sample = []
        kakogod = np.arange(self.softmax_size)
        kakogod.shape = (self.softmax_size, 1)

        for exmp in soft_sample:
            vec_sample.append(np.sum(exmp * kakogod, 0))

        return np.array(vec_sample)

    def plot_hidden_units_activation_probabilities(self, mini_batch, counter):
        kakogod = []

        for exmp in mini_batch:
            kakogod.append(self.activate_hidden(exmp, True))

        hMean = np.array(kakogod)
        img = Image.fromarray(hMean * 256)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save("debbuging/h_activation/h-prob" + str(counter) + ".png", format="png")

    def plot_hist(self, counter, w_update, h_update, v_update):
        # Weights
        plt.subplot(231)
        plt.hist(self.weights.flatten())
        plt.title('mm = %g' % np.mean(np.fabs(self.weights.flatten())))

        # Hidden biases
        plt.subplot(232)
        plt.hist(self.h_biases.flatten())
        plt.title('mm = %g' % np.mean(np.fabs(self.h_biases.flatten())))

        # Visible biases
        plt.subplot(233)
        plt.hist(self.v_biases.flatten())
        plt.title('mm = %g' % np.mean(np.fabs(self.v_biases.flatten())))

        # Weights update
        plt.subplot(234)
        plt.hist(w_update.flatten())
        plt.title('mm = %g' % np.mean(np.fabs(w_update.flatten())))

        # Hidden biases update
        plt.subplot(235)
        plt.hist(h_update.flatten())
        plt.title('mm = %g' % np.mean(np.fabs(h_update.flatten())))

        # Visible biases update
        plt.subplot(236)
        plt.hist(v_update.flatten())
        plt.title('mm = %g' % np.mean(np.fabs(v_update.flatten())))

        plt.savefig("debbuging/histograms/histogram" + str(counter) + ".png")
        plt.clf()

    def plot_rec_err(self):
        plt.plot(self.rec_error_arr)
        plt.title("Reconstruction error")
        plt.savefig("debbuging/reconstruction_err/rec_err.png")
        plt.clf()

    def plot_sample(self, sample):
        sample.shape = (28, 28)
        img = Image.fromarray(sample * 256)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save("debbuging/samples/sample" + str(int(time.time())) + ".png", format="png")
        sample.shape = (self.n_visible,)

    def plot_filters(self, counter):
        filters = None
        for col in self.weights.T:
            cpy_col = np.copy(col)
            cpy_col.shape = (28, 28)
            if filters is None:
                filters = cpy_col
            else:
                filters = np.concatenate((filters, cpy_col),1)

        minn = filters.min()
        maxx = filters.max()
        filters -= minn
        filters *= 256 / (maxx +1e-6)
        img = Image.fromarray(filters)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save("debbuging/filters/filters" + str(counter) + ".png", format="png")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
