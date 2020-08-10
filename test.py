import softmax_rbm
import rbmaeda

training_set = [
                [1, 2, 3, 3, 3],
                [0, 2, 1, 3, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 1, 0, 3],
                [2, 1, 1, 3, 2]
            ]

rbm = softmax_rbm.SoftmaxRBM(5, 3, 4)
rbm.set_visible_biases(training_set)
rbm.calc_prob_model(training_set)
# softmax_rbm.print_matrix(rbm.prob_model)