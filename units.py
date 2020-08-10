from math import exp
from random import uniform
import random

class StochasticBinaryUnit:

    def __init__(self, value=0, bias=0):
        self.value = value
        self.bias = bias

    # vjerovatnoca aktivacije racuna se preko logistic sigmoid funkcije f(x) = 1 / 1 + e^-x
    # vrijednost cvora se onda update-uje na osnovu te vjerovatnoce
    def activate_log(self, excitation):
        prob = 1.0 / (1 + exp(-(self.bias + excitation)))
        num = uniform(0, 1)
        self.value = 1 if num <= prob else 0

    # vraca samo vjerovatnocu aktivacije. Korisno kod CD ucenja
    def get_act_prob(self, excitation):
        return 1.0 / (1 + exp(-(self.bias + excitation)))

    def print_me(self):
        print("value: " + str(self.value) + " bias: " + str(self.bias))



class SoftmaxUnit:

    def __init__(self, num_of_units=0, values=None, biases=None):
        self.num_of_units = num_of_units
        self.units = []

        for i in range(self.num_of_units):
            value = 0 if values == None else values[i]
            bias = 0 if biases == None else biases[i]
            self.units.append(StochasticBinaryUnit(value, bias))

    # Vraca vjerovatnocu aktivacije za svaku binarnu jedinicu unutar date softmax jedinice
    def get_activation_probs(self, h_units, w_matrices, v_unit_indx):
        # Racunamo vjerovatnoce aktivacije za svaku binarnu jedinicu
        probs = []

        for i in range(self.num_of_units):
            excitation = self.units[i].bias
            norm = 0

            for j in range(len(h_units)):
                excitation += h_units[j].value * w_matrices[i][v_unit_indx][j]

            for k in range(self.num_of_units):
                sum = 0

                for j in range(len(h_units)):
                    sum += h_units[j].value * w_matrices[k][v_unit_indx][j]
                
                norm += exp(self.units[k].bias + sum)

            probs.append(exp(self.units[i].bias + excitation) / norm)

        return probs

    # Aktivira se samo jedna binarna jedinica u softmax-u, ostale se gase
    def activate(self, h_units, w_matrices, v_unit_indx):

        for unit in self.units:
            unit.value = 0

        probs = self.get_activation_probs(h_units, w_matrices, v_unit_indx)

        # Aktiviramo jednu od jedinica koristeci Stochastic Accaptance metod
        i = random.randint(0, self.num_of_units - 1) # Slucajno biramo jedinicu
        rand_num = random.uniform(0, 1) # Broj odabran uniformno sa (0,1)

        # jedinica se aktivira sa vjerovatnocom proporcijalnom njenoj vjerovatnoci aktivacije
        # Postupak nastavljamo sve dok se neka od jedinica ne aktivira
        while rand_num > probs[i]:
            i = random.randint(0, self.num_of_units - 1) # Slucajno biramo jedinicu
            rand_num = random.uniform(0, 1) # Broj odabran uniformno sa (0,1)
        
        self.units[i].value = 1

    def print_me(self):
        print("Num of units: " + str(self.num_of_units))

        for i in range(self.num_of_units):
            self.units[i].print_me()