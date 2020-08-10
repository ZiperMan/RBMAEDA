import units
import numpy
import math
import random

class SoftmaxRBM:

    def __init__(self, v_num, h_num, sfmax_size):

        # Inicijalizacija vidljivog sloja
        self.v_num = v_num
        self.v_units = []
        self.sfmax_size = sfmax_size

        for i in range(self.v_num):
            self.v_units.append(units.SoftmaxUnit(self.sfmax_size))

        # Inicijalizacija skrivenog sloja
        self.h_num = h_num
        self.h_units = []

        for i in range(self.h_num):
            self.h_units.append(units.StochasticBinaryUnit())

        # Inicijalizacija matrica tezina
        self.w_matrices = []

        for i in range(self.sfmax_size):
            w_matrix = []

            for j in range(self.v_num):
                col = []

                for k in range(self.h_num):
                    col.append(numpy.random.normal(0.0, 0.01))

                w_matrix.append(col)

            self.w_matrices.append(w_matrix)

        # Inicijalizacija learning rate-a
        self.learning_rate = 0.9
    
    # Postavljanje bias-a za vidljivi sloj na osnovu training seta
    def set_visible_biases(self, training_data):
        prop_matrix = []

        for i in range(self.v_num):
            col = []

            for j in range(self.sfmax_size):
                col.append(0.0)

            prop_matrix.append(col)

        for i in range(len(training_data)):
            for j in range(self.v_num):
                prop_matrix[j][training_data[i][j]] += (1.0 / len(training_data))

        for i in range(len(prop_matrix)):
            for j in range(len(prop_matrix[i])):
                if prop_matrix[i][j] != 0:
                    self.v_units[i].units[j].bias = math.log2(prop_matrix[i][j] / (1 - prop_matrix[i][j]))

    # Postavlja vrijednost vidljivog sloja na zadati vektor
    def set_v_units(self, v_units):

        for i in range(self.v_num):
            for j in range(self.sfmax_size):
                self.v_units[i].units[j].value = 0
        
        for i in range(self.v_num):
            self.v_units[i].units[v_units[i]].value = 1

    # Aktivira vidljivi sloj na osnovu skrivenog
    def v_activation(self):
        for i in range(self.v_num):
            self.v_units[i].activate(self.h_units, self.w_matrices, i)

    # Aktivira skriveni sloj na osnovu vidljivog
    def h_activation(self):
        for i in range(self.h_num):
            h_unit = self.h_units[i]
            excitation = 0

            for j in range(self.v_num):
                for k in range(self.sfmax_size):
                    excitation += self.v_units[j].units[k].value * self.w_matrices[k][j][i]

            h_unit.activate_log(excitation)

    # Energija datog vidljivog i odgovarajuceg skrivenog sloja
    def energy_f(self, v_units):
        self.set_v_units(v_units) # Postavljamo vidljivi sloj na zadati vektor
        self.h_activation()# Racunamo nevidljivi sloj na osnovu vidljivog

        # Energija dobijene konfiguracije
        energy = 0
        sum = 0

        for i in range(self.v_num):
            for j in range(self.h_num):
                for k in range(self.sfmax_size):
                    sum += self.h_units[j].value * self.v_units[i].units[k].value * self.w_matrices[k][i][j]

        energy -= sum
        sum = 0

        for i in range(self.v_num):
            for k in range(self.sfmax_size):
                sum += self.v_units[i].units[k].value * self.v_units[i].units[k].bias

        energy -= sum
        sum = 0

        for i in range(self.h_num):
            sum += self.h_units[i].value * self.h_units[i].bias

        energy -= sum
        sum = 0

        return energy

    # Pomocna funkcija za cd_n
    def cd_n_calc_expectation_weights(self, data, v_indx, v_sm_indx, h_indx):
        sum = 0.0

        for x in data:
            self.set_v_units(x)
            self.h_activation()
            sum += self.v_units[v_indx].units[v_sm_indx].value * self.h_units[h_indx].value

        return sum / len(data)

    def cd_n_calc_expectation_v_biases(self, data, v_indx, v_sm_indx):
        pass

    def cd_n_calc_expectation_h_biases(self, data, h_indx):
        pass

    # Contrastive Divergence learning
    def cd_n(self, training_data, n=1):

        # Napravimo rekonstrukcije training set-a prema trenutnoj topologiji mreze
        reconstructions = []

        for x in training_data:
            self.set_v_units(x) # Postavljamo vidljivi sloj na vektor iz training set-a
            self.h_activation() # Racunamo skriveni sloj na osnovu tog vektora
            self.v_activation() # Racunamo rekonstrukciju na osnovu izracunatog vidljivog sloja
            reconstructions.append(list.copy(self.v_units)) # Dodajemo rekonstrukciju

        # Prolazimo kroz sve grane svih tezinskih matrica i update-ujemo ih
        for k in range(self.sfmax_size):
            for i in range(self.v_num):
                for j in range(self.h_num):
                    self.w_matrices[k][i][j] += self.learning_rate * 
                    (self.cd_n_calc_expectation_weights(training_data, i, k, j) - 
                    self.cd_n_calc_expectation_weights(reconstructions, i, k, j))

        # Prolazimo kroz sve biase vidljivog sloja i update-ujemo ih
        for k in range(self.sfmax_size):
            for i in range(self.v_num):
                pass

        # Prolazimo krzo sve biase skrivenog sloja i update-ujemo ih
        for j in range(self.h_num):
            pass

    # Racuna vjerovatnosni model tj. vjerovatnocu aktivacije svake binarne jedinice u vidljivom sloju
    # Na osnovu data seta prema kojem je RBM trenirana
    def calc_prob_model(self, training_data):
        self.prob_model = []

        # Inicijalizacija matrice
        for i in range(self.v_num):
            col = []

            for j in range(self.sfmax_size):
                col.append(0.0)

            self.prob_model.append(col)

        training_len = len(training_data)

        # Racunanje i sabiranje vjerovatnoca za svaki vektor iz training_data
        for training_vec in training_data:
            self.set_v_units(training_vec)
            self.h_activation()

            # print("V UNITS: ")
            # self.print_v_units()
            # self.print_h_units()
            # print("V BIASES: ")
            # self.print_v_biases()
            # print("W MATRICES: ")
            # self.print_w_matrices()

            for i in range(self.v_num):
                probs = self.v_units[i].get_activation_probs(self.h_units, self.w_matrices, i)

                kakogod = "Probs: "
                for j in range(self.sfmax_size):
                    kakogod += str(probs[j]) + " "
                    self.prob_model[i][j] += probs[j] / training_len
                # print(kakogod)
                # return

    # Binarno trazenje
    def bin_srch(self, arr, l, r, x):
        m = (l + r) / 2

        if m == 0:
            if x <= arr[0]:
                return 0
        elif arr[m - 1] < x:

            if arr[m] >= x:
                return m
            else:
                return self.bin_srch(arr, m + 1, r, x)
        else:
            return self.bin_srch(arr, l, m, x)

    # Sample-uje vektor u skladu sa raspodjelom koju aproksimira RBM pomocu roulette tehnike i binarnog trazenja
    def sample_me_roulette(self):
        new_ind = []

        for i in range(self.v_num):
            cum_sum = []

            for j in range(self.sfmax_size):

                if j == 0:
                    cum_sum.append(self.prob_model[i][j])
                else:
                    cum_sum.append(cum_sum[j - 1] + self.prob_model[i][j])

            new_ind.append(self.bin_srch(cum_sum, random.uniform(0, 1)) + 1)

        return new_ind

    def print_w_matrices(self):
        print("Ukupno matrica: " + str(len(self.w_matrices)))

        for i in range(len(self.w_matrices)):
            print("Matrica " + str(i) + ":")

            for j in range(len(self.w_matrices[i])):
                kakogod = ""

                for k in range(len(self.w_matrices[i][j])):
                    kakogod += str(self.w_matrices[i][j][k]) + "  "

                print(kakogod)

    def print_v_biases(self):
        for i in range(len(self.v_units)):
            kakogod = ""

            for j in range(len(self.v_units[i].units)):
                kakogod += str(self.v_units[i].units[j].bias) + "  "

            print(kakogod)

    def print_v_units(self):
        for i in range(len(self.v_units)):
            kakogod = "Softmax unit " + str(i) + ": "

            for j in range(len(self.v_units[i].units)):
                kakogod += str(self.v_units[i].units[j].value) + " "

            print(kakogod)

    def print_h_units(self):
        kakogod = "Hidden layer: "
        
        for i in range(self.h_num):
            kakogod += str(self.h_units[i].value) + " "

        print(kakogod)

def print_matrix(m):
    for i in range(len(m)):
        kakogod = ""

        for j in range(len(m[i])):
            kakogod += str(m[i][j]) + " "

        print(kakogod)