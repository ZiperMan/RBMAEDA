import softmax_rbm
import random

class Individual:
    def __init__(self, state=None, fitness=0):
        if state is None:
            self.state = []
        else:
            self.state = []

            for i in range(len(state)):
                self.state.append(state[i])

        self.fitness = fitness

    def print_me(self):
        kakogod = ""
        
        for i in range(len(self.state)):
            kakogod += str(self.state[i]) + " "

        kakogod += "Fitness: " + str(self.fitness)
        print(kakogod)

class RBMAEDA:
    def __init__(self, NUM_ITERATION=10000, POP_SIZE=100, DOMINANT_SUB_SIZE=80, SPACE_DIMENSION=5, SPACE_SIZE=4):
        self.NUM_ITERATION = NUM_ITERATION
        self.POP_SIZE = POP_SIZE
        self.DOMINANT_SUB_SIZE = DOMINANT_SUB_SIZE
        self.POPULATION = []
        self.SPACE_DIMENSION = SPACE_DIMENSION
        self.SPACE_SIZE = SPACE_SIZE
        self.SFMAX_RBM = softmax_rbm.SoftmaxRBM(self.SPACE_DIMENSION, 2 * self.SPACE_DIMENSION, self.SPACE_SIZE)

    # Slucajno generisanje pocetne populacije iz prostora
    def generate_initial_pop(self):

        for i in range(self.POP_SIZE):
            state = []

            for j in range(self.SPACE_DIMENSION):
                state.append(random.randint(0, self.SPACE_SIZE - 1))

            self.POPULATION.append(Individual(state, self.fitness_f(state)))
            # self.POPULATION = sorted(self.POPULATION, key=lambda ind: ind.fitness)

    # F1 sphere
    def sphere_f(self, x):
        sum = 0

        for i in range(len(x)):
            sum += x[i] * x[i]

        return sum

    # Fitness function
    def fitness_f(self, x):
        return self.sphere_f(x)

    # Surogat model, koji koristimo kao procjenu fitnes funkcije
    def surrogate_model(self, x):
        x_energy = self.SFMAX_RBM.energy_f(x)
        min_energy = None
        sum_energy = 0.0

        for ind in self.POPULATION:
            curr_energy = self.SFMAX_RBM.energy_f(ind.state)
            sum_energy += curr_energy

            if min_energy is None or curr_energy < min_energy:
                min_energy = curr_energy

        return - (x_energy - min_energy) / sum_energy

    # Selekcija dominantnog skupa pomocu Stochastic Acceptance metode. Ne zahtjeva sortiranu popolaciju
    def form_dominant_subset(self):
        dominant_subset = []
        pop_fitnes = 0

        for ind in self.POPULATION:
            pop_fitnes += ind.fitness

        while len(dominant_subset) != self.DOMINANT_SUB_SIZE:
            i = random.randint(0, self.POP_SIZE - 1) # Slucajno biramo jedinku
            i_prob = self.POPULATION[i].fitness / pop_fitnes # Racunamo njen dio u totalnom fitnesu populacije 
            rand_num = random.uniform(0, 1) # Broj odabran uniformno sa (0,1)

            # ind se dodaje u dominantni skup sa vjerovatnocom proporcijalnom ind.fitness
            if rand_num < i_prob:
                dominant_subset.append(self.POPULATION[i])

        return dominant_subset

    # Formiranje nove populacije na osnovu dominantnog skupa i sample-ovanih jedinki
    def form_new_pop(self, dominant_subset, sampled_set):
        self.POPULATION = dominant_subset + sampled_set

    def run(self):
        self.generate_initial_pop() # Inicijalizacija pocetne populacije

        for i in range(self.NUM_ITERATION):
            dominant_subset = self.form_dominant_subset() # Formiranje dominantnog podskupa
            
            # U prvom prolazu postavljamo biase vidljivog sloja na osnovu dominantnog skupa
            if i == 0:
                self.SFMAX_RBM.set_visible_biases(dominant_subset)

            # Treniranje RBM na osnovu dominantnog skupa
            self.SFMAX_RBM.cd_n(dominant_subset)

            # Samplujemo RBM za novih POP_SIZE - DOMINANT_SUB_SIZE jedinki
            sampled_set = []

            for i in range(self.POP_SIZE - self.DOMINANT_SUB_SIZE):
                sampled_set.append(self.SFMAX_RBM.sample_me())

            # Formiramo novu populaciju na osnovu dominant set-a i sample-ovanih jedniki
            self.form_new_pop(dominant_subset, sampled_set)

    def print_population(self):
        for ind in self.POPULATION:
            ind.print_me()

    def print_ind_list(self, x):
        for ind in x:
            ind.print_me()