import numpy as np

class EDA:
    def __init__(self, pop_size, dominant_size, gen_model, fitness_f, dim, max_val) -> None:
        self.pop_size = pop_size
        self.dominant_size = dominant_size
        self.gen_model = gen_model
        self.fitness_f = fitness_f
        self.dim = dim
        self.max_val = max_val

    def form_pop(self, sample=None):

        # If there is no sample, randomly initialize population
        if sample is None:
            self.pop = np.random.randint(self.max_val, size=(self.pop_size, self.dim))
        else:
        # If there is sample, merge dominant set and sample to form new population
            self.pop = np.concatenate((self.dominant, sample))

        # calculate fitness for every individual in population
        self.pop_fitness = np.vectorize(self.fitness_f, signature='(n)->()')(self.pop)

    # Form dominant subset from the population using truncation selection method
    # TODO: Try some other selection method ?? roulette wheel ??
    def form_dominant_subset(self):
        self.dominant = []

        dic = {}

        for i in range(self.pop_size):
            dic[self.pop_fitness[i]] = i

        for fit in np.sort(self.pop_fitness)[-self.dominant_size:]:
            self.dominant.append(self.pop[dic[fit]])

        self.dominant = np.array(self.dominant)

    def run(self, n_iterations):
        # Get initial population
        self.form_pop()

        for i in range(n_iterations):

            # Form dominant subset
            self.form_dominant_subset()

            # print("DOMINANT SUBSET", self.dominant)
            
            # Train generative model using dominant subset
            self.gen_model.train(self.dominant,mini_batch_size=10,n_epochs=10, alg="pcd")

            # Sample from the generative model
            sample = self.gen_model.sample(self.pop_size - self.dominant_size)

            # print("SAMPLED FROM RBM", sample)

            # Form new population based on dominant set and sample from the generative model
            self.form_pop(sample)

            # print("NEW POPULATION", self.pop)

        # Return the best individual and best fitness as solution
        best_ind = None
        best_fitness = None

        for ind in self.pop:
            fitness = self.fitness_f(ind)

            if best_fitness is None or fitness > best_fitness:
                best_fitness = fitness
                best_ind = ind 

        return [best_fitness, best_ind]
