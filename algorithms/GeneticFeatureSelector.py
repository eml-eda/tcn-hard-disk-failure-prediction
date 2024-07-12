# import pandas as pd
import modin.pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
from tqdm import tqdm


class GeneticFeatureSelector:
    def __init__(self, X, y, n_population=10, n_generation=2):
        """
        Initialize the GAFeatureSelection class.

        Parameters:
        - X: The input features.
        - y: The target variable.
        - n_population: The number of individuals in each generation (default: 10).
        - n_generation: The number of generations (default: 2).
        """
        self.X = X
        self.y = y
        self.n_population = n_population
        self.n_generation = n_generation
        self.hof = None
        self.toolbox = base.Toolbox()
        self._initialize_deap()

    def _initialize_deap(self):
        """
        Initializes the DEAP framework for genetic algorithm feature selection.

        This method sets up the necessary components of the DEAP framework, including the fitness function,
        individual representation, and the toolbox with various genetic operators.

        Returns:
            None
        """
        def biased_random():
            return 1 if random.random() < 0.75 else 0

        # A positive weight indicates that the genetic algorithm should try to maximize that objective,
        # while a negative weight indicates that it should try to minimize that objective.
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # Fitness function
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self.toolbox.register("attr_bool", biased_random)
        # Initialize the individual with random binary values
        self.toolbox.register("individual", tools.initRepeat,
                              creator.Individual, self.toolbox.attr_bool, len(self.X.columns))
        # Initialize the population with individuals
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register("evaluate", self.get_fitness)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def avg(self, l):
        """
        Returns the average between list elements.

        Parameters:
        l (list): A list of numeric values.

        Returns:
        float: The average of the list elements.
        """
        return sum(l) / float(len(l))

    def get_fitness(self, individual):
        """
        Calculate the fitness of a feature subset.

        Parameters:
        - individual (list): Binary representation of the feature subset.

        Returns:
        - fitness (tuple): Tuple containing the fitness value.

        The fitness is calculated based on the following steps:
        1. If the individual contains at least one feature (value 1), the following steps are performed:
            - Identify the indices of features with value 0.
            - Remove the corresponding columns from the input dataset.
            - Convert the remaining dataset to a one-hot encoded format.
            - Apply logistic regression as the classification algorithm.
            - Calculate the average cross-validation score using 5-fold cross-validation.
            - Return the fitness value as a tuple containing the average cross-validation score.
        2. If the individual contains no features (all values are 0), the fitness value is 0.

        Note: This method assumes that the input dataset (self.X) and the target variable (self.y) are already defined.

        """
        if individual.count(0) != len(individual):
            # Get index with value 0
            cols = [index for index in range(len(individual)) if individual[index] == 0]

            # Get features subset
            X_parsed = self.X.drop(self.X.columns[cols], axis=1)
            X_subset = pd.get_dummies(X_parsed)

            # Apply classification algorithm
            clf = LogisticRegression()

            # Calculate the average cross-validation score
            return (self.avg(cross_val_score(clf, X_subset, self.y, cv=5)),)
        else:
            return (0,)

    def run_genetic_algorithm(self):
        """Run the genetic algorithm.

        This method initializes and runs the genetic algorithm for feature selection.
        It creates a population, defines statistics, and performs the evolutionary process.
        The best individuals are stored in the Hall of Fame.

        Returns:
            list: The best individuals found by the genetic algorithm.
        """
        pop = self.toolbox.population(n=self.n_population)
        self.hof = tools.HallOfFame(self.n_population * self.n_generation)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2,
                                       ngen=self.n_generation, stats=stats, halloffame=self.hof,
                                       verbose=True)
        return self.hof

    def best_individual(self):
        """Get the best individual.

        Returns the fitness values, the best individual, and the best features selected.

        Returns:
            tuple: A tuple containing the fitness values, the best individual, and the best features selected.
        """
        max_accuracy = 0.0
        best_individual = None
        for individual in tqdm(self.hof, desc="Processing individuals"):
            if individual.fitness.values[0] > max_accuracy:
                max_accuracy = individual.fitness.values
                best_individual = individual

        best_features = [list(self.X)[i] for i in range(len(best_individual)) if best_individual[i] == 1]
        return best_individual.fitness.values, best_individual, best_features


# if __name__ == '__main__':
#     # Manually pass the parameters
#     dataframe_path = 'your_dataframe.csv'  # Replace with your CSV file path
#     n_pop = 10
#     n_gen = 2

#     # Read dataframe from CSV
#     df = pd.read_csv(dataframe_path, sep=',')

#     # Encode labels column to numbers
#     le = LabelEncoder()
#     # Identify the most relevant features that distinguish between the mutation points.
#     le.fit(df['predict_val'])
#     y = le.transform(df['predict_val'])
#     X = df.drop(columns=['predict_val'])

#     # Create an instance of the GeneticFeatureSelector class
#     selector = GeneticFeatureSelector(X, y, n_population=n_pop, n_generation=n_gen)

#     # Get accuracy with all features
#     individual = [1 for i in range(len(X.columns))]
#     print("Accuracy with all features: \t" + str(selector.get_fitness(individual)) + "\n")

#     # Apply genetic algorithm
#     hof = selector.run_genetic_algorithm()

#     # Select the best individual
#     accuracy, individual, header = selector.best_individual()
#     print('Best Accuracy: \t' + str(accuracy))
#     print('Number of Features in Subset: \t' + str(individual.count(1)))
#     print('Individual: \t\t' + str(individual))
#     print('Feature Subset: ' + str(header))

#     print('\n\nCreating a new classifier with the result')

#     # Read dataframe from CSV one more time
#     df = pd.read_csv(dataframe_path, sep=',')

#     # With feature subset
#     X = df[header]

#     clf = LogisticRegression()
#     scores = cross_val_score(clf, X, y, cv=5)
#     print("Accuracy with Feature Subset: \t" + str(selector.avg(scores)) + "\n")
