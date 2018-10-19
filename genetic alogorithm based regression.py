# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:40:11 2018

@author: pulki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import floor
from random import random, sample ,choice
from tqdm import tqdm
from numpy.linalg import pinv
from numpy import array, dot, mean



def multiple_linear_regression(inputs, outputs):
    X, Y = np.array(inputs), np.array(outputs)
    X_t, Y_t = X.transpose(), Y.transpose()
    coeff = np.dot((pinv((np.dot(X_t, X)))), (np.dot(X_t, Y)))
    Y_p = np.dot(X, coeff)
    Y_mean = np.mean(Y)
    SST = np.array([(i - Y_mean) ** 2 for i in Y]).sum()
    SSR = np.array([(i - j) ** 2 for i, j in zip(Y, Y_p)]).sum()
    COD = (1 - (SSR / SST)) * 100.0
    av_error = (SSR / len(Y))
    return {'COD': COD, 'coeff': coeff, 'error': av_error}


def check_termination_condition(best_individual):
    if ((best_individual['COD'] >= 96.0)
            or (generation_count == max_generations)):
        return True
    else:
        return False
    
    
def create_individual(individual_size):
    return [random() for i in range(individual_size)]   


def create_population(individual_size, population_size):
    return [create_individual(individual_size) for i in range(population_size)]

def get_fitness(individual, inputs ,outputs):
    predicted_outputs = dot(array(inputs), array(individual))
    output_mean = mean(outputs)
    SST = array([(i - output_mean) ** 2 for i in outputs]).sum()
    SSR = array([(i - j) ** 2 for i, j in zip(outputs, predicted_outputs)]).sum()
    COD = (1 - (SSR / SST)) * 100.0
    average_error = (SSR / len(outputs))
    return {'COD': COD, 'error': average_error, 'coeff': individual}


def evaluate_population(population , inputs , outputs):
    fitness_list = [get_fitness(individual, inputs , outputs)
                    for individual in tqdm(population)]
    error_list = sorted(fitness_list, key=lambda i: i['error'])
    best_individuals = error_list[: selection_size]
    best_individuals_stash.append(best_individuals[0]['coeff'])
    print('Error: ', best_individuals[0]['error'],
          'COD: ', best_individuals[0]['COD'])
    return best_individuals



def crossover(parent_1, parent_2):
    child = {}
    loci = [i for i in range(0, individual_size)]
    loci_1 = sample(loci, floor(0.5*(individual_size)))
    loci_2 = [i for i in loci if i not in loci_1]
    chromosome_1 = [[i, parent_1['coeff'][i]] for i in loci_1]
    chromosome_2 = [[i, parent_2['coeff'][i]] for i in loci_2]
    child.update({key: value for (key, value) in chromosome_1})
    child.update({key: value for (key, value) in chromosome_2})
    return [child[i] for i in loci]


def mutate(individual):
    loci = [i for i in range(0, individual_size)]
    no_of_genes_mutated = floor(probability_of_gene_mutating*individual_size)
    loci_to_mutate = sample(loci, no_of_genes_mutated)
    for locus in loci_to_mutate:
        gene_transform = choice([-1, 1])
        change = gene_transform*random()
        individual[locus] = individual[locus] + change
    return individual

def get_new_generation(selected_individuals):
    parent_pairs = [sample(selected_individuals, 2)
                    for i in range(population_size)]
    offspring = [crossover(pair[0], pair[1]) for pair in parent_pairs]
    offspring_indices = [i for i in range(population_size)]
    offspring_to_mutate = sample(
        offspring_indices,
        floor(probability_of_individual_mutating*population_size)
    )
    mutated_offspring = [[i, mutate(offspring[i])]
                         for i in offspring_to_mutate]
    for child in mutated_offspring:
        offspring[child[0]] = child[1]
    return offspring

dataset = pd.read_csv('company_profit_data.csv')
dataset.drop('State' , inplace  =True , axis = 1)

y = dataset.iloc[: , -1:]
dataset.drop('Profit' , inplace = True , axis =1)

from sklearn.preprocessing import StandardScaler
normalizer_x = StandardScaler()
dataset = normalizer_x.fit_transform(dataset)
normalizer_y = StandardScaler()
y = normalizer_y.fit_transform(y)



from sklearn.model_selection import train_test_split
inputs , x_test , outputs , y_test = train_test_split(dataset , y , test_size = 0.2)

print(multiple_linear_regression(inputs , outputs))

individual_size = len(inputs[0])
population_size = 1000
selection_size = floor(0.1*population_size)
max_generations = 50
probability_of_individual_mutating = 0.1
probability_of_gene_mutating = 0.25
best_possible = multiple_linear_regression(inputs, outputs)
best_individuals_stash = [create_individual(individual_size)]
initial_population = create_population(individual_size, 1000)
current_population = initial_population
termination = False
generation_count = 0

while termination is False:
    current_best_individual = get_fitness(best_individuals_stash[-1], inputs ,outputs)
    print('Generation: ', generation_count)
    best_individuals = evaluate_population(current_population , inputs , outputs)
    current_population = get_new_generation(best_individuals)
    termination = check_termination_condition(current_best_individual)
    generation_count += 1

else:
    print(get_fitness(best_individuals_stash[-1], inputs , outputs))

best = get_fitness(best_individuals_stash[-1], inputs , outputs)
weight = np.array(best['coeff'])
weight = weight.reshape((3,1))

y_pred = x_test @ weight
y_test = normalizer_y.inverse_transform(y_test)
y_pred = normalizer_y.inverse_transform(y_pred)