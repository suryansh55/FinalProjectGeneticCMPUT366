from witness import WitnessState
import random
import os
import copy

population_size = 100
max_generations = 500
mutation_rate = 0.02
crossover_rate = 0.8
elitism_count = 2
max_chromosome_length = 50

score_success = 100.00
score_reached_fail_sep = 10.00
score_stuck = 0.00
score_limit = 0.00
penalty_per_step = 0.01

def generate_random_chromosome(max_length):
    "This function generates a list of random moves (0=U , 1=D, 2=R, 3=L)"

    length = random.randint(1, max_length)
    chromosome = [random.randint(0, 3) for _ in range(length)]
    return chromosome

def generate_initial_population(size, max_length):
    population = []
    for i in range(0, size):
        population.append(generate_random_chromosome(max_length))
    return population

def  calculate_fitness(chromosome, initial_state):
    
    current_state = initial_state.copy()
    path_valid = True

    for each_move in chromosome:

        valid_moves = current_state.successors()
        if each_move in valid_moves:
            current_state.apply_action(each_move)
        
        else:
            path_valid = False
            break

    if not path_valid:

        return 0

    if current_state.is_solution():

        fitness = 10000 - len(chromosome)

        return fitness
        
    elif current_state.has_tip_reached_goal():

        fitness = 1000

        return fitness
    else:

        distance_to_goal = current_state.heuristic_value()
        max_dist = initial_state._columns + initial_state._lines
        fitness = max(1, 100 * (1 - distance_to_goal/max_dist))
        return int(fitness)
    

def select_parents_tournament(population, fitness_scores, tournament_size=3):
    
    
    selected_parents = []
    for i in (0,1):
        contender_indices = random.sample(range(len(population)),tournament_size)
        best_contender_index = -1
        best_fitness = -1
        for index in contender_indices:
            if fitness_scores[index] > best_fitness:
                best_contender_index = index
        
        selected_parents.append(population[best_contender_index])
    return selected_parents[0], selected_parents[1]

def crossover_single_point(parent1, parent2):
    
    
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1, parent2
    
    point = random.randint(1, min(len(parent1), len(parent2))-1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]

    return offspring1,offspring2

def mutate_chromosome(chromosome, mutation_rate):
    
    
    mutate_chromosome = list(chromosome)
    for i in range(0, len(mutate_chromosome)):
        if random.random() < mutation_rate:
            mutate_chromosome[i] = random.randint(0,3)

    return mutate_chromosome

def run_genetic_algorithm(puzzle_file_path):
    
    initial_state = WitnessState()
    initial_state.read_state(puzzle_file_path)
    print(f"Solving puzzle: {puzzle_file_path}")

    population = generate_initial_population(population_size, max_chromosome_length)

    for generation in range(0,max_generations):

        fitness_scores = [ calculate_fitness(chromo, initial_state) for chromo in population]

        pop_with_fitness = list(zip(population, fitness_scores))

        pop_with_fitness.sort(key=lambda item: item[1], reverse=True)

        best_chromosome, best_fitness = pop_with_fitness[0]

        #print(best_fitness)

        if best_fitness >= 10000 - max_chromosome_length:
            print(f"Solution found in generation {generation}!")
            return best_chromosome
        
        new_population = []

        for i in range(elitism_count):

            new_population.append(pop_with_fitness[i][0])

        while len(new_population) < population_size:
            parent1, parent2 = select_parents_tournament(population, fitness_scores)


            offspring1, offspring2 = parent1, parent2
            if random.random() < crossover_rate:
                offspring1, offspring2 = crossover_single_point(parent1, parent2)


            offspring1 = mutate_chromosome(offspring1, mutation_rate)
            offspring2 = mutate_chromosome(offspring2, mutation_rate)

            new_population.append(offspring1)
            if len(new_population) < population_size:
                new_population.append(offspring2)

        population = new_population

        if generation % 20 == 0:
            print(f"Generation {generation}, Best Fitness: {best_fitness}")

    print("Maximum generations reached. No solution found.")
    best_chromosome, best_fitness = pop_with_fitness[0]
    print(f"Best fitness achieved: {best_fitness}")
    return best_chromosome


def calculate_weights_fitness(weights, training_puzzle_files, solver_step_limit=50):

    total_score = 0.00
    processed_puzzled_count = 0

    if not training_puzzle_files:
        return 0.00
    for puzzle_file in training_puzzle_files:

        try:
            initial_state = WitnessState()
            initial_state.read_state(puzzle_file)

            outcome, path_length = solve_with_heuristic(initial_state, weights, step_limit=solver_step_limit)
            puzzle_score = 0.00
            if outcome == "Success":
                puzzle_score = score_success

                puzzle_score -= path_length * penalty_per_step

                puzzle_score = max(0, puzzle_score)

            elif outcome == "Reached Goal, Failed Separation":
                puzzle_score = score_reached_fail_sep
                puzzle_score -= path_length * penalty_per_step * 0.1

            elif outcome == "Stuck":
                puzzle_score =score_stuck

            elif outcome == "Step Limit Reached":
                puzzle_score = score_limit

            total_score += puzzle_score
            processed_puzzled_count += 1

        except FileNotFoundError:
            print(f"Warning: Training puzzle file not found: {puzzle_file}, skipping.")
        except Exception as e:
        # Catch other potential errors during loading or solving
            print(f"Warning: Error processing puzzle {puzzle_file}: {e}, skipping.")

    if processed_puzzled_count > 0:
        average_score = total_score / processed_puzzled_count
        return average_score
    else:
        return 0.00

def solve_with_heuristic(initial_state, weights, step_limit=100):


    if not isinstance(weights, (list, tuple)) or len(weights) != 3:
        raise ValueError("Weights must be a list or tuple of 3 numbers [w1, w2, w3]")
    w1, w2, w3 = weights

    current_state = initial_state.copy()
    current_path_length = 0
    visited_nodes = set()

    visited_nodes.add((current_state._line_tip, current_state._column_tip))

    max_dist = initial_state._lines + initial_state._columns

    if max_dist == 0:
        max_dist = 1

    for _ in range(step_limit):

        if current_state.has_tip_reached_goal():
            if current_state.is_solution():
                return "Success" , current_path_length
            
            else:
                return "Reached Goal, Failed Separation", current_path_length
            
        possible_moves = current_state.successors()

        if not possible_moves:
            return "Stuck", current_path_length
        
        best_move = -1

        best_heuristic_value = -float('inf')
        current_distance_to_goal = current_state.heuristic_value()

        for move in possible_moves:
            next_state_candidate = current_state.copy()
            next_state_candidate.apply_action(move)
            next_node = (next_state_candidate._line_tip, next_state_candidate._column_tip)

            if next_node in visited_nodes:
                continue

            next_distance_to_goal = next_state_candidate.heuristic_value()

            f1 = -next_distance_to_goal / max_dist

            f2 = 0.0

            if next_distance_to_goal < current_distance_to_goal:
                f2 = 1.0

            elif next_distance_to_goal > current_distance_to_goal:
                f2 = -1.0

            f3 = - (current_path_length + 1.0)

            h_value = (w1 * f1) + (w2 * f2) + (w3 * f3)

            if h_value > best_heuristic_value:
                best_heuristic_value = h_value
                best_move = move

        if best_move == -1:
            return "Stuck", current_path_length
        
        current_state.apply_action(best_move)
        current_path_length += 1
        visited_nodes.add((current_state._line_tip, current_state._column_tip))

    return "Step Limit Reached", current_path_length


