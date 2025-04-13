# Required imports
from witness import WitnessState # Make sure this is accessible
import random
import os
import copy
import time # For timing the evolution
import heapq # Priority Queue
# --- GA Parameters --- 
POPULATION_SIZE = 10
MAX_GENERATIONS = 2 # Start with fewer for testing, increase later
MUTATION_RATE = 0.1   # Probability for each weight to mutate (might need tuning)
MUTATION_STRENGTH = 0.5 # Std deviation of Gaussian noise added during mutation
CROSSOVER_RATE = 0.8    # Probability for two parents to crossover
ELITISM_COUNT = 2       # Number of best individuals to carry over
TOURNAMENT_SIZE = 3
WEIGHT_RANGE = (-5.0, 5.0) # Range for initial random weights
SOLVER_EXPANSION_LIMIT_TRAIN = 10000 # Step limit for solver during fitness eval
NUM_WEIGHTS = 7         # We have 3 features w1, w2, w3

# --- Scoring Parameters ---
SCORE_SUCCESS = 100.0
SCORE_REACHED_FAIL_SEP = 10.0
SCORE_STUCK = 0.0
SCORE_LIMIT = 0.0
PENALTY_PER_STEP = 0.01

# --- Helper Functions for WEIGHT EVOLUTION ---

def generate_initial_population_weights(size, num_weights, weight_range=(-5.0, 5.0)):
    """Generates the initial population of weight vectors."""
    population = []
    for _ in range(size): # Corrected: generates 'size' individuals
        weights = [random.uniform(weight_range[0], weight_range[1]) for _ in range(num_weights)]
        population.append(weights)
    # print(f"Generated initial population with {size} individuals.") # Optional Verbose
    return population

def crossover_weights_average(parent1, parent2):
    """Performs average crossover between two weight vectors."""
    # Ensure parents are lists/tuples of numbers of the same length
    if len(parent1) != len(parent2):
        # Handle error or return parents unchanged
        print("Warning: Parents have different lengths in crossover. Returning originals.")
        return list(parent1), list(parent2)
    offspring1 = [(p1 + p2) / 2.0 for p1, p2 in zip(parent1, parent2)]
    # Return two identical offspring (simplest form of average crossover)
    # Use deepcopy if there's any chance lists contain mutable elements, but numbers are fine.
    return offspring1, offspring1

def mutate_weights_gaussian(chromosome, mutation_rate, mutation_strength=0.5):
    """Applies Gaussian mutation to a weight vector."""
    mutated_chromosome = list(chromosome) # Work on a copy
    for i in range(len(mutated_chromosome)): # Corrected: iterates over all weights
        if random.random() < mutation_rate:
            mutation_amount = random.gauss(0, mutation_strength)
            mutated_chromosome[i] += mutation_amount
            # Optional: Clamp weights if they should stay within a specific range
            # mutated_chromosome[i] = max(WEIGHT_RANGE[0], min(mutated_chromosome[i], WEIGHT_RANGE[1]))
    return mutated_chromosome

def mutate_weights_gaussian_with_reset(chromosome, mutation_rate, mutation_strength=0.5, reset_prob=0.1, weight_range=(-5.0, 5.0)):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            if random.random() < reset_prob: # Chance to reset completely
                mutated_chromosome[i] = random.uniform(weight_range[0], weight_range[1])
            else: # Otherwise, add Gaussian noise
                mutated_chromosome[i] += random.gauss(0, mutation_strength)
    return mutated_chromosome

def select_parents_tournament(population, fitness_scores, tournament_size=3):
    """Selects two parents using tournament selection."""
    selected_parents = []
    population_size = len(population)
    if population_size == 0: return None, None # Handle empty population

    for _ in range(2): # Select two parents
        actual_tournament_size = min(tournament_size, population_size)
        if actual_tournament_size <= 0: return None, None # Should not happen if pop_size > 0

        contender_indices = random.sample(range(population_size), actual_tournament_size)

        best_contender_index_in_sample = -1
        best_fitness_in_sample = -float('inf')

        # Find the best contender *within the sample*
        for index in contender_indices:
            if fitness_scores[index] > best_fitness_in_sample:
                best_fitness_in_sample = fitness_scores[index]
                best_contender_index_in_sample = index

        if best_contender_index_in_sample != -1:
             selected_parents.append(population[best_contender_index_in_sample])
        else:
             # Fallback if something went wrong (e.g., all fitnesses were -inf?)
             print("Warning: Tournament selection failed to find a best contender. Selecting random.")
             selected_parents.append(random.choice(population))

    if len(selected_parents) == 2:
        # Return copies to prevent modification issues if lists are reused
        return list(selected_parents[0]), list(selected_parents[1])
    else:
        # Fallback if selection failed
        print("Warning: Selection failed to produce two parents. Returning random.")
        return list(random.choice(population)), list(random.choice(population))


def _get_cell_color(state, r, c):
    """Safely gets the color code from state._cells[r][c].
       Returns -1 if the coordinates (r, c) are outside the cell grid bounds.
       Assumes color codes are stored as numbers (like 0 for empty, 1+ for colors).
    """
    if 0 <= r < state._lines and 0 <= c < state._columns:
        # Ensure returned value is an integer if needed elsewhere
        return int(state._cells[r][c])
    else:
        # Indicates coordinates are off the grid where cells exist
        return -1
def calculate_segment_features(prev_line, prev_col, move, next_state):
    """
    Calculates features based on the grid segment just added by 'move'.

    Args:
        prev_line (int): Line coordinate *before* the move.
        prev_col (int): Column coordinate *before* the move.
        move (int): The move taken (0=U, 1=D, 2=R, 3=L).
        next_state (WitnessState): The state *after* the move was applied.

    Returns:
        tuple: (c1, c2, f4)
               c1, c2: Integer color codes of adjacent cells (-1 if off-grid).
               f4: Float 1.0 if the segment is on the grid edge, 0.0 otherwise.
    """
    c1, c2 = -1, -1 # Default: assume off-grid
    f4 = 0.0        # Default: assume not on edge

    # Grid dimensions from the state object
    grid_lines = next_state._lines
    grid_cols = next_state._columns

    # Aliases for previous position for clarity
    r, c = prev_line, prev_col

    # Determine adjacent cells based on the move and calculate features
    if move == 0: # Moved UP: (r, c) -> (r+1, c). Vertical segment added at v_seg[r][c]
        # Check if segment column is on the left (c=0) or right (c=grid_cols) edge
        f4 = 1.0 if c == 0 or c == grid_cols else 0.0
        # Adjacent cells are at row 'r', to the left (col c-1) and right (col c) of the segment
        c1 = _get_cell_color(next_state, r, c - 1) # Cell to the left
        c2 = _get_cell_color(next_state, r, c)     # Cell to the right
    elif move == 1: # Moved DOWN: (r, c) -> (r-1, c). Vertical segment added at v_seg[r-1][c]
        # Check if segment column is on the left (c=0) or right (c=grid_cols) edge
        f4 = 1.0 if c == 0 or c == grid_cols else 0.0
        # Adjacent cells are at row 'r-1', to the left (col c-1) and right (col c)
        c1 = _get_cell_color(next_state, r - 1, c - 1) # Cell to the left
        c2 = _get_cell_color(next_state, r - 1, c)     # Cell to the right
    elif move == 2: # Moved RIGHT: (r, c) -> (r, c+1). Horizontal segment added at h_seg[r][c]
        # Check if segment row is on the top (r=0) or bottom (r=grid_lines) edge
        f4 = 1.0 if r == 0 or r == grid_lines else 0.0
        # Adjacent cells are at column 'c', above (row r-1) and below (row r) the segment
        c1 = _get_cell_color(next_state, r - 1, c) # Cell above
        c2 = _get_cell_color(next_state, r, c)     # Cell below
    elif move == 3: # Moved LEFT: (r, c) -> (r, c-1). Horizontal segment added at h_seg[r][c-1]
        # Check if segment row is on the top (r=0) or bottom (r=grid_lines) edge
        f4 = 1.0 if r == 0 or r == grid_lines else 0.0
        # Adjacent cells are at column 'c-1', above (row r-1) and below (row r)
        c1 = _get_cell_color(next_state, r - 1, c - 1) # Cell above
        c2 = _get_cell_color(next_state, r, c - 1)     # Cell below

    return c1, c2, f4



def calculate_weights_fitness(weights, training_puzzle_files, solver_expansion_limit=1000, verbose_fitness=False):
    """Calculates fitness by averaging solver performance across training puzzles."""
    total_score = 0.0
    total_success_expansions = 0
    success_count = 0
    processed_puzzle_count = 0
    if not training_puzzle_files: return 0.0

    # Print weights being evaluated only if requested (can be noisy)
    # if verbose_fitness: print(f"  Evaluating weights {['{:.2f}'.format(w) for w in weights]}:")

    for idx, puzzle_file in enumerate(training_puzzle_files):
        puzzle_score = 0.0 # Default score if error occurs
        try:
            initial_state = WitnessState()
            initial_state.read_state(puzzle_file)
            # Set verbose_solver=False here normally, only enable for deep debugging

            outcome, path_length, _ = solve_with_A_star(initial_state, weights,
                                                       expansion_limit=solver_expansion_limit)
            
            if outcome == "Success":
                success_count += 1
                if path_length >= 0:
                    puzzle_score = SCORE_SUCCESS - (path_length * PENALTY_PER_STEP)
                    puzzle_score = max(0, puzzle_score) # Ensure non-negative
                else:
                    puzzle_score = SCORE_SUCCESS * 0.9
            elif outcome == "Reached Goal, Failed Separation":
                puzzle_score = SCORE_REACHED_FAIL_SEP
            # Implicit: Stuck/Limit get 0 score based on defaults

            # Verbose output for fitness calculation (shows per-puzzle results)
            if verbose_fitness:
                print(f"    Puzzle {idx+1} ({os.path.basename(puzzle_file)}): Outcome='{outcome}', Len={path_length}, Score={puzzle_score:.2f}")

            processed_puzzle_count += 1 # Count successful processing
            total_score += puzzle_score

        except FileNotFoundError:
            # print(f"Warning: Training puzzle file not found: {puzzle_file}, skipping.")
            pass # Or decrement total puzzle count if averaging later
        except Exception as e:
            print(f"Warning: Error processing puzzle {puzzle_file}: {e}, skipping.")

    # Calculate average score
    if processed_puzzle_count > 0:
        average_score = total_score / processed_puzzle_count
        return average_score
    else:
        print("Warning: No puzzles processed successfully in fitness calculation.")
        return 0.0


def manhattan_distance(r1, c1, r2, c2):
    """Calculates Manhattan distance between two points (r1, c1) and (r2, c2)."""
    return abs(r1 - r2) + abs(c1 - c2)

# --- Main Evolution Function ---

def evolve_heuristic_weights(training_files_list,
                             pop_size=POPULATION_SIZE,
                             generations=MAX_GENERATIONS,
                             num_weights=NUM_WEIGHTS,
                             mutation_rate=MUTATION_RATE,
                             mutation_strength=MUTATION_STRENGTH,
                             crossover_rate=CROSSOVER_RATE,
                             elitism_count=ELITISM_COUNT,
                             tournament_size=TOURNAMENT_SIZE,
                             weight_range=WEIGHT_RANGE,
                             solver_expansion_limit=SOLVER_EXPANSION_LIMIT_TRAIN,
                             verbose_every_n_gen=10):
    """
    Runs the GA to evolve heuristic weights. Now uses parameters passed in.
    """
    # Ensure elitism isn't larger than population
    actual_elitism_count = min(elitism_count, pop_size)
    if elitism_count > pop_size:
        print(f"Warning: Elitism count ({elitism_count}) > Population size ({pop_size}). Setting to {pop_size}.")
        actual_elitism_count = pop_size

    population = generate_initial_population_weights(pop_size, num_weights, weight_range)

    print("--- Starting Heuristic Weight Evolution ---")
    print(f"Population Size: {pop_size}, Generations: {generations}")
    print(f"Mutation Rate: {mutation_rate:.3f}, Mutation Strength: {mutation_strength:.3f}")
    print(f"Crossover Rate: {crossover_rate:.2f}, Elitism Count: {actual_elitism_count}")
    print(f"Tournament Size: {tournament_size}")
    print(f"Initial Weight Range: {weight_range}")
    print(f"Evaluating fitness across {len(training_files_list)} training puzzles.")
    print(f"Solver Step Limit during Training: {solver_expansion_limit}")
    print("-" * 50) # Separator

    start_time = time.time()
    best_overall_weights = None
    best_overall_fitness = -float('inf')
    fitness_history = [] # Track best fitness per generation

    for generation in range(generations):
        gen_start_time = time.time()

        # --- Evaluate Population ---
        # Generally set verbose_fitness=False here unless debugging specific generation
        fitness_scores = [calculate_weights_fitness(chromo, training_files_list, solver_expansion_limit, verbose_fitness=False)
                          for chromo in population]

        # --- Combine, Sort, Track Best ---
        pop_with_fitness = list(zip(population, fitness_scores))
        pop_with_fitness.sort(key=lambda item: item[1], reverse=True)

        # Handle potential errors if fitness calculation returned non-numbers (e.g., None)
        # This assumes fitness_scores contains valid numbers
        try:
             current_best_weights_gen, current_best_fitness_gen = pop_with_fitness[0]
             average_fitness_gen = sum(fs for fs in fitness_scores if isinstance(fs, (int, float))) / len(fitness_scores) # Robust average
             fitness_history.append(current_best_fitness_gen)
        except IndexError:
            print(f"Error: Population empty or fitness calculation failed at generation {generation}. Stopping.")
            break
        except TypeError:
            print(f"Error: Non-numerical fitness scores found at generation {generation}. Stopping.")
            # Consider printing fitness_scores here for debugging
            break


        new_best_overall_found = False
        if current_best_fitness_gen > best_overall_fitness:
            best_overall_fitness = current_best_fitness_gen
            best_overall_weights = list(current_best_weights_gen) # Store copy
            new_best_overall_found = True

        # --- Reporting ---
        # Print every N generations OR if a new overall best is found
        if generation == 0 or generation % verbose_every_n_gen == (verbose_every_n_gen - 1) or new_best_overall_found:
            elapsed_gen_time = time.time() - gen_start_time
            print(f"Gen {generation+1: <4}/{generations}: Best Fit={best_overall_fitness: <8.4f} (Avg Fit={average_fitness_gen: <8.4f}) Gen Time={elapsed_gen_time:.2f}s", end="")
            if new_best_overall_found:
                 # Format weights for slightly nicer printing
                 weights_str = ', '.join([f'{w:.3f}' for w in best_overall_weights])
                 print(f" *New Best* Weights=[{weights_str}]")
            else:
                 print() # Just newline if not new best


        # --- Create Next Generation ---
        new_population = []
        # Elitism
        for i in range(actual_elitism_count):
             new_population.append(list(pop_with_fitness[i][0])) # Add copy

        # Reproduction
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents_tournament(population, fitness_scores, tournament_size)
            if parent1 is None: # Handle edge case from selection function
                print("Warning: Parent selection failed, using random parents for fill.")
                parent1, parent2 = random.choice(population), random.choice(population)

            # Crossover
            offspring1, offspring2 = list(parent1), list(parent2) # Default to parents if no crossover
            if random.random() < crossover_rate:
                 offspring1, offspring2 = crossover_weights_average(parent1, parent2)

            # Mutation
            #offspring1 = mutate_weights_gaussian(offspring1, mutation_rate, mutation_strength)
            offspring1  = mutate_weights_gaussian_with_reset(offspring1, mutation_rate, mutation_strength)
            #offspring2 = mutate_weights_gaussian(offspring2, mutation_rate, mutation_strength)
            offspring2 = mutate_weights_gaussian_with_reset(offspring2,mutation_rate, mutation_strength)

            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)

        population = new_population # Replace population for next generation

    # --- Final Output ---
    total_evolution_time = time.time() - start_time
    print("-" * 50) # Separator
    print(f"Evolution finished after {generations} generations.")
    print(f"Total time: {total_evolution_time:.2f} seconds ({total_evolution_time/60:.2f} minutes)")
    if best_overall_weights is not None:
        print(f"Best overall fitness (Avg Score on Training Set): {best_overall_fitness:.4f}")
        weights_str = ', '.join([f'{w:.4f}' for w in best_overall_weights])
        print(f"Best weights found: [{weights_str}]")
    else:
        print("No best weights found (potential error during run).")

    # Optional: Plot fitness history using matplotlib
    # import matplotlib.pyplot as plt
    # plt.plot(fitness_history)
    # plt.title("Best Fitness per Generation")
    # plt.xlabel("Generation")
    # plt.ylabel("Best Fitness (Avg Score)")
    # plt.grid(True)
    # plt.show()

    return best_overall_weights


def calculate_A_star_heuristic(state, g_score, weights, max_dist):

    w1, w2, w3, w4, w5, w6, w7 = weights

    f1 = -state.heuristic_value() / max_dist
    f3 = -float(g_score)
    f7 = 0.0
    current_tip_line = state._line_tip
    current_tip_col = state._column_tip
    dot_locations = state._dot_locations # Assumes this exists
    nearest_dot_color = 0

    if dot_locations:
        nearest_dot_dist = float('inf')
        for dot in dot_locations:
            dist = manhattan_distance(current_tip_line, current_tip_col, dot['r'], dot['c'])
            if dist < nearest_dot_dist:
                nearest_dot_dist = dist
                nearest_dot_color = dot['color']
    wrong_dots = []
    if nearest_dot_color > 0:
        for dot in dot_locations:
            if dot['color'] != nearest_dot_color:
                wrong_dots.append(dot)

    if wrong_dots:
        min_dist_to_wrong = float('inf')
        for wrong_dot in wrong_dots:
            dist = manhattan_distance(current_tip_line, current_tip_col, wrong_dot['r'], wrong_dot['c'])
            min_dist_to_wrong = min(min_dist_to_wrong, dist)
        epsilon = 0.1
        if min_dist_to_wrong >= 0: f7 = -1.0 / (min_dist_to_wrong + epsilon)

    heuristic_weighted_sum = (w1 * f1) + (w3 * f3) + (w7 * f7)

    h = max(0.0, -heuristic_weighted_sum)

    return h

def solve_with_A_star(initial_state, weights, expansion_limit=1000):


    if not isinstance(weights, (list, tuple)) or len(weights) != NUM_WEIGHTS:
        print(f"Error: Incorrect number of weights ({len(weights)}) for A*.")
        return "Error", -1, None
    
    start_node_state = initial_state.copy()
    start_node_state._dot_locations = initial_state._dot_locations # Ensure dots are copied if not done by default

    max_dist = initial_state._lines + initial_state._columns
    if max_dist == 0: max_dist = 1.0

    start_g = 0
    start_h = calculate_A_star_heuristic(start_node_state, start_g, weights, max_dist)  #comment if base A* testing
    #start_h = start_node_state.heuristic_value() #uncomment if base A* testing
    start_f = start_g + start_h

    open_set_heap = [(start_f, start_g, hash(start_node_state), start_node_state, [])]
    heapq.heapify(open_set_heap)

    g_scores = {start_node_state: 0}

    closed_set = set()

    nodes_expanded = 0

    while open_set_heap and nodes_expanded < expansion_limit:
        nodes_expanded +=1

        current_f, current_g, _, current_state, current_path = heapq.heappop(open_set_heap)
        if current_state in closed_set:
            continue

        closed_set.add(current_state)

        if current_state.is_solution():
            print(f"  A* Success: Found solution after expanding {nodes_expanded} nodes.") # Debug
            
            return "Success", current_g, current_path
        
        possible_moves = current_state.successors()
        for move in possible_moves:
            neighbour_state = current_state.copy()
            neighbour_state.apply_action(move)
            neighbour_state._dot_locations = current_state._dot_locations

            if neighbour_state in closed_set:
                continue

            tentative_g_score = current_g + 1

            if neighbour_state in g_scores and tentative_g_score >= g_scores[neighbour_state]:
                continue

            new_path = current_path + [move]
            g_scores[neighbour_state] = tentative_g_score

            h_score = calculate_A_star_heuristic(neighbour_state, tentative_g_score, weights, max_dist)  #comment if base A* testing
            #h_score = neighbour_state.heuristic_value()  #uncomment if base A* testing
            f_score = tentative_g_score + h_score

            heapq.heappush(open_set_heap, (f_score, tentative_g_score, hash(neighbour_state), neighbour_state, new_path))
        
    if not open_set_heap:
        return "Failed", -1,nodes_expanded
    else:
        return "Limit Reached", -1, nodes_expanded
