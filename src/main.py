from os import listdir
from os.path import isfile, join
from witness import WitnessState
import genetic_algo as ga
import os
def main():   


    problems_dir = 'problems/puzzles_3x3'
    training_puzzle_files = [] # Make sure path is correct
    file_100 = '3x3_100'
    full_path_100 = os.path.join(problems_dir, file_100)
    if os.path.isfile(full_path_100):
        training_puzzle_files.append(full_path_100)
    else:
        print(f"Warning: Training file not found and skipped: {full_path_100}")

    for i in range(1000, 10030): # range(start, stop) goes up to stop-1
        filename = f'3x3_{i}'
        full_path = os.path.join(problems_dir, filename)
        # Optional but recommended: Check if the file exists before adding
        if os.path.isfile(full_path):
            training_puzzle_files.append(full_path)
        else:
           # print(f"Warning: Training file not found and skipped: {full_path}")
           pass

    print(f"Generated list of {len(training_puzzle_files)} training puzzle files.")
    evolved_weights = ga.evolve_heuristic_weights(training_puzzle_files)
    print("Done evolving," ,evolved_weights)


if __name__ == "__main__":
    main()
    
    
