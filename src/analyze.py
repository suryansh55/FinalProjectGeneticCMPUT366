# --- Imports ---
import os
import random # May be needed by WitnessState or your GA file indirectly
# Make sure these are correctly imported or defined in your script
from witness import WitnessState
import genetic_algo as ga
# from your_ga_file import solve_with_heuristic # Import the solver function

# --- Analysis Setup ---


best_evolved_weights = [4.064, -2.129, 1.240, 1.914, -0.620, 0.536, 3.200] # Pretrained

# This code should match how you created the list for the 'evolve_heuristic_weights' call
problems_dir = 'problems/puzzles_4x4'
test_files = [] # Use the same list the GA trained on
f100_path = os.path.join(problems_dir, '4x4_100')
if os.path.isfile(f100_path): test_files.append(f100_path)
for i in range(1, 1000): # Adjust range to match your files
    fp = os.path.join(problems_dir, f'4x4_{i}')
    if os.path.isfile(fp): test_files.append(fp)
# --- End of list generation ---

# Set a step limit for the analysis run (make it generous)
analysis_solver_step_limit = 10000

# --- Analysis Function ---
def analyze_heuristic_performance(weights, puzzle_file_list, step_limit):
    """
    Runs the solver with given weights on a list of puzzles and counts the outcomes.
    Returns a dictionary mapping outcome strings to counts.
    """
    # Use a dictionary to store counts for each outcome category
    outcomes = {
        "Success": 0,
        "Reached Goal, Failed Separation": 0,
        "Stuck": 0,
        "Step Limit Reached": 0,
        "Error": 0 # To count puzzles that cause exceptions
    }
    total_puzzles_in_list = len(puzzle_file_list)
    processed_count = 0 # Count how many puzzles were actually processed

    print(f"\n--- Running Analysis ---")
    print(f"Analyzing {len(weights)} weights: {['{:.3f}'.format(w) for w in weights]}")
    print(f"Using {total_puzzles_in_list} puzzle files with step limit {step_limit}.")

    for i, puzzle_file in enumerate(puzzle_file_list):
        # Optional: Print progress for long lists
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{total_puzzles_in_list}...")
        try:
            # Load fresh state for each puzzle
            initial_state = WitnessState()
            initial_state.read_state(puzzle_file)

            # Run the solver (ensure this function is defined/imported)
            # Set verbose_solver=False for clean analysis output
            outcome, path_length, _ = ga.solve_with_A_star(
                initial_state,
                weights,
                step_limit# Keep this False for aggregated analysis
            )

            # Increment the counter for the specific outcome
            if outcome in outcomes:
                outcomes[outcome] += 1
            else:
                print(f"Warning: Unknown outcome '{outcome}' from solver on {os.path.basename(puzzle_file)}.")
                outcomes["Error"] += 1 # Count unexpected outcomes

            processed_count += 1

        except FileNotFoundError:
            print(f"Warning: Puzzle file not found during analysis: {puzzle_file}, skipping.")
            # Don't increment processed_count if file not found
        except Exception as e:
            print(f"Warning: Error processing puzzle {os.path.basename(puzzle_file)} during analysis: {e}")
            outcomes["Error"] += 1
            processed_count += 1 # Count it as processed, but failed with error

    print("\n--- Analysis Results ---")
    if processed_count == 0:
        print("No puzzles were processed successfully.")
        return outcomes

    print(f"Total puzzles processed: {processed_count}/{total_puzzles_in_list}")
    # Print counts and percentages for each outcome
    for outcome, count in outcomes.items():
        percentage = (count / processed_count * 100)
        print(f"- {outcome:<30}: {count:<4} ({percentage:.1f}%)")

    return outcomes

# --- Run the Analysis ---
if __name__ == "__main__":
    if not test_files:
        print("Error: No test files found or generated. Cannot run analysis.")
    else:
        # Make sure solve_with_heuristic is defined/imported correctly
        # Perform the analysis using the best weights and the training file list
        analysis_results = analyze_heuristic_performance(
            best_evolved_weights,
            test_files, # Analyze on the set the GA trained on
            analysis_solver_step_limit
        )

        # --- Interpretation Guidance ---
        print("\n--- Interpretation Guidance ---")
        # Check if processed_count is valid before division
        if analysis_results and sum(analysis_results.values()) > analysis_results["Error"]:
             processed_ok = sum(analysis_results.values()) - analysis_results["Error"]
             success_rate = analysis_results["Success"] / processed_ok * 100 if processed_ok > 0 else 0
             fail_sep_rate = analysis_results["Reached Goal, Failed Separation"] / processed_ok * 100 if processed_ok > 0 else 0
             stuck_rate = analysis_results["Stuck"] / processed_ok * 100 if processed_ok > 0 else 0
             limit_rate = analysis_results["Step Limit Reached"] / processed_ok * 100 if processed_ok > 0 else 0

             print(f"Success Rate on Training Set: {success_rate:.1f}%")

             if analysis_results["Error"] > 0:
                 print(f"Note: {analysis_results['Error']} puzzles caused errors.")
             if limit_rate > 10: # If more than 10% timed out
                 print(f"NOTE: 'Step Limit Reached' occurred {limit_rate:.1f}% of the time. The limit ({analysis_solver_step_limit}) might still be too low, or heuristic causes inefficient paths.")
             # Compare primary failure modes (excluding Success and Error)
             primary_failure_mode = max(
                 ("Reached Goal, Failed Separation", fail_sep_rate),
                 ("Stuck", stuck_rate),
                 ("Step Limit Reached", limit_rate), # Include limit rate here too
                 key=lambda item: item[1]
             )

             print(f"\nPrimary reason for non-success appears to be: '{primary_failure_mode[0]}' ({primary_failure_mode[1]:.1f}%)")

             if primary_failure_mode[0] == "Reached Goal, Failed Separation":
                 print("-> Suggests focusing on improving HEURISTIC FEATURES regarding color separation.")
             elif primary_failure_mode[0] == "Stuck":
                  print("-> Suggests the GREEDY SOLVER STRATEGY might be limiting. Consider A* search.")
             elif primary_failure_mode[0] == "Step Limit Reached" and limit_rate > 10:
                  print("-> Suggests increasing the SOLVER STEP LIMIT further or addressing path inefficiency.")
        else:
             print("Could not determine primary failure mode due to lack of processed puzzles or errors.")