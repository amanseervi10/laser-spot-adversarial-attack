def log_to_file(message, filename):
    with open(filename, "a") as f:
        f.write(str(message) + "\n")


def print_summary(args, total_time, non_zero_scores, zero_score_times,
                    avg_time, avg_zero_time, max_zero_time,
                    avg_non_zero_score, avg_non_zero_time):
    """ Print attack summary in a structured format. """

    if not args.stats:
        return  # Skip logging if no stats file is provided

    divider = "=" * 60
    log_to_file(f"\n{divider}", args.stats)
    log_to_file(f"{' ATTACK PARAMETERS ':^60}", args.stats)
    log_to_file(f"{divider}", args.stats)

    log_to_file(f"{'Model:':<20} {args.model}", args.stats)
    log_to_file(f"{'Wavelength:':<20} {args.wavelength} nm", args.stats)
    log_to_file(f"{'Number of Dots:':<20} {args.num_dots}", args.stats)
    log_to_file(f"{'Iterations:':<20} {args.iterations}", args.stats)
    log_to_file(f"{'Restarts:':<20} {args.restarts}", args.stats)
    log_to_file(f"{'Moves per Dot:':<20} {args.moves_per_dot}", args.stats)
    log_to_file(f"{'Images Saved:':<20} {'Yes' if args.save_image else 'No'}", args.stats)
    log_to_file(f"{'Log file:':<20} {args.log}", args.stats)

    log_to_file(f"{divider}\n", args.stats)

    # Performance Summary
    log_to_file(f"{' PERFORMANCE METRICS ':^60}", args.stats)
    log_to_file(f"{divider}", args.stats)

    log_to_file(f"{'Total Time:':<40} {total_time:.2f} sec", args.stats)
    log_to_file(f"{'Attack Success Rate:':<40} {len(zero_score_times) / (len(non_zero_scores) + len(zero_score_times)) * 100:.2f}%", args.stats)
    log_to_file(f"{'Average Run Time:':<40} {avg_time:.2f} sec", args.stats)
    log_to_file(f"{'Average Time for Score 0:':<40} {avg_zero_time:.2f} sec", args.stats)
    log_to_file(f"{'Max Time for Score 0:':<40} {max_zero_time:.2f} sec", args.stats)
    log_to_file(f"{'Average Score when not 0:':<40} {avg_non_zero_score:.3f}", args.stats)
    log_to_file(f"{'Average Time for non-zero Scores:':<40} {avg_non_zero_time:.2f} sec", args.stats)

    log_to_file(f"{divider}\n", args.stats)


def parse_stats(args):
    total_time = 0
    count = 0
    zero_score_times = []
    non_zero_scores = []
    non_zero_times = []

    with open(args.log, 'r') as file:
        content = file.readlines()

    i = 0
    while i < len(content):
        if content[i].startswith("Processed"):
            if i + 3 >= len(content):  # Ensure enough lines exist
                break
            
            score_line = content[i + 2].strip()
            time_line = content[i + 3].strip()

            if "Best score:" in score_line and "Time taken:" in time_line:
                score = float(score_line.split(":")[-1])
                time = float(time_line.split(":")[-1].replace("s", "").strip())
                
                total_time += time
                count += 1

                if score == 0:
                    zero_score_times.append(time)
                else:
                    non_zero_scores.append(score)
                    non_zero_times.append(time)
            
            i += 4  # Move to the next entry
        else:
            i += 1

    avg_time = total_time / count if count else 0
    avg_zero_time = sum(zero_score_times) / len(zero_score_times) if zero_score_times else 0
    max_zero_time = max(zero_score_times) if zero_score_times else 0
    avg_non_zero_score = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0
    avg_non_zero_time = sum(non_zero_times) / len(non_zero_times) if non_zero_times else 0

    log_to_file("-----------------------------------------------------------------------------", args.stats)
    log_to_file("-----------------------------------------------------------------------------", args.stats)

    print_summary(args, total_time, non_zero_scores, zero_score_times,
                avg_time, avg_zero_time, max_zero_time,
                avg_non_zero_score, avg_non_zero_time)
