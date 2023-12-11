import pstats

# Create a Stats object
p = pstats.Stats('profile_stats.prof')

# Sort the statistics by cumulative time spent
p.sort_stats('cumulative').print_stats(10)  # Print the top 10 functions

# You can also sort by other criteria like 'time', 'calls', etc.
