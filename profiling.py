import pstats

# Create a Stats object
p = pstats.Stats('profile_stats.prof')

# Sort the statistics by cumulative time spent
p.sort_stats('cumulative').print_stats(10)

