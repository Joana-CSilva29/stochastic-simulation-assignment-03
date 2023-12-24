import pstats

"""Script to profile the performance of the code"""

p = pstats.Stats('profile_stats.prof')

p.sort_stats('cumulative').print_stats(10) 

