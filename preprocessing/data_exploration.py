import sys
import correlation
import info
import vis

def do_sub(file):
    info.data_exploration_info(file)
    vis.data_exploration_vis(file)
    correlation.correlation(file)
    

if __name__ == '__main__':
    file = sys.argv[1]
    do_sub(file)