import time
from os import listdir
from os.path import isfile, join

from bootstrap import Bootstrap
from levin_tree_search import BFSLevin
from model import Model, UniformModel
from witness import WitnessState

def main():   
    states = {}
    problems_folder = 'problems/puzzles_4x4'
    puzzle_files = [f for f in listdir(problems_folder) if isfile(join(problems_folder, f))]
    easy_instance = None
    j = 1
            
    puzzle_files = sorted(puzzle_files)
    
    for filename in puzzle_files:
        if '.' in filename:
            continue
        
        with open(join(problems_folder, filename), 'r') as file:
            puzzle = file.readlines()
            
            i = 0
            while i < len(puzzle):
                k = i
                while k < len(puzzle) and puzzle[k] != '\n':
                    k += 1
                s = WitnessState()
                s.read_state_from_string(puzzle[i:k])

                if filename == '4x4_469':
                    easy_instance = s

                states['puzzle_' + str(j)] = s
                i = k + 1
                j += 1   
 
    print('Attempting to solve easy instance with uniform model')

    model_uniform = UniformModel()
    levin = BFSLevin()
    cost, expanded, _ = levin.search(easy_instance, model_uniform, 2000)
    print("Found solution with cost ", cost, " and expanded ", expanded)
    # easy_instance.plot()

    # Uncomment the following lines once you have a working implementation of LevinTS
    print('Loaded ', len(states), ' instances.')
    start = time.time()
    model = Model()
    levin = BFSLevin()
    bootstrap = Bootstrap(states)

    bootstrap.train_model(levin, model)

    print('Total time: ', time.time() - start)
if __name__ == "__main__":
    main()
    
    
