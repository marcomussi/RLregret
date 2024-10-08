import json

config = {
    "S" : 3,
    "A" : 3,
    "H" : 10,
    "K" : 1000000,
    "alg_lst" : ["BF", "BFI", "CH", "CHI", "MVP"],
    "n_trials" : 10,
    "n_cores" : 10
}

with open('configs/config6.json', 'w') as f:
    json.dump(config, f)

# conda activate py39

# python3 parallel_runner.py config

# scp mussi@turing.deib.polimi.it:RLregret/results.zip .
