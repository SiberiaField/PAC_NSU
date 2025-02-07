import numpy as np
import argparse

'''def second_var():
    real_prob = (1 / real_arr.size) * (1 - P)
    synthetic_prob = (1 / real_arr.size) * P
    probs = np.concatenate([np.repeat(real_prob, real_arr.size),
                           np.repeat(synthetic_prob, real_arr.size)])
    arrs = np.concatenate([real_arr, synthetic_arr])
    res = rng.choice(arrs, real_arr.size, p=probs, replace=False)
    print(f"\nSecond var:\n{res}\n")'''

parser = argparse.ArgumentParser(description= "random_select")
parser.add_argument('real_in_file', type=str,
                    help="Path to input file with real values")
parser.add_argument('synthetic_in_file', type=str,
                    help="Path to input file with synthetic values")
parser.add_argument('prob', type=float,
                    help="Probability of taking synthetic values")

real_input = open(parser.parse_args().real_in_file, 'r')
synthetic_input = open(parser.parse_args().synthetic_in_file, 'r')

real_arr = np.fromstring(real_input.read(), sep=' ', dtype=float)
synthetic_arr = np.fromstring(synthetic_input.read(), sep=' ', dtype=float)
P = parser.parse_args().prob
if P < 0 or P > 1:
    raise ValueError("P must be in range [0, 1], type = float\n")

rng = np.random.default_rng()
def first_var():
    probs = rng.uniform(0, 1, real_arr.size)
    res = np.where(probs > P, real_arr, synthetic_arr)
    print(f"\nFirst var:\n{res}")

def second_var():
    probs = rng.choice([True, False], real_arr.size, p=[1 - P, P])
    res = np.where(probs, real_arr, synthetic_arr)
    print(f"\nSecond var:\n{res}\n")

first_var()
second_var()