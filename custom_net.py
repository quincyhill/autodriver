from turtle import window_height
import numpy as np
import math
import pandas as pd
import random
from typing import Union, List

def sigmoid(x):
    # Our activation function, it decides if the output is 0 or 1
    return 1 / (1 + math.exp(-x))
    
def generate_weight() -> float:
    # Generate a random weight between 0 and 100
    pos_or_neg = random.randint(0, 1)

    weight_value = 0
    if pos_or_neg:
        weight_value = random.random() * 100
    else:
        weight_value = -1 * random.random() * 100
    return weight_value

def calculate_weighted_sum_3_input() -> float:
    weighted_sum = 0
    # Calculate the weighted sum, this case it simulates 3 inputs
    for i in range(3):
        # Weight is a real number
        weight_value: float = generate_weight()

        # Activation aka x is an integer either 0 or 1
        activation_value: int = random.randint(0, 1)
        
        print(f"Input {i} has activation: {activation_value} and weight: {weight_value}")

        weighted_sum += (activation_value * weight_value)
    return weighted_sum
        


    
def calculate_activation_from_3_random_inputs_with_bias(bias: float) -> int:
    # Based of the equation: w1 * x1 + w2 * x2 + ... + wn * xn + b
    # The activation is based on the sigmoid function
    weighted_sum = calculate_weighted_sum_3_input()
        
    print("Weighted sum before bias: ", weighted_sum)

    # Add the bias to complete
    weighted_sum += bias
    
    print("Weighted sum before sigmoid: ", weighted_sum) 
    # The value needs to be between 0 and 1
    weighted_sum = sigmoid(weighted_sum)

    print("Weighted sum after sigmoid: ", weighted_sum)
    # still missing our threshold value
    
    # Our activation value after the sigmoid rounded
    print(round(weighted_sum))
    
    return round(weighted_sum)
    
def am_i_going_to_the_festival() -> int:
    # Our three inputs aka activation values
    # weather is good
    x0 = 1
    
    # going with gf / bf
    x1 = 0

    # festival is close
    x2 = 1
    
    # Our three weights aka how much we care about said input
    # in this case we really care about the weather and equally care less about going with bf / gf and if the festival is close by
    w0 = 6
    w1 = 2
    w2 = 2
    
    # Our threshold, we need to be above this value in order to actually make the decision to go to the festival
    threshold = 5
    
    # now calcuate the weighted sum
    weighted_sum = (w0 + w1) + (w1 * x1) + (w2 * x2) 
    
    if weighted_sum > threshold:
        print("I'm going to the festival")
        return 1
    else:
        print("I'm not going to the festival")
        return 0
if __name__ == '__main__':
    am_i_going_to_the_festival()
    
# Also each vector or neuron has its own bias as well so this -10 was just a single one