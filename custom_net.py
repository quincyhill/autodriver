from turtle import window_height
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import Union


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


    
def generate_weight() -> float:
    pos_or_neg = random.randint(0, 1)
    weight_value = 0
    if pos_or_neg:
        weight_value = random.random() * 100
    else:
        weight_value = -1 * random.random() * 100
    return weight_value
    
def calculate_weighted_sum(bias: Union[int, float]) -> None:
    weighted_sum = 0
    
    for _ in range(5 * 5):
        weight_value = generate_weight()
        activation_value = random.random()
        weighted_sum += (activation_value * weight_value)
        
    weighted_sum += bias
    
    print("Weighted sum before sigmoid: ", weighted_sum) 
    # The value needs to be between 0 and 1
    weighted_sum = sigmoid(weighted_sum)
    print("Weighted sum after sigmoid: ", weighted_sum)
    
calculate_weighted_sum(-10)

# Also each vector or neuron has its own bias as well so this -10 was just a single one
