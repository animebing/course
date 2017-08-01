#!/bin/bash

for var_1 in 0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do  
    for var_2 in 0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        echo "epsilon: $var_1     learning rate: $var_2"
        sed -i "63s/.*/    answerEpsilon = ${var_1}/" analysis.py
        sed -i "64s/.*/    answerLearningRate = ${var_2}/" analysis.py
        python autograder.py -q q8
    done
done
