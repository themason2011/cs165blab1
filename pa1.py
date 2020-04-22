import numpy as np
# Starter code for CS 165B HW2 Spring 2019

def calc_centroid(data):

    cen = np.mean(data, axis = 0)
    return cen


def create_discriminant(centroid_pos, centroid_neg):
    return

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """

    dim = training_input[0][0]
    num_a = training_input[0][1]
    num_b = training_input[0][2]
    num_c = training_input[0][3]

    classA = training_input[1:1+num_a]
    classB = training_input[2+num_b:2+num_a+num_b]
    classC = training_input[2+num_a+num_b:-1]

    centroidA = calc_centroid(classA)
    centroidB = calc_centroid(classB)
    centroidB = calc_centroid(classB)

    print(centroidA)
    print(centroidB)
    print(centroidC)

    pass
