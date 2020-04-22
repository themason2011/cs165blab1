# Mason Corey, CS 165B Spring 2020
import math
import numpy as np
import warnings

def run_train_test(training_input, testing_input):
    #Find number of each data point class, parse data points, and find the centroid of each data set. Dimension not needed bc numpy will automatically detect dimension by length of data points
    num_a = training_input[0][1]
    a_data = training_input[1:1+num_a]
    a_centroid = np.mean(a_data,axis = 0,dtype=np.float64)

    num_b = training_input[0][2]
    b_data = training_input[2 + num_a: 2 + num_a + num_b]
    b_centroid = np.mean(b_data,axis = 0,dtype=np.float64)
    
    c_data = training_input[1 + num_a + num_b: ]
    c_centroid = np.mean(c_data,axis = 0,dtype=np.float64)

    #Calculate disciminants and orthogonal vectors for each class
    AB_disc = (np.linalg.norm(a_centroid) ** 2 - np.linalg.norm(b_centroid) ** 2) / 2.0
    AB_orth = np.subtract(a_centroid,b_centroid)

    AC_disc = (np.linalg.norm(a_centroid) ** 2 - np.linalg.norm(c_centroid) ** 2) / 2.0
    AC_orth = np.subtract(a_centroid,c_centroid)

    BC_disc = (np.linalg.norm(b_centroid) ** 2 - np.linalg.norm(c_centroid) ** 2) / 2.0
    BC_orth = np.subtract(b_centroid,c_centroid)

    #Still don't need dimension for test. Find number of each data point class, parse data points, and check each class to see what training model determines them to be. Increment FPRs and all that for each class as you go
    num_a_test = testing_input[0][1]
    a_data_test = testing_input[1:1+num_a]

    num_b_test = testing_input[0][2]
    b_data_test = testing_input[2 + num_a: 2 + num_a + num_b]

    num_c_test = testing_input[0][3]
    c_data_test = testing_input[1 + num_a + num_b: ]

    tp_a, tp_b, tp_c, fp_a, fp_b, fp_c, fn_a, fn_b, fn_c, tn_a, tn_b, tn_c, pos_a, pos_b, pos_c, neg_a, neg_b, neg_c = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    #Iterate through A-labelled points
    for i in range(1,num_a_test+1):
        index = i-1
        
        AB_dot = np.dot(testing_input[i],AB_orth)
        AC_dot = np.dot(testing_input[i],AC_orth)
        BC_dot = np.dot(testing_input[i],BC_orth)

        #Check whether it's A or B, give priority to A
        if AB_dot - AB_disc >= 0:
            #Check whether it's A or C, give priority to A
            if AC_dot - AC_disc >= 0:
                class_type='A'
            else:
                class_type='C'
        else:
            #Check whether it's B or C, give priority to B
            if BC_dot - BC_disc >= 0:
                class_type='B'
            else:
                class_type='C'

        #Check which class type was predicted and iterate values for the contingency tables accordingly
        if class_type == 'A':
            tp_a += 1
            tn_b += 1
            tn_c += 1
            pos_a += 1
            neg_b += 1
            neg_c += 1

        elif class_type == 'B':
            fp_b += 1
            fn_a += 1
            tn_c += 1
            neg_a += 1
            pos_b += 1
            neg_c += 1

        elif class_type == 'C':
            fp_c += 1
            fn_a += 1
            tn_b += 1
            neg_a += 1
            neg_b += 1
            pos_c += 1

    #Iterate through B-labelled points
    for i in range(num_a_test+1,num_a_test+num_b_test+1):
        index = i-1
        
        AB_dot = np.dot(testing_input[i],AB_orth)
        AC_dot = np.dot(testing_input[i],AC_orth)
        BC_dot = np.dot(testing_input[i],BC_orth)

        #Check whether it's A or B, give priority to A
        if AB_dot-AB_disc >= 0:
            #Check whether it's A or C, give priority to A
            if AC_dot-AC_disc >= 0:
                class_type='A'
            else:
                class_type='C'
        else:
            #Check whether it's B or C, give priority to B
            if BC_dot-BC_disc >= 0:
                class_type='B'
            else:
                class_type='C'

        #Check which class type was predicted and iterate values for the contingency tables accordingly
        if class_type == 'A':
            fp_a += 1
            fn_b += 1
            tn_c += 1
            pos_a += 1
            neg_b += 1
            neg_c += 1

        elif class_type == 'B':
            tp_b += 1
            tn_a += 1
            tn_c += 1
            neg_a += 1
            pos_b += 1
            neg_c += 1

        elif class_type == 'C':
            fp_c += 1
            fn_b += 1
            tn_a += 1
            neg_a += 1
            neg_b += 1
            pos_c += 1

    #Iterate through C-labelled points
    for i in range(num_a_test+num_b_test+1,num_a_test+num_b_test+num_c_test+1):
        index = i-1
        
        AB_dot = np.dot(testing_input[i],AB_orth)
        AC_dot = np.dot(testing_input[i],AC_orth)
        BC_dot = np.dot(testing_input[i],BC_orth)

        #Check whether it's A or B, give priority to A
        if AB_dot - AB_disc >= 0:
            #Check whether it's A or C, give priority to A
            if AC_dot - AC_disc >= 0:
                class_type='A'
            else:
                class_type='C'
        else:
            #Check whether it's B or C, give priority to B
            if BC_dot - BC_disc >= 0:
                class_type='B'
            else:
                class_type='C'

        #Check which class type was predicted and iterate values for the contingency tables accordingly
        if class_type == 'A':
            fp_a += 1
            fn_c += 1
            tn_b += 1
            pos_a += 1
            neg_b += 1
            neg_c += 1

        elif class_type == 'B':
            fp_b += 1
            fn_c += 1
            tn_a += 1
            neg_a += 1
            pos_b += 1
            neg_c += 1

        elif class_type == 'C':
            tp_c += 1
            tn_a += 1
            tn_b += 1
            neg_a += 1
            neg_b += 1
            pos_c += 1

    #Calculate tpr, fpr, error_rate, acc, and prec for all classes. Then average them and store in a dictionary and return
    tpr_a = tp_a/float(num_a_test)
    tpr_b = tp_b/float(num_b_test)
    tpr_c = tp_c/float(num_c_test)

    fpr_a = fp_a/(float(num_b_test)+float(num_c_test))
    fpr_b = fp_b/(float(num_a_test)+float(num_c_test))
    fpr_c = fp_c/(float(num_a_test)+float(num_b_test))

    error_rate_a = (fp_a+fn_a)/(pos_a+neg_a)
    error_rate_b = (fp_b+fn_b)/(pos_b+neg_b)
    error_rate_c = (fp_c+fn_c)/(pos_c+neg_c)
    
    acc_a = (tp_a+tn_a)/(pos_a+neg_a)
    acc_b = (tp_b+tn_b)/(pos_b+neg_b)
    acc_c = (tp_c+tn_c)/(pos_c+neg_c)

    prec_a = tp_a/pos_a
    prec_b = tp_b/pos_b
    prec_c = tp_c/pos_c

    results = {}
    results['tpr'] = (tpr_a+tpr_b+tpr_c)/3.0
    results['fpr'] = (fpr_a+fpr_b+fpr_c)/3.0
    results['error_rate'] = (error_rate_a+error_rate_b+error_rate_c)/3.0
    results['accuracy'] = (acc_a+acc_b+acc_c)/3.0
    results['precision'] = (prec_a+prec_b+prec_c)/3.0

    return results