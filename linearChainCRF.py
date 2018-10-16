"""
Conditional Random Field Implementation
Copyright (c) 2018 Marko Melnick
"""

import pandas as pd

import time
import json
import datetime
import os.path

from collections import Counter
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
pd.set_option('display.max_columns',50)

SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None
STARTING_LABEL  = "ASTART"
STARTING_LABEL_INDEX = 0


def _gradient(params, *args):
    return GRADIENT * -1

def _callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0

def _generate_potential_table_marko(params, label_index_d, num_labels, df, dfgrp, inference=True):
    """
    Generates a potential table using given observations.
    * potential_table[t][prev_y, y]
        := exp(inner_product(params, feature_vector(prev_y, y, X, t)))
        (where 0 <= t < len(X))
    """
    tables = list()

    print(df.head())
    for t in range(len(dfgrp)):
        table = np.zeros((num_labels, num_labels))

        current_label = dfgrp["-1,y"].iloc[t]
        feature_index_list = label_index_d[current_label]
        print(current_label)
        print(feature_index_list)
        if inference == False:
            print("params are: ", params)
    #
    # training_feature_data[0] is [
    #     [((0, 1), {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}), ((-1, 1), {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}),
    #      ((1, 3), {1705, 90, 506, 1709}), ((-1, 3), {1706, 397, 399, 1429, 91}), ((3, 3), {1428, 396, 398, 391}),
    #      ((6, 1), {466}), ((2, 1), {1760, 1761, 1762, 719, 1757, 1758, 1759}), ((1, 1), {749}), ((7, 8), {652}),
    #      ((-1, 8), {653}), ((0, 7), {1210, 1212, 1206, 1214}), ((-1, 7), {1215, 1211, 1213, 1207}), ((5, 5), {254}),
    #      ((-1, 5), {255}), ((4, 5), {864})], [((1, 2), {32, 34, 36, 38, 40, 42, 44, 46, 48, 20, 22, 24, 26, 28, 30}),
    #                                           ((-1, 2), {33, 35, 37, 39, 41, 43, 45, 47, 49, 21, 23, 25, 27, 29, 31}),
    #                                           ((3, 2), {965, 966, 1456, 1461, 1462, 535, 1722}),
    #                                           ((3, 6), {450, 452, 428}), ((-1, 6), {453, 451, 429}),
    #                                           ((8, 2), {683, 694}), ((7, 2), {1224, 1225, 1220, 1221}),
    #                                           ((5, 5), {1674, 1676, 286}), ((-1, 5), {1675, 1677, 287}),
    #                                           ((5, 1), {329}), ((-1, 1), {769, 330}), ((3, 3), {402, 1558}),
    #                                           ((-1, 3), {522, 403}), ((7, 8), {658}), ((-1, 8), {659}),
    #                                           ((3, 1), {1370}), ((1, 3), {1520, 521}), ((2, 1), {1802}),
    #                                           ((3, 4), {150}), ((-1, 4), {151}), ((1, 1), {768}), ((1, 7), {802}),
    #                                           ((-1, 7), {803}), ((3, 7), {1847})],
    #     [((2, 1), {64, 66, 68, 70, 72, 74, 76, 78, 80, 50, 82, 52, 84, 54, 86, 56, 58, 60, 62}),
    #      ((-1, 1), {65, 67, 69, 71, 73, 75, 77, 79, 81, 51, 83, 53, 85, 55, 87, 57, 59, 61, 63}),
    #      ((5, 1), {1688, 1689, 318}), ((1, 3), {361}), ((-1, 3), {362, 1591, 1549, 1551}), ((3, 6), {434}),
    #      ((-1, 6), {435}), ((8, 2), {688}), ((-1, 2), {689, 565}), ((3, 3), {1544, 1590, 1548, 1550}),
    #      ((3, 2), {564, 1727}), ((6, 1), {495, 494, 487}), ((4, 5), {196}), ((-1, 5), {197}), ((1, 7), {810}),
    #      ((-1, 7), {811}), ((7, 4), {848}), ((-1, 4), {849})],
    #     [((1, 3), {96, 98, 100, 102, 122, 104, 106, 108, 120, 110, 112, 124, 114, 116, 118, 88, 90, 92, 94}),
    #      ((-1, 3), {97, 121, 99, 101, 103, 105, 107, 123, 109, 111, 113, 115,
    #
    #








    # for t in range(len(X)):
    #     table = np.zeros((num_labels, num_labels))
    #     if inference:
    #         print("Inference is running")
    #         for (prev_y, y), score in feature_set.calc_inner_products(params, X, t):
    #             if prev_y == -1:
    #                 table[:, y] += score
    #             else:
    #                 table[prev_y, y] += score
    #     else:
    #         for (prev_y, y), feature_ids in X[t]:
    #             score = sum(params[fid] for fid in feature_ids)
    #             if prev_y == -1:
    #                 table[:, y] += score
    #             else:
    #                 table[prev_y, y] += score
    #     table = np.exp(table)
    #     if t == 0:
    #         table[STARTING_LABEL_INDEX+1:] = 0
    #     else:
    #         table[:,STARTING_LABEL_INDEX] = 0
    #         table[STARTING_LABEL_INDEX,:] = 0
    #     tables.append(table)
    #
    #     if t == 1:
    #         print(table)
    #
    # return tables





def _log_likelihood(params, *args):
    """
    Calculate likelihood and gradient
    """
    # x,y,z,f = args
    total_logZ = 0

    df, featuredf, label_index_d, labels, squared_sigma = args

    grps = df['sentence'].dropna().unique().tolist()
    d_grp = {grp: df.loc[df["sentence"] == grp] for grp in grps}

    for grp in grps:
        dfgrp = d_grp[grp]
        # print(dfgrp)
        potential_table = _generate_potential_table_marko(params, label_index_d, len(labels), df,dfgrp, inference=False)


    # for df in d_grp[]

    # for X_features in training_feature_data:
    #     potential_table = _generate_potential_table(params, len(label_dic), feature_set, X_features, inference=False)
    #
    #     alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
    #     total_logZ += log(Z) + \
    #                   sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
    #     for t in range(len(X_features)):
    #         potential = potential_table[t]
    #         for (prev_y, y), feature_ids in X_features[t]:
    #             # Adds p(prev_y, y | X, t)
    #             if prev_y == -1:
    #                 if t in scaling_dic.keys():
    #                     prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
    #                 else:
    #                     prob = (alpha[t, y] * beta[t, y])/Z
    #             elif t == 0:
    #                 if prev_y is not STARTING_LABEL_INDEX:
    #                     continue
    #                 else:
    #                     prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
    #             else:
    #                 if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
    #                     continue
    #                 else:
    #                     prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
    #             for fid in feature_ids:
    #                 expected_counts[fid] += prob
    #
    # likelihood = np.dot(empirical_counts, params) - total_logZ - \
    #              np.sum(np.dot(params,params))/(squared_sigma*2)
    #
    # gradients = empirical_counts - expected_counts - params/squared_sigma
    # global GRADIENT
    # GRADIENT = gradients
    #
    # global SUB_ITERATION_NUM
    # sub_iteration_str = '    '
    # if SUB_ITERATION_NUM > 0:
    #     sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'
    # print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', likelihood * -1)
    #
    # SUB_ITERATION_NUM += 1
    #
    # return likelihood * -1





class LinearChainCRF():
    """
    Linear-chain Conditional Random Field
    """

    # For L-BFGS
    squared_sigma = 10.0
    params = ""

    def _importDataframe(self, fi):
        f2 = fi[:-5] + "_parsed.data"
        if os.path.isfile(f2) == False:
            print("parsing file...")
            with open(fi) as infile, open(f2,"w") as outfile:
                counter = 1
                for line in infile:
                    line = line[:-1] + " " + str(counter) + "\n"
                    if line[0] == " ":
                        counter = counter + 1
                    else:
                        outfile.write(line)
        df = pd.read_csv(f2, sep=" ", names=["word", "pos", "y", "sentence"])
        return df

    def getFeatures(self):
        df = self.df
        grps=df['sentence'].dropna().unique().tolist()

        self.labels = ["*"] + df['y'].dropna().unique().tolist()  # Make list of labels with * as start to put in as factors

        df_final = pd.DataFrame()
        d_grp = {grp : df.loc[df["sentence"]==grp] for grp in grps}
        for grp in grps:
            df = d_grp[grp].copy()

            #Get prev labels
            df = df.reset_index(drop=True) # Reset the index for each group and dont add new index as column
            df["y_factor"] = pd.Categorical(df["y"], self.labels).codes.astype(str) # Factorize the labels
            df["prev_y"] = df["y"].shift(1)
            df.at[0, "prev_y"] = "*"           # Add dummy start
            df["prev_y_factor"] = pd.Categorical(df["prev_y"], self.labels).codes.astype(str)

            df["prev_y,y"] = df["prev_y_factor"] + ", " + df["y_factor"]
            df["-1,y"] = str(-1) + ", " + df["y_factor"]

            #Define features
            df["U[0]"] = df["word"]
            df["POS_U[0]"] = df["pos"]
            df["U[+1]"] = df["word"].shift(-1)
            df["B[0]"] = df["U[0]"] + " " + df["U[+1]"]
            df["POS_U[+1]"] = df["pos"].shift(-1)
            df["POS_B[0]"] = df["POS_U[0]"] + " " + df["POS_U[+1]"]
            
            df["U[+2]"] =      df["word"].shift(-2)
            df["POS_U[+2]"] =  df["pos"].shift(-2)
            df["POS_B[+1]"] =  df["POS_U[+1]"] + " " + df["POS_U[+2]"]
            df["POS_T[0]"] =   df["POS_U[0]"] + " " + df["POS_U[+1]"] + " " + df["POS_U[+2]"]
            df["U[-1]"] =      df["word"].shift(1)
            df["B[-1]"] =      df["U[-1]"] + " " + df["U[0]"]
            df["POS_U[-1]"] =  df["pos"].shift(1)
            df["POS_B[-1]"] =  df["POS_U[-1]"] + " " + df["POS_U[0]"]
            df["POS_T[-1]"] =  df["POS_U[-1]"] + " " + df["POS_U[0]"] + " " + df["POS_U[+1]"]
            df["U[-2]"] =     df["word"].shift(2)
            df["POS_U[-2]"] =  df["pos"].shift(2)
            df["POS_B[-2]"] =  df["pos"].shift(2) + " " + df["pos"].shift(1)
            df["POS_T[-2]"] =  df["pos"].shift(2) + " " + df["pos"].shift(1) + " " + df["pos"]
            df_final = df_final.append(df, ignore_index=True)

        df = df_final
        df = df.reset_index(drop=True)  # Reset the index for each group
        #Combine label, prev and current label, with list of features

        for col in [s for s  in list(df) if "[" in s]:
            df["-1,y:"+col] = df["-1,y"] + ":" + df[col]
            df["prev_y,y:"+col] = df["prev_y,y"] + ":" + df[col]

        # Save list of labels combined with features to make new master featureslist
        self.featurelist_col = [s for s in list(df) if ":" in s]
        self.df = df

    def featureSum(self):
        df_final = pd.DataFrame()
        for col in self.featurelist_col:
            df = self.df.groupby(col).sum()
            df.rename(columns = {"sentence":"global_count"}, inplace=True) #Rename sentence to counts
            df["feature"] = df.index
            df["feature_type"] = col
            df["label"] = df["feature"].str.split(":", expand=True)[0]
            df_final = df_final.append(df)

        df_final = df_final.reset_index(drop=True)  # Reset the index for each group
        self.featuredf = df_final

    def getLabelsIndex(self):
        print("Geting indexes for labels, and   prev and current labels...")
        df = self.featuredf
        gp = df.groupby('label')
        self.label_index_d = gp.groups

    def _estimate_parameters(self):

        print("_gradient is ", _gradient)
        # print("Length of feature set is", len(self.feature_set))


        print('* Squared sigma:', self.squared_sigma)
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')


        _log_likelihood(self.params, self.df,
                            self.featuredf,
                            self.label_index_d,
                            self.labels,
                            self.squared_sigma)


        #
        # self.params, log_likelihood, information = fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
        #               x0=np.zeros(self.amount_feature),
        #               args=(self.featuredf,
        #                     self.feature_globalcounts_d,
        #                     self.dic_index_current_labels,
        #                     self.dic_index_prev_and_current_labels,
        #                     self.squared_sigma),
        #               callback=_callback)


    def train(self, fi, model_filename):
        start_time = time.time()
        print('[%s] Start training' % datetime.datetime.now())
        print("Reading data set...")
        self.df = self._importDataframe(fi)

        print("Parsing features...")
        self.getFeatures()

        print( str(len(self.labels)),  " labels found, list of labels : ", self.labels)


        print("Calculating global counts for each label*feature ...")
        self.featureSum()
        print("self.featurelist col is : ", self.featurelist_col)
        self.getLabelsIndex()
        print(self.featuredf)
        self.featuredf.to_csv("featuredf.tsv",sep="\t",index=None)
        # print(self.feature_globalcounts_d["prev label, label:U[0]"])
        self._estimate_parameters()


        elapsed_time = time.time() - start_time
        print('* Elapsed time: %f' % elapsed_time)
        print('* [%s] Training done' % datetime.datetime.now())
