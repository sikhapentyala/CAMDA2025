import os
import sys
import importlib

import pandas as pd
import numpy as np

from pandas import DataFrame
from typing import Dict, Any
from generators.models.base import BaseDataGenerator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PRO-GENE-GEN"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PRO-GENE-GEN/models/Private_PGM"))

pgg = importlib.import_module("model")

# 
# We adapt the Private-PGM model from 
# https://github.com/MarieOestreich/PRO-GENE-GEN/tree/main
# applied to the TCGA data.
# 
# Uses quantile binning into 4 bins, defaults to epsilon = 7, 
#  delta = 0.00001, and 10000 iterations
#
# University of Washington Tacoma PPML Lab
#   - Shane Menzies
#   - Sikha Pentyala
#   - Daniil Filienko
#   - Jineta Banerjee
#   - Martine De Cock
#
class PGG_PGM_DataGenerator(BaseDataGenerator):
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        super().__init__(config, split_no)
        self.epsilon = (self.generator_config["epsilon"] if 
            "epsilon" in self.generator_config else 7.0)
        print(f"{self.epsilon}..........")
        self.delta = (self.generator_config["delta"] if 
            "delta" in self.generator_config else 1e-5)
        self.iterations = (self.generator_config["iterations"] if 
            "iterations" in self.generator_config else 10000)

    def label(self, y_values: DataFrame) -> DataFrame:
        unique_labels = np.unique(y_values.iloc[:, 0].values)
        self.label_mapping = {}
        for i in range(len(unique_labels)):
            self.label_mapping[unique_labels[i]] = i

        self.delabel_mapping = unique_labels
        
        return y_values.map(lambda x: self.label_mapping[x])

    def discretize(self, data: DataFrame) -> DataFrame:
        # 4 bins
        alphas = [0.25, 0.5, 0.75]
        self.num_bins = len(alphas) + 1
        
        data_copy = data.copy()
        data = data.apply(pd.to_numeric, errors='coerce')
        data_quantile = np.quantile(data, alphas, axis=0)

        statistic_dict = {}
        mean_dict = {}
        quantile_dict = {}
        for col in data.columns:
            col_quantiles = data_quantile[:, data.columns.get_loc(col)]
            try:
                discrete_col = np.digitize(data[col], col_quantiles)
            except:
                print("Could not discretize column", col, data[col].values)
            data[col] = discrete_col
            quantile_dict[col] = col_quantiles

            statistic_dict[col] = []
            mean_dict[col] = []
            for bin_idx in range(self.num_bins):
                bin_arr = data_copy[col][discrete_col == bin_idx]
                statistic_dict[col].append(len(bin_arr))
                next_mean = np.mean(bin_arr)
                if np.isnan(next_mean):
                    # Estimate mean 
                    if bin_idx == 0:
                        next_mean = (np.min(data_copy[col]) + col_quantiles[0]) / 2
                    elif bin_idx == (self.num_bins - 1):
                        next_mean = (np.max(data_copy[col]) + col_quantiles[-1]) / 2
                    else:
                        next_mean = (col_quantiles[bin_idx] + col_quantiles[bin_idx + 1]) / 2
                mean_dict[col].append(next_mean)

        self.statistic_dict = statistic_dict
        self.mean_dict = mean_dict
        self.quantile_dict = quantile_dict
        return data

    def dediscretize(self, data: DataFrame) -> DataFrame:

        dedisc_data = DataFrame(dtype=float).reindex_like(data)
        for col in data.columns:
            means = np.asarray(self.mean_dict[col])
            dedisc_data.loc[:, col] = data.loc[:, col].map(lambda x: means[x])

        return dedisc_data
        
    def delabel(self, labels: DataFrame) -> DataFrame:
        return labels.map(lambda x: self.delabel_mapping[x])
    
    def train(self):
        data_save_dir = os.path.join(self.config["dir_list"]["home"],
                                     self.config["dir_list"]["data_save_dir"])

        real_save_dir = os.path.join(data_save_dir, self.dataset_name, "real")

        # Read data from files
        X_train = pd.read_csv(os.path.join(real_save_dir, f"X_train_real_split_{self.split_no}.csv"))
        y_train = pd.read_csv(os.path.join(real_save_dir, f"y_train_real_split_{self.split_no}.csv"))

        # Bin and label training data
        X_train = self.discretize(X_train)
        y_train = self.label(y_train)
        self.label_column = y_train.columns[0]

        # Fill in domain
        self.domain = {}
        self.domain[self.label_column] = len(self.label_mapping)
        for col in X_train.columns:
            self.domain[col] = self.num_bins

        self.train_data = pd.concat([X_train, y_train], axis=1)

        print(f"Private_PGM({self.label_column}, True, e={self.epsilon}, delta={self.delta})")
        print(f"Training for {self.iterations} iterations")

        self.model = pgg.Private_PGM(self.label_column, True, self.epsilon, self.delta)
        self.model.train(self.train_data, self.domain, num_iters=self.iterations)

    def generate(self):
        synth = DataFrame(self.model.generate(), columns=self.train_data.columns)
        print(synth)
        X_synth, y_synth = synth.iloc[:, :-1], synth.iloc[:, -1]
        X_synth = self.dediscretize(X_synth)
        y_synth = self.delabel(y_synth)

        print(X_synth)
        print(y_synth)

        return X_synth, y_synth


    def load_from_checkpoint(self):
        pass

    def generate_for_type(self):
            pass
        
