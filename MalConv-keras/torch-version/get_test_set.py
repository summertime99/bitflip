import os
import random
import csv
import pandas as pd

filenames = set()

result = pd.read_csv("result.csv", header=None, names=["fn", "label0", "score"])
data = pd.read_csv("/home/sample/wy/bitflip/MalConv-keras/src/androzoo1.csv", header=None, names=["fp", "label1"])

data["fn"] = data["fp"].apply(lambda x: x.strip().split("/")[-1])

merged = pd.merge(result, data, on=["fn"], how="left")

merged = merged[["fp", "label0"]]

merged.to_csv("ori_test.csv", index=False, header=None)