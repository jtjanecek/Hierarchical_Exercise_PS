import pandas as pd
import os.path
import glob
import shutil

subj_list = pd.read_csv("subj_list.csv")

input_path = "/tmp/yassamri2/Tsukuba/2018_Intervention/PS_log_data/"
output_path = "raw_data/"

for timepoint in ["P1", "P2"]:
	for subj in subj_list['Subject'].values:
		subj_path = input_path + timepoint + "_log/" + str(subj) + "_MDTO_log.txt"
		if not os.path.isfile(subj_path):
			print(timepoint, subj)
		else:
			shutil.copyfile(subj_path, output_path + timepoint + "_log/" + os.path.basename(subj_path))
