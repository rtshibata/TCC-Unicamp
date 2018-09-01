#!/usr/bin/python
import sys
from collections import defaultdict
import csv
#import oneR
#import zeroR

def produce_rules(class_name, class_label):
	list_rules = []
	rule = class_label + " -> " +  class_name
	list_rules.append(rule)
	return list_rules


#CSV file name
csv_name = 'CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'
#path to csv file in your computer
path_2_csv = "/home/master/Dropbox/TCC_RenatoShibata/base"
path_2_csv_ic ="/home/ec2010/ra082674/TCC/data"
with open(path_2_csv_ic + "/" + csv_name,'rb') as csv_playersdata:
	data_reader = csv.reader(csv_playersdata,delimiter=',')

	class_counts = defaultdict(int)
	firstline = True
	for row in data_reader:
	#get label names
		if firstline:
			class_label = row[-1]
			firstline = False
			continue
		else:
			if row[-1]=="Yes":
				class_counts["Yes"] += 1
			elif row[-1]=="No":
				class_counts["No"] += 1
			else:
				sys.exit("Non existant class Error")
	#compare which class has more frequency
	#default: 50% of Yes => rule of Yes
	if class_counts["Yes"] >= class_counts["No"]:
		print(produce_rules("Yes",class_label))
	else:
		print(produce_rules("No",class_label))
		
