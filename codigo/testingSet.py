#!/usr/bin/python
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TestingSet:
	"""classe que usa um conjunto de instancias como teste e retorna a accuracy de um modelo que eh passado como uma lista de rules a essa classe
	"""
	def __init__(self,data_reader,list_rules):
		'''		
		header_line = data_reader.columns
		target = header_line[-1]
		y = data_reader[target]

		print("Scaling features...Z score")
		print(".................")
		X = data_reader.iloc[:,:-1]
		scaler = StandardScaler()
		std_matrix = scaler.fit_transform(X)
		X = pd.DataFrame(data=std_matrix,columns=header_line[:-1])
		X["Ballond'OrNominee"] = y		
		print(X)
		self.data = X
		'''
		self.data = data_reader
		self.list_rules = list_rules 

	def begin_testingZeroR(self):
		header_line = self.data.columns
		class_column =self.data[header_line[-1]]
		




		rule = self.list_rules[0]
		tokens = rule.split(" ")	

		if (tokens[0] != header_line[-1]) or (len(tokens) != 3 ):
			print "Formato da training set nao corresponde ao formato da regra"
			sys.exit(1)
		
		maskOK = (class_column == tokens[2])
		maskNotOK = (class_column != tokens[2])
		samplesOK = self.data[maskOK]  
		samplesNotOK = self.data[maskNotOK]
			
		acc = float(len(samplesOK))/float(len(class_column))
		
		print "Accuracy ",acc * 100, " %"
		print("Correct predicted Instances: ", len(samplesOK.index))
		'''		
		for player_name in samplesOK.index:
			print player_name
		'''

	def begin_testingOneR(self):
		header_line = self.data.columns
		playername_column = self.data.index
		class_column =self.data[header_line[-1]]
		count_correct = 0
		correct_players = []
		n_true_negative=0
		n_true_positive=0
		n_false_positive=0
		n_false_negative=0
		#print(playername_column)
		#includes Z-score scaler
		
		'''
		scaler = StandardScaler()
		std_matrix = scaler.fit_transform(self.data.ix[:,:-1])
		self.data = pd.DataFrame(data=std_matrix,columns=header_line[:-1])
		print("Z-score TESTING")
		self.data = self.data.set_index(playername_column)
		self.data["Ballond'OrNominee"] = class_column
		#print(self.data.ix[:,-1])
		'''
		#begin testing
		for rule in self.list_rules:
			tokens = rule.split(" ")
			#print tokens 

			if (tokens[5] != header_line[-1]) or (len(tokens) != 8):
				print "Formato da training set nao corresponde ao formato da regra"
				sys.exit(1)

			attr_column = self.data[tokens[1]]
			
			vals = tokens[3].split("to")
			#print vals
			begin_val = float(vals[0])
			end_val = float(vals[1])	

			maskAttr = ((attr_column >= begin_val) & (attr_column < end_val))
			selected = self.data[maskAttr]

			for player_name in selected.index:
				cls = class_column.loc[player_name]

				if (cls == tokens[7]) and (cls=="Yes"):
					correct_players.append(player_name)
					#print player_name
					n_true_positive +=1
					count_correct += 1
				elif (cls == tokens[7]) and (cls=="No"):
					correct_players.append(player_name)
					#print player_name
					n_true_negative +=1
					count_correct += 1
				elif (cls != tokens[7]) and (cls=="No"):
					#print player_name
					n_false_positive +=1
				elif (cls != tokens[7]) and (cls=="Yes"):
					n_false_negative +=1


		acc = float(count_correct)/float(len(class_column))
		print "Accuracy ",acc * 100, " %"
		print("Correct predicted Instances: ",len(correct_players))
		print("True positive: {},  False Negative: {}".format(n_true_positive,n_false_negative))
		print("False positive: {}, True Negative: {}".format(n_false_positive,n_true_negative))
		'''
		for player_name in correct_players:
			print (class_column.loc[player_name]+" "+str(player_name))
		'''
			
