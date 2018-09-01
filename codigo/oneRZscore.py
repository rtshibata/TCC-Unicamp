#!/usr/bin/python
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math
from collections import defaultdict

class OneR:
	"""classe que usa classificador OneR

	It generates one rule for each predictor attribute in the data,
then selects only 1 attribute, whose set of rules gives the smallest total error(maximizes the number of correct instances divided by the total number of instances)

	PS: In this implementation, it assumes there are only non-negative values in the dataset

	PS2: In this implementation it was not implemented in a way that the rules generated respects the min_no_bucket

	"""
	#construtor
	def __init__(self,data_reader,class_values,min_no_bucket,n_discret_bins,bin_range_precision):
		#self.list_rules = [] 
		self.data = data_reader
		self.class_values = class_values
		self.min_no_bucket = min_no_bucket
		self.n_discret_bins = n_discret_bins	
		self.bin_range_precision = bin_range_precision
		self.dict_tmp_rules = {} # key = attr, value = list of rules associated to that attr
		self.chosen_attr = '' #best attribute of OneR
	
	#returns list of rules of the chosen attribute whose criteria is met in min_no_bucket
	def get_final_rules(self):
		if (not(self.dict_tmp_rules.has_key(self.chosen_attr))):
			print("Error: no attribute was chosen/found for the final set of rules")
			print("----------------------------------------------------------")
		else:
			
			list_rules = self.dict_tmp_rules[self.chosen_attr]		
			return list_rules
	
	#generate set of rules for dict_tmp_rules
	def produce_rules(self,attr_label,attr_value, class_value, class_label):
		#count the number of instances for the proposed rule
		count_inst = 0

		attr_column = self.data[attr_label]

		maskAttr = (attr_column == attr_value)
		selected = self.data[maskAttr]
		
		header_line = self.data.columns
		class_column =self.data[header_line[-1]]

		for player_index in selected.index:
			#print "count",count_inst,player_index
			if class_column.loc[player_index] == class_value:
				count_inst += 1
			if count_inst >= self.min_no_bucket:
				rule = "IF " + str(attr_label) + " = " + str(attr_value) + " THEN " + str(class_label) + " = " +  str(class_value)
				#search if it is first time producing rules for attr_label
				if (not(self.dict_tmp_rules.has_key(attr_label))):
					self.dict_tmp_rules[attr_label] = [rule]
				else:
					self.dict_tmp_rules[attr_label] += [rule]
				return True
		#print("following rule only has only {} instances and it's needed at least {}".format(count_inst,self.min_no_bucket))
		return False

	#return the name of the attribute with the smallest error and its smallest error in a tuple
	def get_attr_small_error(self,l_all_attr,header_line):
		l_total_error = []
		for list_attr_errors in l_all_attr:			
			#calculate the total error for all discrete values	
			total_error = 0				
			for discrete_value, error in list_attr_errors:
				total_error += math.fabs(error)
			l_total_error.append(total_error)
		#get the smallest error
		smallest_error = min(l_total_error)	  
		index_min = np.argmin(l_total_error)
		chosen_attr = header_line[index_min]
		return chosen_attr,smallest_error

	#substitute a certain column's values for its correspondent discret value
	def	substitute_for_discret(self,discret_values,attr_name):
		#header_line = next(self.data)#gets the attribute names
		#for attr_name in header_line:		
		'''attr_values = self.data[attr_name]#update the position in the column(starts from 1)
		for j in range(len(attr_values)):
			for i in range(len(discret_values)):#tests to see in which discret interval does it belong
				interval = discret_values[i]
				#break down its name to get its numeric values
				begin, end = interval.split("to")
				if attr_values[j] >= float(begin) and attr_values[j] < float(end):
					#substitute for its discret value
					attr_values[j] = interval
		'''
		print "discretizing...",attr_name
		for i in range(len(discret_values)):
			interval_name = discret_values[i]
			#break down its name to get its numeric values
			l = interval_name.split("to")	
			#substitute for its discret value  	
			#np.where((self.data[attr_name] >= (float)begin) & (self.data[attr_name] < (float)end), interval_name, self.data[attr_name]) 
			begin, end = np.array(l,dtype=np.float64)
			#print begin,end, interval_name,attr_name
			#self.data[attr_name] = where((self.data[attr_name] >= begin) & (self.data[attr_name] < end), interval_name, self.data[attr_name]) 
			self.data[attr_name].loc[(self.data[attr_name] >= begin) & (self.data[attr_name] < end)] = interval_name
			#print self.data[attr_name]

	#discretize the numeric values in a predetermined number of intervals(n_discret_bins) AND substitute the numeric values to new discret values
	def discretize_data(self):
		#getting the intervals among all the values of all attributes
		matrix_discret_values = []
		#header_line = next(self.data)#gets the attribute names
		#header_line = self.data.loc[0]
		header_line = self.data.columns

		for attr_name in header_line:
			if attr_name == header_line[-1]:
				break
			attr_values = self.data[attr_name]#gets all its values
			#print self.data[attr_name]
			max_val = self.data[attr_name].max()
			min_val = self.data[attr_name].min()
			#print "valsss",min_val, max_val,attr_name
			n = self.n_discret_bins 
			delta = (float(max_val) - float(min_val))/float(n)
			#print delta,n
			interval_init = min_val
			discret_values = []
			for i in range(n):#save discret values in array
				interval_end = np.where(i==n-1,float("inf"),interval_init+delta)
				interval_init = float("{0:.{prec}f}".format(interval_init, prec=self.bin_range_precision))
				interval_end = float("{0:.{prec}f}".format(float(interval_end), prec=self.bin_range_precision))

				interval_name = str(interval_init)+"to"+str(interval_end)
				interval_init += delta
				discret_values.append(interval_name)
			self.substitute_for_discret(discret_values,attr_name)
			
			matrix_discret_values.append(discret_values)
		
		np_matrix = np.array(matrix_discret_values)
		np_matrix_data = np_matrix.T

		#print np_matrix_data.shape
		#print header_line.shape
		#data_frame_discret_values = pd.DataFrame(data=np_matrix[1:,:],columns=np_matrix[0,:])
		data_frame_discret_values = pd.DataFrame(data=np_matrix_data,columns=header_line[:-1])
		return data_frame_discret_values

	#run the algorithm
	def begin_oneR(self):
		header_line = self.data.columns
		class_column = self.data[header_line[-1]]#gets all values related to class column
		class_label = header_line[-1]

		scaler = StandardScaler()
		std_matrix = (scaler.fit_transform(self.data.ix[:,:-1]))
		self.data = pd.DataFrame(data=std_matrix,columns=header_line[:-1])	
		self.data["Ballond'OrNominee"] = class_column
		print("Z-score")
		#print(self.data)

		m_discret_values = self.discretize_data()
		#print m_discret_values
		l_all_attr =[]
		attr_names = []
		accuracy = {}
		list_c_majority =[]
		#header_line = next(self.data)
		

		#estimate the error related to each attribute
		for attr_name in header_line:
			if attr_name == header_line[-1]:
				break
			attr_values = self.data[attr_name]#gets all its values			
			#using vectorization 
			maskYes = (class_column == 'Yes')
			maskNo = (class_column == 'No')

			#array com apenas os valores discretos que correspondem a Yes
			attr_v_yes = attr_values[maskYes]
			#array com apenas os valores discretos que correspondem a No
			attr_v_no = attr_values[maskNo]
						
			#dicionario onde key=valor discreto, value=numero de vezes que corresponde a classe 'Yes'
			counts_yes = defaultdict(int)
			counts_yes = attr_v_yes.value_counts().to_dict()
			#dicionario onde key=valor discreto, value=numero de vezes que corresponde a classe 'No'
			counts_no = defaultdict(int)			
			counts_no = attr_v_no.value_counts().to_dict()	
		
			#calculates the accuracy of each attribute
			total_counts = 0
			n_correct_values = 0
			#store each discret value and its correspondent class value
			c_majority = {}

			#total_counts = sum(counts_yes.values())+sum(counts_no.values())
			for discret_value in m_discret_values[attr_name]: 
				if (not(counts_yes.has_key(discret_value))):
						countsYES = 0
				else:
					countsYES=counts_yes[discret_value]
				if (not(counts_no.has_key(discret_value))):
					countsNO = 0
				else:
					countsNO=counts_no[discret_value]
					
				if 	(countsYES >= countsNO and \
					self.produce_rules(attr_name,discret_value, 'Yes', class_label)):
					c_majority[discret_value]='Yes'
					n_correct_values += countsYES
					total_counts += countsYES + countsNO
					#list_tmp_rules = self.list_rules
				elif (countsYES < countsNO and \
					self.produce_rules(attr_name,discret_value, 'No', class_label)):
					n_correct_values += countsNO
					c_majority[discret_value]='No'
					total_counts += countsYES + countsNO
					#list_tmp_rules = self.list_rules


			list_c_majority.append((attr_name,c_majority))
			accuracy[attr_name] = float(n_correct_values)/float(total_counts)
		
		#list of attributes with maximum accuracy
		#GETS THE ATTRIBUTE WHOSE ALL RULES PRODUCES SMALLEST ERROR RATE
		max_acc = max(accuracy.values())
		chosen_attrs=[key for key, value in accuracy.items() if value == max_acc]
		#pick first attribute arbitraly if there are more than one with max_acc
		self.chosen_attr = chosen_attrs[0]
		print "CHOSEN ATTR",self.chosen_attr
		attr_values = self.data[self.chosen_attr]#gets all its values
		print "ACCURACY OF SET Of RULES: ",accuracy[self.chosen_attr]*100," %"

		'''
		#gets class value associated with a certain discret value
		l = [item for item in list_c_majority if item[0] == chosen_attr]
		#print l
		_, c_majority = l[0] 
		for discret_value in m_discret_values[chosen_attr]:
			#class_value = class_column[i]
			class_value = c_majority[discret_value]
			if(not(self.produce_rules(chosen_attr,discret_value, class_value, class_label))):
				print("following rule does not satisfy minimum number of instances {}".format(self.min_no_bucket))
				rule = "IF " + str(chosen_attr) + " = " + str(discret_value) + " THEN " + str(class_label) + " = " +  str(class_value)
				print("-------->",rule)
		'''
				
		print """
-----------------
oneR: finished!
-----------------
		"""
		
