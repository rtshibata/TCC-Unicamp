#!/usr/bin/python
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split

class OneR:
	"""classe que usa classificador OneR

	It generates one rule for each predictor attribute in the data,
then selects only 1 attribute, whose set of rules gives the smallest total error 

	PS: In this implementation, it assumes there are only non-negative values in the dataset

	PS2: In this implementation it was not implemented in a way that the rules generated respects the min_no_bucket

	"""
	#construtor
	def __init__(self,data_reader,class_values,min_no_bucket,n_discret_bins,bin_range_precision,k_fold_crossover):
		self.list_rules = []
		self.data = data_reader
		self.class_values = class_values
		self.min_no_bucket = min_no_bucket
		self.n_discret_bins = n_discret_bins	
		self.bin_range_precision = bin_range_precision
		self.k_fold_crossover = k_fold_crossover	

	#generate set of rules in a list
	def produce_rules(self,attr_label,attr_value, class_value, class_label):
		rule = "IF " + str(attr_label) + " = " + str(attr_value) + " THEN " + str(class_label) + " = " +  str(class_value)
		self.list_rules.append(rule)

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
			delta = (float(max_val) - float(min_val))/n
			#print delta,n
			interval_init = min_val
			discret_values = []
			for i in range(n):#save discret values in array
				interval_end = np.where(i==n-1,float("inf"),interval_init+delta)
				interval_init = float("{0:.{prec}f}".format(interval_init, prec=self.bin_range_precision))
				interval_end = float("{0:.{prec}f}".format(interval_end, prec=self.bin_range_precision))

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

	def testing_rules(self,X_test,chosen_attr,discret_value,class_value,class_label):
		class_column = X_test[class_label]#gets all values related to class column
		attr_values = X_test[chosen_attr]

		#testar em quais instancias de X_test as condicoes da regraeh verdadeira
		maskCorrect = (class_column == class_value)
		attr_v_correct = attr_values[maskCorrect]

		#calcular porcentagem de acertos  
		accuracy = float(len(attr_v_correct))/float(len(attr_values))	

		return accuracy, len(attr_v_correct)
		

	#run the algorithm
	def begin_oneR(self):
		m_discret_values = self.discretize_data()
		#print m_discret_values
		#header_line = next(self.data)
		header_line = self.data.columns
		class_column = self.data[header_line[-1]]#gets all values related to class column
		
		for cross_over_i in range(self.k_fold_crossover):
			print "------------------------------------------" 
			print "Starting ",cross_over_i," -th cross over validation" 
			print "------------------------------------------"
			l_all_attr =[]
			attr_names = []
			accuracy = {}
			list_c_majority =[]
			# k-fold cross over validation # separating training and testing data 
			X_train, X_test = train_test_split(self.data,test_size = float(1)/float(self.k_fold_crossover),shuffle=True)
			#estimate the error related to each attribute
			for attr_name in header_line:
				if attr_name == header_line[-1]:
					break
				attr_values = X_train[attr_name]#gets all its values			
				#using vectorization 
				maskYes = (class_column == 'Yes')
				maskNo = (class_column == 'No')

				#array com apenas os valores discretos que correspondem a Yes
				attr_v_yes = attr_values[maskYes]
				#array com apenas os valores discretos que correspondem a No
				attr_v_no = attr_values[maskNo]
						
				#dicionario onde key=valor discreto, value=numero de vezes que corresponde a classe 'Yes'
				counts_yes = defaultdict(int)
				counts_yes = attr_v_yes.value_counts().to_dict(into=counts_yes)
				#dicionario onde key=valor discreto, value=numero de vezes que corresponde a classe 'No'
				counts_no = defaultdict(int)			
				counts_no = attr_v_no.value_counts().to_dict(into=counts_no)	
		
				#calculates the accuracy of each attribute
				total_counts = 0
				n_correct_values = 0
				#store each discret value and its correspondent class value
				c_majority = {}

				#print self.data[attr_name]
				#print "counts yes", counts_yes
				#print "counts no", counts_no
				
					#print discret_value,counts_yes[discret_value],counts_no[discret_value]
				total_counts = sum(counts_yes.values())+sum(counts_no.values())
				for discret_value in m_discret_values[attr_name]: 
					#total_counts += counts_yes[discret_value]+counts_no[discret_value]
					if counts_yes[discret_value] >= counts_no[discret_value]:
						c_majority[discret_value]='Yes'
						n_correct_values += counts_yes[discret_value]
					else:
						n_correct_values += counts_no[discret_value]
						c_majority[discret_value]='No'
				list_c_majority.append((attr_name,c_majority))
				accuracy[attr_name] = float(n_correct_values)/float(total_counts)
				#print accuracy
				#print list_c_majority		
				#print n_correct_values,total_counts
				print attr_name,accuracy[attr_name]
				'''#attr_names.insert(0,attr_values[0]) #gets the attribute name
				attr_dict = defaultdict(int)
				i=0	#first element value in the class column

				for discret_value in attr_values:
					#assuming there are ONLY 2 class values
					if class_column[i] == self.class_values[0]:
						attr_dict[discret_value] += 1#+1 for Yes
						i += 1 #next element			
					elif class_column[i] == self.class_values[1]:
						attr_dict[discret_value] -= 1#-1 for No 
						i += 1 #next element
					#order in crescent order by the discret inverval values
					list_attr_errors = sorted(attr_dict.items())
					#print list_attr_errors
					l_all_attr.append(list_attr_errors) 
		
			chosen_attr, min_error = self.get_attr_small_error(l_all_attr,header_line)
			'''			
			#list of attributes with maximum accuracy
			max_acc = max(accuracy.values())
			chosen_attrs=[key for key, value in accuracy.items() if value == max_acc]
			#pick first attribute arbitraly if there are more than one with max_acc
			chosen_attr = chosen_attrs[0]
			print "CHOSEN ATTR",chosen_attr
			#produce set of rules
			class_label = header_line[-1]
			#print class_label
			#attr_values = X_train[chosen_attr]#gets all its values

			l = [item for item in list_c_majority if item[0] == chosen_attr]
			#print l
			_, c_majority = l[0] 
			total_correct_instances = 0
			for discret_value in m_discret_values[chosen_attr]:
				#class_value = class_column[i]
				class_value = c_majority[discret_value]
				self.produce_rules(chosen_attr,discret_value, class_value, class_label)
				print "using testing data....."
				accuracy, n_correct_instances = self.testing_rules(X_test,chosen_attr,discret_value,class_value,class_label)
				print "RULE ACCURACY: ",accuracy*100," %"
				total_correct_instances += n_correct_instances

			for rule in self.list_rules:
				print rule

			n_X_test = X_test.shape[0]
			#total number of correct instances in all rules / k times the number of instances in the training data
			total_accuracy = float(total_correct_instances)/float(self.k_fold_crossover*n_X_test)
			print "SET OF RULES AVERAGE ACCURACY: ",total_accuracy*100," %"
			#erase all previous set of rules			
			self.list_rules = []

		print "------------------------------------------" 
		print self.k_fold_crossover,""" - fold crossover validation
 
-----------------
oneR: finished!
-----------------
		"""
		
