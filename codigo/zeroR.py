#!/usr/bin/python
import sys
from collections import defaultdict
#import oneR
#import zeroR

class ZeroR:
	"""classe que usa classificador zeroR

	Ignores the influence of all attribute values for classification,
	the produced rule simply predicts the majority class 

	"""
	#construtor
	def __init__(self,data_reader,class_values):
		self.list_rules = []
		self.data = data_reader
		self.class_values = class_values
		self.class_counts = defaultdict(int)
	#generate set of rules in a list
	def produce_rules(self,class_name, class_label):
		rule = class_label + " -> " +  class_name
		self.list_rules.append(rule)
		
	#return set of rules
	def get_selected_rules(self):
		return self.list_rules
	#return class with highest frequency and its frequency
	def get_class_max_count(self):
		max_ = max(self.class_counts)
		total_counts = sum(self.class_counts.values()) 
		freq = float(self.class_counts[max_])/float(total_counts)
		return self.class_values[max_], freq
	#run the algorithm
	def begin_zeroR(self):
		firstline = True
		for row in self.data:
			#get label names (header line)
			if firstline:
				class_label = row[-1]
				firstline = False
				continue
			else:
				n_class_values = len(self.class_values)
				found_class = False
				#counts each class frequency
				for i in range(n_class_values):
					if row[-1]== self.class_values[i]:
						self.class_counts[i] += 1
						found_class = True
						break
				#exit if there is a non registered class
				if not found_class:
					sys.exit("Non existant class Error")
					
		#compare which class has more frequency
		#default: 50% of Yes => rule of Yes
		maxcount_class, freq = self.get_class_max_count()
		self.produce_rules(maxcount_class,class_label)
		print "ACCURACY: ",freq*100," %"
		print """
-----------------
zeroR: finished!
-----------------
		"""
		
