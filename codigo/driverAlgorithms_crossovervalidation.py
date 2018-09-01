#!/usr/bin/python
import sys
import csv
import oneR_crossovervalidation
import zeroR
#import discretize
import pandas as pd

path_2_csv_home = "../base"
path_2_csv_ic ="/home/ec2010/ra082674/TCC/data"

if len(sys.argv) != 2: 
	print "entrar com 1 string como argumento: path do arquivo csv(entre aspas)\n"
	exit(1)

path_2_csv = sys.argv[1] #'../base/CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'
#'/home/ec2010/ra082674/TCC/TCC_RenatoShibata/base/CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'



#csv_name = 'CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'
#path_2_csv = path_2_csv_home + "/" + csv_name

with open(path_2_csv,'rb') as csv_playersdata:
	data_reader = csv.reader(csv_playersdata,delimiter=',')
	print """
----------------
loading data...
----------------
	"""
	#Ballon D'or nominee values
	class_values = ["Yes","No"]
	inst_zeroR = zeroR.ZeroR(data_reader, class_values)
	print ("ZeroR algorithm...")		
	inst_zeroR.begin_zeroR()
	for rule in inst_zeroR.list_rules:
		print "set of rules:  "
		print "* ",rule
		print "\n"
	
data_reader = pd.read_csv(path_2_csv)

#minimal number of instances per bucket
min_no_bucket = 6    #minimun number of instances to satisfy a rule in a leaf
n_discret_bins = 10  #number of bins that will be equally divided the interval of values of a certain attribute
bin_range_precision = 4 #number of decimal digits to use when discretizing to bins

# k-fold cross over validation # separating training and testing data 
k_fold_crossover=10
#for cross_over_i in range(k):
#	X_train, X_test = train_test_split(data_reader,test_size = float(1)/float(k),shuffle=True)
inst_oneR = oneR_crossovervalidation.OneR(data_reader,class_values,min_no_bucket,n_discret_bins,bin_range_precision,k_fold_crossover)
print ("OneR algorithm...")		
inst_oneR.begin_oneR()
#for rule in inst_oneR.list_rules:
#	print "set of rules: "
#	print rule
