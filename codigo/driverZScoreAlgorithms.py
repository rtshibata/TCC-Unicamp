#!/usr/bin/python
from __future__ import with_statement
import sys

#sys.path.append('/home/renato/weka-3-8-2/weka.jar')
from sklearn.preprocessing import StandardScaler
import csv

import oneR
import zeroR
import decisiontreeNoZScoreKBest
import log_regression
import svm

import testingSet
import pandas as pd

'''
import java.io.FileReader as FileReader
import weka.core.Instances as Instances
import weka.classifiers.trees.J48 as J48

from javabridge import JWrapper
import javabridge
import weka.core.jvm as jvm

jvm.start()
'''

path_2_csv_home = "../base"
path_2_csv_ic ="/home/ec2010/ra082674/TCC/TCC_RenatoShibata/base"


if len(sys.argv) != 3: 
	print "entrar com 2 string como argumento: path do csv de training(entre aspas) e o path do csv de testing\n"
	exit(1)

path_2_csv = sys.argv[1] #'../base/CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'
#  '/home/ec2010/ra082674/TCC/tcc-unicamp/TCC_RenatoShibata/base/CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'

path_2_testcsv = sys.argv[2]# '/home/renato/Dropbox/TCC_RenatoShibata/base/2016-2017season.csv'
#  '/home/ec2010/ra082674/TCC/tcc-unicamp/TCC_RenatoShibata/base/2016-2017season.csv'

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
		print "set of rules: "
		print "* ",rule
		print "\n"
	
data_reader = pd.read_csv(path_2_csv)
data_test = pd.read_csv(path_2_testcsv,index_col=0)

header_line = data_reader.columns
target = header_line[-1]
y = data_reader[target]

#Z score
print("Scaling features...Z score")
print(".................")
X = data_reader.iloc[:,:-1]
scaler = StandardScaler()
std_matrix = scaler.fit_transform(X)
data_reader = pd.DataFrame(data=std_matrix,columns=header_line[:-1])	
data_reader["Ballond'OrNominee"] = y


#data_reader
#minimal number of instances per bucket
min_no_bucket = 0    #minimun number of instances to satisfy a rule in a leaf
n_discret_bins = 10  #number of bins that will be equally divided the interval of values of a certain attribute
bin_range_precision = 7 #number of decimal digits to use when discretizing to bins
inst_oneR = oneR.OneR(data_reader.copy(),class_values,min_no_bucket,n_discret_bins,bin_range_precision)
print """
----------------
	"""
print ("OneR algorithm...")		
inst_oneR.begin_oneR()
list_rulesOneR = inst_oneR.get_final_rules()
for rule in list_rulesOneR:
	print "set of rules: "
	print rule



inst_logR = log_regression.LogisticRegression(data_reader.copy(),class_values)
print """
----------------
	"""
print ("Logistic Regression algorithm...")		
inst_logR.begin_alg()
param_classif = inst_logR.get_param() 
print param_classif

inst_dtree = decisiontreeNoZScoreKBest.DecisionTree(data_reader.copy(),class_values)
print """
----------------
	"""
print ("Decision Tree Classifier algorithm...")		
inst_dtree.begin_alg()
param_classif = inst_dtree.get_param() 
print param_classif

inst_svm = svm.SVM(data_reader.copy(),class_values)
print """
----------------
	"""
print ("Support Vector Machines(SVM) algorithm...")
inst_svm.begin_alg()
param_weight = inst_svm.get_param() 
print param_classif

'''
inst_NormalEq = normal_equations.NormalEquations(data_reader,class_values)
inst_NormalEq.begin_alg()
inst_NormalEq.get_theta()


file = FileReader(path_2_csv)
data = Instances(file)
data.setClassIndex(data.numAttributes() - 1)

c4_5= J48()
j48.buildClassifier(data)
print j48
'''

print """

-------
testing data...ZeroR
-------
"""

inst_test1 = testingSet.TestingSet(data_test,inst_zeroR.list_rules)
inst_test1.begin_testingZeroR()



print """
-------
testing data...OneR
-------
"""
inst_test2 = testingSet.TestingSet(data_test,list_rulesOneR)
inst_test2.begin_testingOneR()



print """
-------
testing data...LogisticRegression
-------
"""

predict_class, score =inst_logR.begin_test(data_test)
#print score
print """
-------
testing data...Decision Tree
-------
"""
predict_class, score = inst_dtree.begin_test(data_test)



print """
-------
testing data...SVM(SVC)
-------
"""
predict_class, score =inst_svm.begin_test(data_test)



