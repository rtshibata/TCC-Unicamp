# Modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NormalEquations:

	#construtor
	def __init__(self,data_reader,class_values):
		self.data = data_reader

		self.class_values = class_values
		self.theta = pd.DataFrame() #empty
		
'''
path_2_train = '/home/ec2010/ra082674/TCC/tcc-unicamp/TCC_RenatoShibata/base/CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'
path_2_test = '/home/ec2010/ra082674/TCC/tcc-unicamp/TCC_RenatoShibata/base/2016-2017season.csv'

if len(sys.argv) != 3: 
	print "entrar com 2 string como argumento: path do csv de training(entre aspas) e o path do csv de testing\n"
	exit(1)
path_2_train = sys.argv[1] #'../base/CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'

path_2_test = sys.argv[2]# '/home/renato/Dropbox/TCC_RenatoShibata/base/2016-2017season.csv'

# Loading data set
X = pd.read_csv(path_2_train)
'''
def begin_alg(self):
	header_line = self.data.columns

	target = header_line[-1]
	y = self.data[target]

	X = self.data.iloc[:,:-1]
	m,n = X.shape

	theta = []
	theta_0 =np.ones((m, 1))
	X.insert(loc=0, column='theta_0', value=theta_0)
	#print X
	
	#print y
    # Normal Equation:
    # theta = inv(X^T * X) * X^T * y

    # For convenience I create a new, tranposed X matrix
	X_transpose = X.transpose()

    # Calculating theta
	XtX = X_transpose.dot(X)
	theta = pd.DataFrame(np.linalg.inv(XtX.values),XtX.columns,XtX.index)
	theta = theta.dot(X_transpose)
	theta = theta.dot(y)
	self.theta = theta

def get_theta(self):
	print self.theta
	return self.theta

'''
def begin_test(self):
	X = self.data.iloc[:,:-1]
	m,n = X.shape
	#mean squared error
	for i in range(n):
		X_i = X.iloc[i,:]
		y_f = theta.dot(X_i)
		mse = y_f - y_test.iloc[i]
		mse = float(mse ** 2)/float(n)

	print mse
'''





