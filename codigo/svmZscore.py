# Modules
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVM:

	#construtor
	def __init__(self,data_reader,class_values):
		self.data = data_reader
		self.clf = pd.DataFrame() #empty
		self.class_values = class_values
		self.param = pd.DataFrame() #empty

		
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
		'''
		theta = []

		theta_0 =np.ones((m, 1))
		X.insert(loc=0, column='theta_0', value=theta_0)
		'''			
		self.clf = SVC()
		self.clf.fit(X,y)
		self.param = self.clf.get_params()
		score = self.clf.score(X,y)
		print("ACCURACY: ",score*100, " %")

	def get_param(self):
		return self.param


	def begin_test(self,data_test):
		header_line = data_test.columns
		playername_column = data_test.index
		class_column =data_test[header_line[-1]]

		target = header_line[-1]
		y = data_test[target]

		X = data_test.iloc[:,:-1]
		m,n = X.shape

		scaler = StandardScaler()
		std_matrix = scaler.fit_transform(X)
		X = pd.DataFrame(data=std_matrix,columns=header_line[:-1])	

		clf = SVC()
		clf.fit(X,y)
		l_classes = clf.predict(X)
		score = clf.score(X,y)
	
		#print maskCorrect

	 	maskCorrect = (l_classes == y)
		#predictYes = pd.DataFrame(data=(l_classes == 'Yes'),index=playername_column)
		#print predictYes
		#maskTP = (predictYes == 'True') & (l_classes == y)

		maskTP = (l_classes == 'Yes') & (l_classes == y)
		maskTN = (l_classes == 'No') & (l_classes == y)
		maskFP = (l_classes == 'Yes') & (l_classes != y)
		maskFN = (l_classes == 'No') & (l_classes != y)

		n_correct_players = maskCorrect.value_counts().loc[[True]][1] if True in maskCorrect.value_counts().index else 0
		n_true_positive = maskTP.value_counts().ix[1,1] if True in maskTP.value_counts().index else 0
		n_false_positive = maskFP.value_counts().ix[1,1] if True in maskFP.value_counts().index else 0
		n_false_negative = maskFN.value_counts().ix[1,1] if True in maskFN.value_counts().index else 0
		n_true_negative = maskTN.value_counts().ix[1,1] if True in maskTN.value_counts().index else 0

		print("Accuracy ",score*100, " %")
		print("Correct predicted Instances: ",n_correct_players)
		print("True positive: {},  False Negative: {}".format(n_true_positive,n_false_negative))
		print("False positive: {}, True Negative: {}".format(n_false_positive,n_true_negative))

		correct_players = maskCorrect[maskCorrect==True]
		for player_name in correct_players.index:
			print (class_column.loc[player_name]+" "+str(player_name))




		return l_classes, score






