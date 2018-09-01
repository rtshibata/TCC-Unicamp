import weka.core.jvm as jvm
import weka.core.converters as Loader

path_2_csv = '../base/CSV2012-2013_ALL_PLAYERS _No_app_no goalkeeper_ No MOM.csv'

data = converters.load_any_file(path_2_csv)
data.class_is_last()

print(data)
