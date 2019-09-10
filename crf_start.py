# encoding=utf8
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import pycrfsuite
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from collections import Counter
from itertools import chain
import scipy.stats
import nltk
#nltk.download('punkt')
import glob
import re

from termcolor import colored


dict_changer01 = { "bo": "p9", "gdyż": "p9", "aby": "p9", "acz": "p9", "aczkolwiek": "p9", "albowiem": "p9", "aż": "p9", "by": "p9", "dlatego": "p9", "czy": "p9",

#dopowiedzenia, skierowanie uwagi na treści
"chyba": "p9", "ewentualnie": "p9", "nawet": "p9", "prawdopodobnie": "p9", "przynajmniej": "p9", "raczej": "p9",
#spójniki do zdania podrzędnego https://sjp.pwn.pl/zasady/Wyrazy-wprowadzajace-zdanie-podrzedne;629774.html
"aby": "p1", "acz": "p1", "aczkolwiek": "p1", "albowiem": "p1", "azali": "p1", "aż": "p1", "ażeby": "p1", "bo": "p1", "boć": "p1", "bowiem": "p1", "by": "p1", "byle": "p1", "byleby": "p1", "chociaż": "p1", "chociażby": "p1", "choć": "p1", "choćby": "p1", "chybaby": "p1", "co": "p1", "cokolwiek": "p1", "czy": "p1", "czyj": "p1", "dlaczego": "p1", "dlatego": "p1", "dokąd": "p1", "dokądkolwiek": "p1", "dopiero": "p1", "dopiero gdy": "p1", "dopóki": "p1", "gdy": "p1", "gdyby": "p1", "gdyż": "p1", "gdzie": "p1", "gdziekolwiek": "p1", "ile": "p1", "ilekolwiek": "p1", "ilekroć": "p1", "im": "p1", "iż": "p1", "iżby": "p1", "jak": "p1", "jakby": "p1", "jaki": "p1", "jakikolwiek": "p1", "jakkolwiek": "p1", "jako": "p1", "jakoby": "p1", "jakżeby": "p1", "jeśli": "p1", "jeśliby": "p1", "jeżeli": "p1", "jeżeliby": "p1", "kędy": "p1", "kiedy": "p1", "kiedykolwiek": "p1", "kiedyż": "p1", "kim": "p1", "kogo": "p1", "komu": "p1", "kto": "p1", "ktokolwiek": "p1", "którędy": "p1", "który": "p1", "ledwie": "p1", "ledwo": "p1", "niech": "p1", "nim": "p1", "odkąd": "p1", "ponieważ": "p1", "póki": "p1", "skąd": "p1", "skądkolwiek": "p1", "skoro": "p1", "zaledwie": "p1", "zanim": "p1", "że": "p1", "żeby": "p1",
#}

#https://sjp.pwn.pl/zasady/370-90-D-1-Zdania-wspolrzedne-polaczone-spojnikami-przeciwstawnymi-wynikowymi-synonimicznymi;629784.html
#dict_changer02 = { #spójniki przeciwstawne 
"a": "p2", "ale": "p2", "aliści": "p2", "inaczej": "p2", "jednak": "p2", "jednakże": "p2", "jedynie": "p2", "lecz": "p2", "natomiast": "p2", "przecież": "p2", "raczej": "p2", "tylko": "p2", "tymczasem": "p2", "wszakże": "p2", "zaś": "p2",
#}

#dict_changer03 = { #spójniki wynikowe
"więc": "p3", "dlatego": "p3","toteż": "p3", "to": "p3", "zatem": "p3", "stąd": "p3", "przeto": "p3", "tedy": "p3", "czyli": "p3",
#}

#https://sjp.pwn.pl/zasady/371-90-D-2-Zdania-wspolrzedne-polaczone-spojnikami-lacznymi-rozlacznymi-wylaczajacymi;629785.html
#dict_changer04 = { 
"i": "p4", "oraz": "p4", "tudzież": "p4", "lub": "p4", "albo": "p4", "bądź": "p4", "czy": "p4", "ani": "p4", "ni": "p4",
#}

#wymienianie po znaku :
#dict_changer05 = { 
":": "p5",
#}

#https://sjp.pwn.pl/zasady/381-90-H-3-Porownania-paralelne-o-konstrukcji-tak-jak-rownie-jak-taki-jaki-tyle-co;629799.html
#dict_changer06 = { 
 "tak": "p6", "jak": "p6", "równie": "p6", "jak": "p6", "taki": "p6", "jaki": "p6", "tyle": "p6", "co": "p6",
#}

#https://sjp.pwn.pl/zasady/391-90-J-7-Przecinek-po-wyrazach-wyrazajacych-okrzyk;629811.html
#dict_changer07 = {  
"ach": "p7", "halo": "p7", "hej": "p7", "ho": "p7", "oj": "p7"
}

two_words_dict = {
	"dlatego": ["iż", "że"], 
	"dopiero": ["gdy"], 
	"podobnie": ["jak"], 
	"potem": ["gdy"], 
	"tak": ["aby", "by", "iż", "jak", "jakby", "że", "żeby"],
	"taki": ["jak", "sam"], 
	"tam": ["gdzie", "skąd"], 
	"teraz": ["gdy"], 
	"to": ["co"], 
	"tym": ["bardziej"], 
	"miarę": ["jak"], 
	"wprzód": ["nim"], 
	"wtedy": ["gdy"]
}

def punctation_test(test_length):
	counter = 0
	while counter < test_length:
		print(X_test[counter][1][11:],y_pred[counter]) # 11 to not see word.lower=
		counter += 1
	

def load_data_train():
	#files = glob.glob("endbooks/train/*.tag")
	files = glob.glob("endbooks/train_only/*.tag") #prus_placowka_1935.tag") # train only on sentences with punctation
	return files

def load_data_test():
	files = glob.glob("endbooks/test/*.tag")
	return files

def load_data_train_noc(): # no comma
	#files = glob.glob("endbooks/train_noc/*.tag")
	files = glob.glob("endbooks/train_noc_only/*.tag")#prus_placowka_1935.tag") # train only on sentences with punctation
	return files

def load_data_test_noc():
	files = glob.glob("endbooks/test_noc/*.tag")
	return files

def tagging(load_data_noc, load_data, dictionary_changer):
	files = load_data_noc
	files2 = load_data
	labeled = []
	for book_noc, book in zip(files, files2):
		root = ET.parse(book_noc).getroot()
		root2 = ET.parse(book).getroot()

		iter1 = root.iterfind('chunk/sentence/tok')
		next(iter1)
		iter2 = root2.iterfind('chunk/sentence/tok')
		next(iter2)
		#c = 0 #maciek test
		for elem, elem2 in zip(root.iterfind('chunk/sentence/tok'), root2.iterfind('chunk/sentence/tok')):
			try:
				nextElem = next(iter1)
				nextElem2 = next(iter2)
				orth = elem.find('orth').text
				#dict_changer = dictionary_changer # zawsze wstawiamy przecinek
				#replace_all(orth, dict_changer)
				tag = elem.find('lex/ctag').text
				tag2 =  tag.split(':')
				if nextElem2.find('orth').text == ',':
					labeled.append((orth, tag2[0], 'P'))
					next(iter2) # ignore punctation
				else:
					labeled.append((orth, tag2[0], 'N'))
				# print(orth, tag,"Tylko pierwszy:", tag2[0])
				#print(labeled[c][0:3])  maciek test
				#c += 1
			except StopIteration:
				break
	#for l in labeled: print(l)
	return labeled

def word_comma_rule(word, dict_changer):
	if word in dict_changer:
		return True
	else:
		return False

def two_word_comma_rule(word, word1, two_words_dict):
	return word in two_words_dict and word1 in two_words_dict[word]


def word2feature(labeled, i):
	word = labeled[i][0]
	postag = labeled[i][1]


	features = {
		'bias': 0.3 ,
		#'word.lower()': word.lower(),

		'word.comma.rule01': word_comma_rule(word, dict_changer01),
		#'word.comma.rule02': word_comma_rule(word, dict_changer02),
		#'word.comma.rule03': word_comma_rule(word, dict_changer03),
		#'word.comma.rule04': word_comma_rule(word, dict_changer04),
		#'word.comma.rule05': word_comma_rule(word, dict_changer05),
		#'word.comma.rule06': word_comma_rule(word, dict_changer06),
		#'word.comma.rule07': word_comma_rule(word, dict_changer07),
		'postag=': postag,
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),
		'imieslow': imieslow(word),
	}
	if word_comma_rule(word, dict_changer01) == True:
		features.update({'word.lower.inter': word.lower()})

	if i > 0 and not(labeled[i-1][0] == '.'):
		word1 = labeled[i-1][0]
		postag1 = labeled[i-1][1]
		features.update({
			#'-1:word.lower()': word1.lower(),

			'-1:word.comma.rule01': word_comma_rule(word1, dict_changer01),

			#'-1:word.comma.rule02': word_comma_rule(word1, dict_changer02),
			#'-1:word.comma.rule03': word_comma_rule(word1, dict_changer03),
			#'-1:word.comma.rule04': word_comma_rule(word1, dict_changer04),
			#'-1:word.comma.rule05': word_comma_rule(word1, dict_changer05),
			#'-1:word.comma.rule06': word_comma_rule(word1, dict_changer06),
			#'-1:word.comma.rule07': word_comma_rule(word1, dict_changer07),
			'-1:word.istitle()': word1.istitle(),
			'-1:word.isupper()': word1.isupper(),
			'-1:word.isdigit()': word1.isdigit(),
			'-1:postag': postag1,
			'-1:imieslow': imieslow(word1),
		})
		if word_comma_rule(word1, dict_changer01) == True:
			features.update({'-1:word.lower.inter': word1.lower()})
	else:
		features['BOS'] = True

	if i > 1 and not(labeled[i-2][0] == '.' or labeled[i-1][0] == '.' ):
		word1 = labeled[i-2][0]
		postag1 = labeled[i-2][1]
		features.update({
			#'-2:word.lower()': word1.lower(),

			'-2:word.istitle()': word1.istitle(),
			'-2:word.isupper()': word1.isupper(),
			'-2:word.isdigit()': word1.isdigit(),
			'-2:postag': postag1,
			'-2:imieslow': imieslow(word1),
		})
	else:
		features['BOS'] = True

	if i < len(labeled)-1 and not(labeled[i+1][0] == '.'):
		word1 = labeled[i+1][0]
		postag1 = labeled[i+1][1]
		features.update({
			#'+1:word.lower()': word1.lower(),

			'+1:word.comma.rule01': word_comma_rule(word1, dict_changer01),

			#'+1:word.comma.rule02': word_comma_rule(word1, dict_changer02),
			#'+1:word.comma.rule03': word_comma_rule(word1, dict_changer03),
			#'+1:word.comma.rule04': word_comma_rule(word1, dict_changer04),
			#'+1:word.comma.rule05': word_comma_rule(word1, dict_changer05),
			#'+1:word.comma.rule06': word_comma_rule(word1, dict_changer06),
			#'+1:word.comma.rule07': word_comma_rule(word1, dict_changer07),
			'+1:word.istitle()': word1.istitle(),
			'+1:word.isupper()': word1.isupper(),
			'+1:word.isdigit()': word1.isdigit(),
			'+1:postag': postag1,
			'+1:imieslow': imieslow(word1),
			'+1word.comma.2_word_rule': two_word_comma_rule(word, word1, two_words_dict)
		})

		if word_comma_rule(word1, dict_changer01) == True:
			features.update({'+1:word.lower.inter': word1.lower()})
		if two_word_comma_rule(word, word1, two_words_dict) == True:
			features.update({'+1:word.lower.two_inter': word1.lower(), 'word.lower.two_inter': word.lower()})
	else:
		features['EOS'] = True

	if i < len(labeled)-2 and not(labeled[i+1][0] == '.' or labeled[i+2][0] == '.'):
		word1 = labeled[i+2][0]
		postag1 = labeled[i+2][1]
		features.update({
			#'+2:word.lower()': word1.lower(),

			'+2:word.istitle()': word1.istitle(),
			'+2:word.isupper()': word1.isupper(),
			'+2:word.isdigit()': word1.isdigit(),
			'+2:postag': postag1,
			'+2:imieslow': imieslow(word1),
		})
	else:
		features['EOS'] = True
	return features

# labeled = labeled()
# print(word2feature(labeled, 0))

# A function for extracting features in documents
def extract_features(doc):
	return [word2feature(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
	return [label for (token, postag, label) in doc]

def get_orth(doc):
	return [token for (token, postag, label) in doc]

def replace_all(text, dic):
	for i, j in dic.items():
		text = text.replace(i, j)
	return text

def imieslow(word): #https://sjp.pwn.pl/zasady/367-90-B-4-Imieslow-zakonczony-na-ac-lszy-wszy;629779.html
	if (word[-4:] == "łszy" or word[-4:] == "wszy" or word[-2:] == "ąc"):
		imi = True # Imiesłów zakończony na -ąc, -łszy, -wszy
	else:
		imi = False
	return imi

def classification_report_all(labels, y_pred, y_test):
	sorted_labels = sorted(
		labels, 
		key=lambda name: (name[1:], name[0])
	)
	print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
	return metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3, output_dict = True)

# not working
def optimize_hyperparameters(X_train, y_train, X_test, y_test):
	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs', 
		max_iterations=20, 
		all_possible_transitions=True
	)
	params_space = {
		'c1': scipy.stats.expon(scale=0.5),
		'c2': scipy.stats.expon(scale=0.05)
	}
	
# use the same metric for evaluation
	# f1_scorer = make_scorer(metrics.flat_f1_score, 
    #                     average='weighted', labels=labels)

# search
	rs = RandomizedSearchCV(crf,
							param_distributions=params_space, 
							cv=2, #3
							verbose=1, 
							n_jobs=-1, 
							n_iter=5) 
							#scoring=f1_scorer)
	rs.fit(X_train, y_train)
	print(rs.cv_results_['params'])
	# print('best params:', rs.best_params_)
	# print('best CV score:', rs.best_score_)
	# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
	# crf = rs.best_estimator_
	# y_pred = crf.predict(X_test)
	# sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
	# print(metrics.flat_classification_report(
	# 	y_test, y_pred, labels=sorted_labels, digits=3
	# ))

def print_transitions(trans_features):
	for (label_from, label_to), weight in trans_features:
		print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
	for (attr, label), weight in state_features:
		print("%0.6f %-8s %s" % (weight, label, attr)) 

def show_punctation(X_test_orth, y_pred, first_word_number, last_word_number):
	zdanie = []	
	for counter in range(first_word_number, last_word_number):	
		raw_output = '{:<20s} {:<6s} {:<1s} {:<6s} {:<1s}'.format(X_test_orth[0][counter] , "pred: " , y_pred[0][counter] , "good: " , y_test[0][counter])
		if y_pred[0][counter] == y_test[0][counter]: 
			print(colored(raw_output, 'green')) #green good
		elif y_pred[0][counter] == 'P':
			print(colored(raw_output, 'yellow'))# yellow if pred:P but good:N
		else:
			print(colored(raw_output, 'red')) #red if pred:N but good:P

#not used
def cross_val(model):
	print("Cross validation")
	scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
	labels = [l for l in tagging(load_data_train_noc(), load_data_train(), dict_changer01) if l[0] != ',']
	X = [[x] for x in extract_features(labels)]
	y = [[x] for x in get_labels(labels)]
	scores = cross_validate(model, X, y, cv = 2, scoring=scoring, return_train_score=False)
	averages = {}
	for s in scores:
		averages[s] = np.mean(scores[s])
	print('average: ', averages)

def kFold(model, folds):
	print('KFold')
	labels = [l for l in tagging(load_data_train_noc(), load_data_train(), dict_changer01) if l[0] != ',']
	X = np.array([[x] for x in extract_features(labels)])
	y = np.array([[x] for x in get_labels(labels)])
	score_dict = {'N': {'precision':0, 'recall':0, 'f1-score':0}, 'P': {'precision':0, 'recall':0, 'f1-score':0},
	'weighted avg': {'precision':0, 'recall':0, 'f1-score':0}}
	kf = KFold(n_splits=folds)
	print(kf)
	for train_index, test_index in kf.split(X):
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		model.fit(X_train, y_train)
		labi = list(crf.classes_)
		y_pred = crf.predict(X_test)
		score = classification_report_all(labi, y_pred, y_test)
		for s in score_dict:
			score_dict[s]['precision'] += score[s]['precision']
			score_dict[s]['recall'] += score[s]['recall']
			score_dict[s]['f1-score'] += score[s]['f1-score']

	score_dict['N']['precision'] /= folds
	score_dict['N']['recall'] /= folds
	score_dict['N']['f1-score'] /= folds
	score_dict['P']['precision'] /= folds
	score_dict['P']['recall'] /= folds
	score_dict['P']['f1-score'] /= folds
	score_dict['weighted avg']['precision'] /= folds
	score_dict['weighted avg']['recall'] /= folds
	score_dict['weighted avg']['f1-score'] /= folds
	print(score_dict)

if __name__ == '__main__':
	
	labels_train = [l for l in tagging(load_data_train_noc(), load_data_train(), dict_changer01) if l[0] != ',']
	#for l in labels_train[-15:]: print(l) maciek test
	X_train = np.array([[x] for x in extract_features(labels_train)])
	y_train = np.array([[x] for x in get_labels(labels_train)])

	labels_test = [l for l in tagging(load_data_test_noc(), load_data_test(), dict_changer01) if l[0] != ','] 
	X_test = np.array([[x] for x in extract_features(labels_test)])
	y_test = np.array([[x] for x in get_labels(labels_test)])
	X_test_orth = [get_orth(labels_test)]
	
	# crf = sklearn_crfsuite.CRF(
	# 	algorithm = 'lbfgs',#'lbfgs', #best pa 15% f1 N 65 / arow 12% N 89
	# 	c1 = 0.01,
	# 	c2 = 0.11,

	# 	max_iterations = 100,
	# 	verbose = True,
	# 	all_possible_transitions = False
	# )

	# crf.fit(X_train, y_train)

	# labi = list(crf.classes_)
	
	# y_pred = crf.predict(X_test)

	# show_punctation(X_test_orth, y_pred, 0, 300)

	# classification_report_all(labi, y_pred, y_test)

	# kFold(crf, 5)


	# print("Top likely transitions:")
	# print_transitions(Counter(crf.transition_features_).most_common(20))
	# print("\nTop unlikely transitions:")
	# print_transitions(Counter(crf.transition_features_).most_common()[-20:])   

	# print("Top positive:")
	# print_state_features(Counter(crf.state_features_).most_common(30))
	# print("\nTop negative:")
	# print_state_features(Counter(crf.state_features_).most_common()[-30:])
	print("Optimizer")
	optimize_hyperparameters(X_train, y_train, X_test, y_test)#not working

	#punctation_test(40) # len(X_test) print everything

#klasa przecinki brana pod uwagę jako word.lower()
#kroswalidacja zbiór testowy i treningowy
#klasy dwu wyrazowe
#dynamiczne okno
#typy błędów przy danych cechach

