from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
import numpy
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')

data = {"text":[], "class":[]}

f = open("SMSSpamCollection.txt", "r")
reader=csv.reader(f,delimiter='\t')
for target, value in reader:
	tokens = []	
	token = tokenizer.tokenize(value)
	for i in token:
		if i not in stop:
			tokens.append(i)

	value = " ".join(tokens).decode('cp1252', 'ignore')
	data["text"].append(value)
	data["class"].append(target)

f.close()

length = len(data["text"])
sample = random.sample(range(0, length), length)
data["text"] = [data["text"][i] for i in sample]
data["class"] = [data["class"][i] for i in sample]

pipeline = Pipeline([
	('vectorizer',  CountVectorizer(ngram_range=(1, 2))),
	('classifier',  MultinomialNB()) ])
	#('classifier',  SVC(kernel='linear')) ])

k_fold = KFold(n=len(data["text"]), n_folds=10)
new_data_text = numpy.asarray(data['text'])
new_data_class = numpy.asarray(data['class'])
scores = []

for train_indices, test_indices in k_fold:
	train_text = new_data_text[train_indices]
	train_y = new_data_class[train_indices]
	test_text = new_data_text[test_indices]
	test_y = new_data_class[test_indices]

	pipeline.fit(train_text, train_y)

	predicted = pipeline.predict(test_text)

	score = pipeline.score(test_text, test_y)
	scores.append(score)

print(metrics.classification_report(test_y, predicted, target_names=['ham', 'spam']))

score = sum(scores) / len(scores)

print "Mean Accuracy: " + str(score)