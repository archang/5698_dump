import sys
import argparse
import numpy as np
from pyspark import SparkContext
import re

def substring(s):
	substrings = s.split(",")
	(substrings(0),substrings(1))
	start = s.find('\'')+1
	end = s.find(',')-1
	return s[start:end]+','+s[s.find(',')+1:s.find(')')]

def toLowerCase(s):
	""" Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
   	"""
    	return s.lower()

def stripNonAlpha(s):
    	""" Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    	return ''.join([c for c in s if c.isalpha()])

if __name__ == "__main__":
    	parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    	parser.add_argument('mode', help='Mode of operation',choices=['file','TF','IDF','TFIDF','SIM','TOP'])
    	parser.add_argument('input', help='Input file or list of files.')
    	parser.add_argument('output', help='File in which output is stored')
    	parser.add_argument('--master',default="local[20]",help="Spark Master")
    	parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    	parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    	args = parser.parse_args()

    	sc = SparkContext(args.master, 'Text Analysis')
    	sc.setLogLevel("ERROR")

    	if args.mode=='TF':
		filename = args.input
		lines = sc.textFile(filename)
		lines.flatMap(lambda s: s.split()) \
			.map(lambda s: toLowerCase(s)) \
			.map(lambda s: stripNonAlpha(s)) \
			.filter(lambda s: s!='') \
			.map(lambda word: (word, 1)) \
			.reduceByKey(lambda x, y: x + y) \
			.saveAsTextFile(args.output)

        	# Read text file at args.input, compute TF of each term,
        	# and store result in file args.output. All terms are first converted to
        	# lowercase, and have non alphabetic characters removed
        	# (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., ""
        	# are also removed
        	pass







    	if args.mode=='TOP':
		filename = args.input
		lines = sc.textFile(filename)
		pairs = lines.map(eval)
		newlines = pairs.sortBy(lambda x: x[1], ascending=False)
		sc.parallelize(newlines.take(20),1).saveAsTextFile(args.output)

        	# Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL),
        	# where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        	# and store result in args.output
        	pass






    	if args.mode=='IDF':

		allFiles = sc.wholeTextFiles(args.input)
		count = allFiles.map(lambda x: (x,1)).reduce(lambda x,y: (x[0]+y[0],x[1]+y[1]))
		numCollections=count[1]
		allFiles=allFiles.flatMapValues(lambda s: s.split()) \
			.mapValues(lambda s: toLowerCase(s)) \
			.mapValues(lambda s: stripNonAlpha(s)) \
			.filter(lambda x:x[1]!='') \
			.distinct()
		wordrdd=allFiles.values().map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y)
		computedrdd=wordrdd.mapValues(lambda x: np.log(numCollections/x))
		computedrdd.saveAsTextFile(args.output)
		#allFiles.flatMapValues(lambda s: s.split()) \
		#	.mapValues(lambda s: toLowerCase(s)) \
		#	.mapValues(lambda s: stripNonAlpha(s)) \
		#	.filter(lambda x,y: y!='') \
		#	.distinct()

		# Read list of files from args.input, compute IDF of each term,
        	# and store result in file args.output.  All terms are first converted to
        	# lowercase, and have non alphabetic characters removed
        	# (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        	# are removed
        	pass






    	if args.mode=='TFIDF':
		TFFile=sc.textFile(args.input).map(eval)
		IDFFile=sc.textFile(args.idfvalues).map(eval)
		joinedRDD=TFFile.join(IDFFile)
		IDFRDD=joinedRDD.mapValues(lambda x: x[0]*x[1])
        	IDFRDD.saveAsTextFile(args.output)
		# Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        	# compute TFIDF score, and store it in file args.output. Both input files contain
        	# strings representing pairs of the form (TERM,VAL),
        	# where TERM is a lowercase letter-only string and VAL is a numeric value.
        	pass






    	if args.mode=='SIM':
		fileone=sc.textFile(args.input).map(eval)
		filetwo=sc.textFile(args.other).map(eval)
		topsum=fileone.join(filetwo).mapValues(lambda x: x[0]*x[1]).reduce(lambda x,y:(x[0]+y[0],x[1]+y[1]))[1]
		bottomleft=fileone.mapValues(lambda x:x*x).reduce(lambda x,y:(x[0]+y[0],x[1]+y[1]))[1]
		bottomright=filetwo.mapValues(lambda x:x*x).reduce(lambda x,y:(x[0]+y[0],x[1]+y[1]))[1]
		sim=topsum/(np.sqrt(bottomleft*bottomright))
		print(sim)

        	# Read  scores from file args.input the scores from file args.other,
        	# compute the cosine similarity between them, and store it in file args.output. Both input files contain
        	# strings representing pairs of the form (TERM,VAL),
        	# where TERM is a lowercase, letter-only string and VAL is a numeric value.
        	pass














