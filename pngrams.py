import tensorflow as tf
import numpy as np
import os, itertools, pickle, random, sys, traceback

N = 3

def deprecated(func):
	"""
	Decorator for other objects that are deprecated.
	"""
	print "[WARNING] call to deprecated function " + func.__name__
	return func

class PNGraph(object):

	"""
	Two-layer classifier that predicts native language from transition probabilities.
	"""

	def __init__(self, N=N, g=29, c=2, h=100, f=tf.sigmoid, alpha=0.001):

		n = sum(g ** i for i in range(1, N + 1))

		self.x = tf.placeholder(shape=[None, n], dtype=tf.float32, name="classifier/x")
		self.y = tf.placeholder(shape=[None, c], dtype=tf.float32, name="classifier/y")

		self.W1 = tf.Variable(tf.random_uniform([n, h], -1.0, 1.0), name="classifier/W1")
		self.b1 = tf.Variable(tf.zeros([1, h]), name="classifier/b1")

		self.h = f(tf.matmul(self.x, self.W1) + self.b1)

		self.W2 = tf.Variable(tf.random_uniform([h, c], -1.0, 1.0), name="classifier/W2")
		self.b2 = tf.Variable(tf.zeros([1, c]), name="classifier/b2")

		y_= tf.matmul(self.h, self.W2) + self.b2
		self.y_ = tf.nn.softmax(y_)

		self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y)
		self.train = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss)

	def batchTrain(self, session, x, y):
		return sum(session.run([self.loss, self.train], feed_dict={
			self.x: x, self.y: y
		})[0])

	def score(self, predicted, actual):
		diff = np.rint(predicted) - actual
		return 0.0 if np.count_nonzero(diff) > 0 else 1.0

	def getName(self):
		return "classifier"

class AEGraph(object):

	"""
	Auto-encoder that reduces dimensionality of transition probability vectors.
	"""

	def __init__(self, N=N, g=29, h=100, f=tf.sigmoid, alpha=0.001):
		
		n = sum(g ** i for i in range(1, N + 1))

		self.x = tf.placeholder(shape=[None, n], dtype=tf.float32, name="autoencoder/x")

		self.W1 = tf.Variable(tf.random_uniform([n, h], -1.0, 1.0), name="autoencoder/W1")
		self.b1 = tf.Variable(tf.zeros([1, h]), name="autoencoder/b1")

		self.h = f(tf.matmul(self.x, self.W1) + self.b1, name="autoencoder/h")

		self.W2 = tf.Variable(tf.random_uniform([h, n], -1.0, 1.0), name="autoencoder/W2")
		self.b2 = tf.Variable(tf.zeros([1, n]), name="autoencoder/b2")

		self.y_ = f(tf.matmul(self.h, self.W2) + self.b2)
		self.loss = tf.reduce_mean(tf.pow(self.x - self.y_, 2))
		self.train = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss)

	def batchTrain(self, session, x, y):
		return session.run([self.loss, self.train], feed_dict={
			self.x: x
		})[0]

	def score(self, predicted, actual):
		# TODO figure out how to score the autoencoder well
		# distance = np.linalg.norm(predicted - actual)
		return 1.0

	def getName(self):
		return "autoencoder"

def getTemporalDistributions(filename):
	return np.load(open(filename))

def getBaiduTrials(dirname):

	filenames = os.listdir(dirname)
	output = []
	maxSize = 0
	for i, filename in enumerate(filenames):

		if not filename.endswith("npy"): continue

		distributions = getTemporalDistributions(dirname + "/" + filename)
		output.append((i, filename[:-4], distributions, (1, 0) if filename.startswith("en") else (0, 1)))
		maxSize = max(maxSize, len(distributions))
	return output, maxSize

def elmult(v1, v2):
	"""
	Multiply two vectors with the same dimension element-wise.
	"""
	assert len(v1) == len(v2)
	return [v1[i] * v2[i] for i in range(len(v1))]

def calculatePNGrams():
	"""
	Dynamic programming algorithm to efficiently calculate grapheme transition probabilities from Baidu CTC output.
	"""

	print "getting all trials.."
	trials, maxSize = getBaiduTrials("melville-moby_dick/npy")

	chars = range(29)
	for i, filename, distribution, _ in trials:

		print "[{} / {}] starting {}..".format(i, len(trials), filename)
		height, width = distribution.shape

		# dynamic programming base case: we need the row vectors
		vecs = {(i,): distribution[i,:] for i in range(height)}
		counts = {(i,): sum(vecs[(i,)]) for i in range(height)}
		trans = {(i,): counts[(i,)] / width for i in range(height)}

		# bigrams to generate
		for n in range(2, N + 1):
			for string in itertools.product(chars, repeat=n):
				a = string[:-1]
				b = string[-1:]
				vecs[string] = elmult(vecs[a][:-len(b)], vecs[b][len(a):])
				counts[string] = sum(vecs[string])
				trans[string] = counts[string] / counts[a]

		newName = "melville-moby_dick/pkl/" + filename + ".pkl"
		print "saving transition probabilities to " + newName + ".."
		pickle.dump(trans, open(newName, "w"))

def getTrials(dirname):

	keys = None
	output = []
	files = os.listdir(dirname)
	for i, filename in enumerate(files):

		if not filename.endswith("pkl"): continue
		d = pickle.load(open(dirname + "/" + filename))
		if not keys: keys = sorted(d)

		if i % 100 == 0: print "{} / {} trials loaded".format(i, len(files))

		x = [d[k] for k in keys]
		y = (1, 0) if filename.startswith("en") else (0, 1)
		output.append((i, filename, x, y))

	return output

def doTraining(graph, train, dev, epochs=10, batchSize=100, session=tf.Session()):

	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	for i in range(epochs):
		random.shuffle(train)
		loss = 0.0
		for j in range(0, len(train) - batchSize, batchSize):
			batch = train[j:j + batchSize]
			_, _, x, y = zip(*batch)
			loss += graph.batchTrain(session, x, y)
		print "epoch {} loss: {:.5f}".format(i, loss / len(train))
		saver.save(session, "melville-moby_dick/model/{}-{}.ckpt".format(graph.getName(), len(train)))
		doTesting(graph, dev, session)

	return session

def doTesting(graph, test, session):
	score = 0.0
	for _, _, x, y in test:
		y_ = session.run(graph.y_, feed_dict={graph.x: [x]})
		score += graph.score(y_, y) #TODO this line won't work for AE

	print "accuracy: {}".format(score / len(test))

def getEmbeddings(graph, test, session):
	embeddings = []
	for _, _, x, _ in test:
		embeddings.append(np.squeeze(session.run(graph.h, feed_dict={graph.x: [x]})))
	return embeddings

def embed(test):
	with tf.Session() as session:
		graph = AEGraph()
		embeddings = getEmbeddings(graph, test, session)
		for i, filename, _, _ in test:
			pickle.dump(embeddings[i], open("melville-moby_dick/pkl-small" + filename, "w"))
			if i % 100 == 0: print "{} / {} embeddings saved".format(i, len(files))

def main():

	print "initializing graphs.."
	pnGraph = PNGraph()
	aeGraph = AEGraph()

	print "loading training set.."
	train = getTrials("melville-moby_dick/train")

	print "loading testing set.."
	test = getTrials("melville-moby_dick/test")

	print "starting training for classification.."
	classSession = doTraining(pnGraph, train, test)

	print "starting training for auto-encoder.."
	encodeSession = doTraining(aeGraph, train, test, epochs=1)

	print "testing classification network.."
	doTesting(pnGraph, test, classSession)

	print "testing auto-encoder.."
	doTesting(aeGraph, test, encodeSession)

	while True:
		try:
			print input("> ")
		except:
			traceback.print_exc(file=sys.stdout)

if __name__ == '__main__':
	main()