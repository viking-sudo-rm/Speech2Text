from __future__ import print_function
import sklearn.manifold, matplotlib
import matplotlib.pyplot as plt
from pngrams import *

model = "melville-moby_dick/model/autoencoder-2739.ckpt"

with tf.Session() as session:

	print("initializing graph..")
	graph = AEGraph()

	print("restoring model..")
	saver = tf.train.Saver()
	saver.restore(session, model)

	print("loading testing set..")
	test = getTrials("melville-moby_dick/test")

	print("generating embeddings..")
	embeddings = getEmbeddings(graph, test, session)
	print(embeddings)

	print("fitting TSNE..")
	tsne = sklearn.manifold.TSNE(n_components=2, perplexity=30.0)
	X_reduced = tsne.fit_transform(embeddings)

	x, y = zip(*X_reduced) # separate into two lists

	fig, ax = plt.subplots()
	ax.scatter(x, y)

	# add text labels
	for i, filename, _, _ in test:

		plt.annotate(
			filename,
			xy=X_reduced[i],
			fontsize=6,
			facecolor="green" if filename.startswith("en") else "red"
		)

	fig.savefig("clusters.png")
	# plt.show()