import numpy as np

class tsne:
    
    def __init__(self, n_components, perplexity=20,learning_rate=10.,num_iters=500,Seed=1):

    self.n_components = n_components
    self.perplexity = perplexity
    self.learning_rate = learning_rate
    self.num_iter = num_iter
    self.random_sate=np.random.RandomState(Seed)
    self.embedding = None
    
    