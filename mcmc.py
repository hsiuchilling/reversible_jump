import numpy as np
import numpy.random as npr

'''
MH returns a Metropolis Hastings object.

Initialization parameters:
1) markov_canon - provides the log conditional density of Markov Kernel.
2) markov_canon_sampler - sampler that generates samples based on current particles.
3) reversible - flag indicating kernel is reversible (default to false).
'''

class MH(object):
	def __init__(self, markov_canon, markov_canon_sampler, pi,
		num_chains = 50, num_evolutions = 100, reversible = False,
		persist_chains = False):
		self.num_samples = num_samples

	def compute_acceptance_probability(marginal1, marginal2, conditional1, conditional2):
		alpha = min(0, marginal2 - marginal1 + conditional1 - conditional2)
		return alpha

	def evolve_chain(self):
		proposals = self.markov_canon_sampler(self.samples)
		conditional1 = self.markov_canon(proposals, self.samples)
		conditional2 = self.markov_canon(self.samples, proposals)
		marginal1 = self.pi(self.samples)
		marginal2 = self.pi(proposals)
		acceptance_probability_vector = compute_acceptance_probability(marginal1, marginal2, conditional1, conditional2)
		acceptance_vector = np.log(npr.uniform(size = self.num_chains)) <= acceptance_probability_vector 
		self.samples = np.array([proposal if accepted else sample for (accepted, proposal, accepted) in zip(self.samples, proposals, acceptance_vector)])


