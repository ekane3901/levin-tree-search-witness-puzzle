import numpy as np

class UniformModel:
    def get_probabilities(self, _):
        return np.ones(4) / 4

class Model:
    def __init__(self):
        self._table = {}
        self._learning_rate = 0.2

    def _ensure_context(self, context):
        if context not in self._table:
            self._table[context] = np.zeros(4, dtype=float)
        return self._table[context]

    def _softmax(self, logits):
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits)

    def get_probabilities(self, context):
        weights = self._ensure_context(context)
        return self._softmax(weights)
    
    def update(self, path):

        states = path.get_states()
        actions = path.get_actions()

        for s, a in zip(states, actions):
            context = s.get_context()
            regular, reversed_ctx = s.get_reversed_context()


            # Update regular context
            weights = self._ensure_context(regular)
            probs = self._softmax(weights)
            gradient = probs.copy()
            gradient[a] -= 1
            self._table[regular] -= self._learning_rate * gradient

            # Update reversed context
            weights_rev = self._ensure_context(reversed_ctx)
            probs_rev = self._softmax(weights_rev)
            gradient_rev = probs_rev.copy()
            gradient_rev[a] -= 1
            self._table[reversed_ctx] -= self._learning_rate * gradient_rev 
            