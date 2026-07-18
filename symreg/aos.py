# AOS: selección adaptativa de operadores (softmax de scores con poda periódica).
import numpy as np


class OperatorManager:
    def __init__(self, unary_catalog, binary_catalog, tau=1.0, lr=0.3, decay=0.99,
                 prune_every=30, min_keep_un=3, min_keep_bin=3, prune_threshold=0.05):
        self.tau = tau; self.lr = lr; self.decay = decay
        self.prune_every = prune_every
        self.min_keep_un = min_keep_un; self.min_keep_bin = min_keep_bin
        self.prune_threshold = prune_threshold
        self.un_list = list(unary_catalog); self.bin_list = list(binary_catalog)
        self.un_scores = {n: 0.0 for n, _ in self.un_list}
        self.bin_scores = {n: 0.0 for n, _ in self.bin_list}
        self.generation = 0

    def softmax_probs(self, scores, names):
        v = np.array([scores[n] for n in names], dtype=np.float64)
        v = v - v.max() if len(v) > 0 else v
        p = np.exp(v / max(1e-6, self.tau)) if len(v) > 0 else v
        p = p / p.sum() if (len(v) > 0 and p.sum() > 0) else (np.ones(len(v))/len(v) if len(v) > 0 else np.array([]))
        return p

    def sample_unary(self):
        names = [n for n, _ in self.un_list]
        p = self.softmax_probs(self.un_scores, names)
        idx = int(np.random.choice(len(names), p=p))
        return self.un_list[idx], p[idx]

    def sample_binary(self):
        names = [n for n, _ in self.bin_list]
        p = self.softmax_probs(self.bin_scores, names)
        idx = int(np.random.choice(len(names), p=p))
        return self.bin_list[idx], p[idx]

    def update_reward(self, used_ops, reward):
        for name in used_ops:
            if name in self.un_scores:
                self.un_scores[name] = self.decay*self.un_scores[name] + self.lr*reward
            if name in self.bin_scores:
                self.bin_scores[name] = self.decay*self.bin_scores[name] + self.lr*reward

    def maybe_prune(self):
        self.generation += 1
        if self.prune_every <= 0 or (self.generation % self.prune_every) != 0:
            return
        # Unarias
        names = [n for n, _ in self.un_list]
        if len(names) > self.min_keep_un:
            p = self.softmax_probs(self.un_scores, names)
            order = list(np.argsort(-p))
            survivors = [self.un_list[i] for i in order if p[i] >= self.prune_threshold]
            if len(survivors) < self.min_keep_un:
                survivors = [self.un_list[i] for i in order[:self.min_keep_un]]
            self.un_list = survivors
            self.un_scores = {n: self.un_scores[n] for n, _ in self.un_list}
        # Binarias
        names = [n for n, _ in self.bin_list]
        if len(names) > self.min_keep_bin:
            p = self.softmax_probs(self.bin_scores, names)
            order = list(np.argsort(-p))
            survivors = [self.bin_list[i] for i in order if p[i] >= self.prune_threshold]
            if len(survivors) < self.min_keep_bin:
                survivors = [self.bin_list[i] for i in order[:self.min_keep_bin]]
            self.bin_list = survivors
            self.bin_scores = {n: self.bin_scores[n] for n, _ in self.bin_list}
