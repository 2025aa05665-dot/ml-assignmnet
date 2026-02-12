from collections import Counter
import math

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left subtree
        self.right = right         # Right subtree
        self.value = value         # Class value if leaf node

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(set(y))

        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        best_gain = -1
        best_feature = None
        best_threshold = None

        # Find best split
        for feature in range(n_features):
            thresholds = set(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree)

    def _entropy(self, y):
        counts = Counter(y)
        entropy = 0
        for count in counts.values():
            p = count / len(y)
            entropy -= p * math.log2(p) if p > 0 else 0
        return entropy

    def _information_gain(self, parent, left_child, right_child):
        n = len(parent)
        n_left, n_right = len(left_child), len(right_child)

        gain = self._entropy(parent)
        gain -= (n_left / n) * self._entropy(left_child)
        gain -= (n_right / n) * self._entropy(right_child)
        return gain

    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)