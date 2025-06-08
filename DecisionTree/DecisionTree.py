import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import utils

dataset = pd.DataFrame({
    'x_0':[7,3,2,1,2,4,1,8,6,7,8,9],
    'x_1':[1,2,3,5,6,7,9,10,5,8,4,6],
    'y': [0,0,0,0,0,0,1,1,1,1,1,1]
})

features = dataset[['x_0', 'x_1']]
labels = dataset['y']

utils.plot_points(features, labels)

decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(features, labels)
decision_tree.score(features, labels)

utils.display_tree(decision_tree)