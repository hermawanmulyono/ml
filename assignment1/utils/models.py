from sklearn.tree import DecisionTreeClassifier


def get_decision_tree(ccp_alpha: float):
    dt = DecisionTreeClassifier(criterion='entropy',
                                splitter='best',
                                ccp_alpha=ccp_alpha)
    return dt

