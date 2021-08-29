from sklearn.tree import DecisionTreeClassifier


def get_decision_tree(ccp_alpha: float) -> DecisionTreeClassifier:
    """Constructs a decision tree with pruning

    Args:
        ccp_alpha: Pruning minimal cost-complexity paramether alpha

    Returns:
        A decision tree classifier object

    """
    dt = DecisionTreeClassifier(criterion='entropy',
                                splitter='best',
                                ccp_alpha=ccp_alpha)
    return dt
