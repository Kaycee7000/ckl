import networkx as nx
from knowledge_engine.mining import frequent_subgraphs
from knowledge_engine.abstraction import abstract_variable
from knowledge_engine.confidence import confidence_from_backtests


def test_frequent_subgraphs_triangle():
    # create graphs with triangle present in two graphs
    G1 = nx.Graph()
    G1.add_edges_from([(1,2),(2,3),(3,1)])
    G2 = nx.Graph()
    G2.add_edges_from([(4,5),(5,6),(6,4)])
    G3 = nx.Graph()
    G3.add_edges_from([(7,8)])
    freq = frequent_subgraphs([G1, G2, G3], k=3, min_support=2)
    assert len(freq) >= 1


def test_abstraction_mapping():
    mapping = {
        'churn': 'customer_attrition_rate',
        'retention': 'customer_retention_rate',
    }
    assert abstract_variable('SaaS Churn Rate', mapping) == 'customer_attrition_rate'
    assert abstract_variable('Monthly Retention', mapping) == 'customer_retention_rate'


def test_confidence_score_monotonic():
    # lower CRPS across folds yields higher score
    r_high = [{'crps': [1.0, 1.2]}, {'crps': [0.9]}]
    r_low = [{'crps': [0.1, 0.2]}, {'crps': [0.05]}]
    s_high = confidence_from_backtests(r_high)
    s_low = confidence_from_backtests(r_low)
    assert s_low > s_high
