import pandas as pd
import numpy as np
import networkx as nx
import bnlearn as bn

def load_samplesize_num_evals(data, type_exp):
    if type_exp == 1:
        if data == 'asia':
            sample_size = 400
            max_bic_eval = 12000
        elif data == 'child':
            sample_size = 500
            max_bic_eval = 20000
        elif data == 'insurance':
            sample_size = 1000
            max_bic_eval = 60000
    elif type_exp == 2:
        if data == 'asia':
            sample_size = 1000
            max_bic_eval = 25000
        elif data == 'child':
            sample_size = 10000
            max_bic_eval = 100000
        elif data == 'insurance':
            sample_size = 20000
            max_bic_eval = 140000
    elif type_exp == 3:
        if data == 'asia':
            pass #experiment not defined
        elif data == 'child':
            sample_size = 25000
            max_bic_eval = 120000
        elif data == 'insurance':
            sample_size = 50000
            max_bic_eval = 160000

    return sample_size, max_bic_eval

# **************************************************************************************
# ******************************* ASIA DATASET *****************************************
# **************************************************************************************

def load_gt_adj_asia():
    data = load_asia_data(sample_size=1000)
    d = data.shape[1]
    nodes = data.columns
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    idx2node = {idx: node for idx, node in enumerate(nodes)}
    gt_edges = [('A', 'T'), ('T', 'E'), ('E', 'X'), ('S', 'L'), ('L', 'E'), ('E', 'X'), ('S', 'B'), ('B', 'D')]
    gt_adj = np.zeros((d, d))
    for edge in gt_edges:
        gt_adj[node2idx[edge[0]], node2idx[edge[1]]] = 1

    return gt_adj, node2idx, idx2node

def load_asia_data(path = '../data/asia10K.csv', sample_size = None, randomized=False):
    '''
    Loads the Asia dataset.
    
    Parameters:
    -----------
    path : str
        Path to the Asia dataset.
    
    Returns:
    --------
    pandas.DataFrame
        The dataset with the columns renamed and the values transformed to 0 and 1.
    '''
    data = pd.read_csv(path)
    data = data.replace({'yes': 1, 'no': 0})
    data.rename(columns={
        'Smoker': 'S',
        'LungCancer': 'L',
        'VisitToAsia': 'A',
        'Tuberculosis': 'T',
        'TuberculosisOrCancer': 'E',
        'X-ray': 'X',
        'Bronchitis': 'B',
        'Dyspnea': 'D'
    }, inplace=True)

    if sample_size is not None:
        if randomized:
            data = data.sample(sample_size)
        else:
            data = data.head(sample_size)

    return data

def load_gt_network_asia():
    '''
    Loads the ground truth graph for the Asia dataset.
    
    Returns:
    --------
    networkx.DiGraph
        The ground truth graph.
    '''

    # Create ground truth graph
    gt = nx.DiGraph()
    gt.add_edges_from([['S', 'L'],
                    ['S', 'B'],
                    ['L', 'E'],
                    ['B', 'D'],
                    ['E', 'X'],
                    ['E', 'D'],
                    ['A', 'T'],
                    ['T', 'E']])
    return gt

# **************************************************************************************
# ******************************* CHILD DATASET ****************************************
# **************************************************************************************

def load_child_data(path='../data/child.csv', sample_size=None, randomized=False):
    '''
    Loads and processes the child data.
    
    Parameters:
    -----------
    path : str, optional
        Path to the child data file.

    Returns:
    --------
    pandas.DataFrame
        The processed dataset.
    '''
    # Load the data
    data = pd.read_csv(path)
    if sample_size is not None:
        if randomized:
            data = data.sample(sample_size)
        else:
            data = data.head(sample_size)

    return data

def load_gt_network_child(path='../data/child.gml'):
    G = nx.read_gml(path)
    return G


# **************************************************************************************
# ******************************* INSURANCE DATASET ************************************
# **************************************************************************************











def load_alarm_data(path = '../data/alarm10K.csv'):
    '''
    Loads the Alarm dataset.
    
    Parameters:
    -----------
    path : str
        Path to the Alarm dataset.
    
    Returns:
    --------
    pandas.DataFrame
        The dataset.
    '''
    data = pd.read_csv(path)
    data.rename(columns={
        'HISTORY': 'H',
        'CVP': 'C',
        'PCWP': 'P',
        'HYPOVOLEMIA': 'V',
        'LVEDVOLUME': 'L',
        'LVFAILURE': 'F',
        'STROKEVOLUME': 'S',
        'ERRLOWOUTPUT': 'O',
        'HRBP': 'R',
        'HREKG': 'E',
        'ERRCAUTER': 'A',
        'HRSAT': 'T',
        'INSUFFANESTH': 'I',
        'ANAPHYLAXIS': 'X',
        'TPR': 'Y',
        'EXPCO2': 'Q',
        'KINKEDTUBE': 'K',
        'MINVOL': 'M',
        'FIO2': 'F',
        'PVSAT': 'V',
        'SAO2': 'W',
        'PAP': 'B',
        'PULMEMBOLUS': 'U',
        'SHUNT': 'Z'
    }, inplace=True)

    return data
    
def load_adult_data(path='../data/adult.csv', sample_size=None):
    '''
    Loads and processes the adult census income data.
    
    Parameters:
    -----------
    path : str, optional
        Path to the adult census income data file.

    Returns:
    --------
    pandas.DataFrame
        The processed dataset.
    '''
    # Load the data
    data = pd.read_csv(path)
    df = data.rename(columns={
    'Age': 'A',
    ' workclass': 'W',
    ' fnlwgt': 'F',
    ' education': 'ED',
    ' education-num': 'EDN',
    ' marital-status': 'M',
    ' occupation': 'O',
    ' relationship': 'R',
    ' race': 'RA',
    ' sex': 'S',
    ' capital-gain': 'CG',
    ' capital-loss': 'CL',
    ' hours-per-week': 'H',
    ' native-country': 'C',
    ' class': 'I'
    })

    df = df.drop(columns=['F', 'EDN'])

    # Discretize the age variable in three groups: <25, 25-60, >60
    df['A'] = pd.cut(df['A'], bins=[0, 25, 60, 100], labels=False)

    # Discretize the capital gain in two groups: <=5000, >5000
    df['CG'] = pd.cut(df['CG'], bins=[-1, 5000, 100000], labels=False)

    # Discretize the capital loss in two groups: <=40, >40
    df['CL'] = pd.cut(df['CL'], bins=[-1, 40, 100000], labels=False)

    # Discretize the hours per week in three groups: <40, 40-60, >60
    df['H'] = pd.cut(df['H'], bins=[0, 40, 60, 100], labels=False)
    # Process the categorical variables
    # workclass = {private, non-private}
    df['W'] = [1 if x == ' Private' else 0 for x in df['W']]

    # education = {high, low}
    df['ED'] = [1 if x in [' Bachelors', ' Masters', ' Doctorate', ' Prof-school'] else 0 for x in df['ED']]

    # marital-status = {married, non-married}
    df['M'] = [1 if x in [' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse'] else 0 for x in df['M']]

    # relationship = {married, non-married}
    df['R'] = [1 if x in [' Husband', ' Wife'] else 0 for x in df['R']]

    # native-country = {US, non-US}
    df['C'] = [1 if x == ' United-States' else 0 for x in df['C']]

    # race = {white, non-white}
    df['RA'] = [1 if x == ' White' else 0 for x in df['RA']]

    # occupation = {office, heavy-work, other}
    df['O'] = [1 if x in [' Adm-clerical', ' Sales', ' Exec-managerial', ' Prof-specialty', ' Tech-support'] else 0 for x in df['O']]

    # binarize sex variable
    df['S'] = [1 if x == ' Male' else 0 for x in df['S']]

    # binarize income variable
    df['I'] = [1 if x == ' >50K' else 0 for x in df['I']]

    if sample_size is not None:
        df = df.head(sample_size)
        #df = df.sample(sample_size)
    
    return df


def load_gt_adult():
    '''
    Loads the ground truth graph for the adult dataset.
    
    Returns:
    --------
    networkx.DiGraph
        The ground truth graph.
    '''

    # Create ground truth graph
    gt = nx.DiGraph()
    gt.add_edges_from([['S', 'O'],
                    ['S', 'H'],
                    ['S', 'R'],
                    ['S', 'ED'],
                    ['S', 'RA'],
                    ['R', 'I'],
                    ['R', 'CL'],
                    ['R', 'CG'],
                    ['R', 'H'],
                    ['R', 'W'],
                    ['R', 'A'],
                    ['R', 'M'],
                    ['R', 'RA'],
                    ['A', 'H'],
                    ['A', 'O'],
                    ['A', 'W'],
                    ['A', 'ED'],
                    ['A', 'M'],
                    ['A', 'C'],
                    ['A', 'RA'],
                    ['A', 'I'],
                    ['H','O'],
                    ['H','W'],
                    ['W', 'O'],
                    ['W', 'ED'],
                    ['W', 'C'],
                    ['O','CL'],
                    ['O','CG'],
                    ['O','I'],
                    ['O','ED'],
                    ['ED','CG'],
                    ['ED','I'],
                    ['ED','RA'],
                    ['ED','C'],
                    ['CG', 'CL'],
                    ['CG', 'I'],
                    ['C','M'],
                    ['C', 'RA'],
                    ['CL', 'I']]
                    )
    return gt






if __name__ == '__main__':
    G = load_gt_network_child()
    print(G.edges)