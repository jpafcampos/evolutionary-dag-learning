import random
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import csv
from pgmpy.readwrite import BIFReader
import time
import scipy.linalg as slin
from scipy.linalg import det
from scipy import linalg
import statistics
import bnlearn as bn
from pgmpy.metrics import structure_score
from pgmpy.models import BayesianNetwork
import gc
from matplotlib import pyplot as plt


def hamming_distance_digraph(G1, G2):
    # Ensure both graphs are directed
    if not G1.is_directed() or not G2.is_directed():
        raise ValueError("Both input graphs must be directed graphs")

    # Get edge sets of the graphs
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())

    # Calculate the symmetric difference of edge sets
    symmetric_diff = edges1.symmetric_difference(edges2)

    # Return the number of edges that differ (Hamming distance)
    return len(symmetric_diff)

def hamming_distance_undirected(G1, G2):
    # Ensure both graphs are directed
    if not G1.is_directed() or not G2.is_directed():
        raise ValueError("Both input graphs must be directed graphs")

    # Get undirected edge sets of the graphs
    edges1 = set([(u, v) if u <= v else (v, u) for u, v in G1.edges()])
    edges2 = set([(u, v) if u <= v else (v, u) for u, v in G2.edges()])

    # Calculate the symmetric difference of undirected edge sets
    symmetric_diff = edges1.symmetric_difference(edges2)

    # Return the number of edges that differ (Hamming distance)
    return len(symmetric_diff)

def learning_factors(ind, target):
    TC=0
    TE=len(target.edges())
    IE=0
    for i in target.edges():
        if ind.has_edge(i[0],i[1]):
            TC+=1
        if ind.has_edge(i[1],i[0]):
            IE+=1
    SLF=TC/TE
    TLF=(TC+IE)/TE

    return SLF,TLF
    
def accuracy_metrics(ind, target):
    '''
    Computes the F1 score between a target and a candidate graph
    Parameters:
    -----------
    target : networkx.DiGraph
        Target graph.
    ind1 : networkx.DiGraph
        Candidate graph.
    nodes : list
        List of nodes in the graph.
    Returns:
    --------
    float
        F1 score between the target and the candidate graph.

    '''
    nodes = list(target.nodes())
    TP=0 # True positive
    FN=0 # False negative
    FP=0 # False positive
    TN=0 # True negative
    for i in nodes:
        for j in nodes:           
            if i!=j:
                if target.has_edge(i,j):
                    if ind.has_edge(i,j):
                        TP+=1
                    else:
                        FN+=1
                else:
                    if ind.has_edge(i,j):
                        FP+=1
                    else:
                        TN+=1

    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    accuracy=(TP+TN)/(TP+TN+FN+FP)
    SHD = FN + FP
    try:
        f1score=2*(recall*precision)/(recall+precision)
    except:
        f1score=0
    return [f1score, accuracy, precision, recall, SHD]