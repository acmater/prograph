from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import copy
import time
import random
import pickle
import tqdm as tqdm
import networkx as nx
import multiprocessing as mp
from functools import partial, reduce
from utils.dataset import Dataset

from colorama import Fore
from colorama import Style

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from molecule import Molecule

class Landscape(ABC):
    """
    Abstract Class that provides basic functionality for landscapes

    Parameters
    ----------

    data : np.array

        Numpy Array containg protein data. Expected shape is (Nx2), with the first
        column being the sequences, and the second being the fitnesses

    seed_seq : str, default=None

        Enables the user to explicitly provide the seed sequence as a string

    seed_id : int,default=0

        Id of seed sequences within sequences and fitness

    csv_path : str,default=None

        Path to the csv file that should be imported using CSV loader function

    custom_columns : {"x_data"    : str,
                      "y_data"    : str
                      "index_col" : int}, default=None

        First two entries are custom strings to use as column headers when extracting
        data from CSV. Replaces default values of "Sequence" and "Fitness".

        Third value is the integer to use as the index column.

        They are passed to the function as keyword arguments

    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'

        String containing all allowable amino acids for tokenization functions

    saved_file : str, default=None

        Saved version of this class that will be loaded instead of instantiating a new one

    Attributes
    ----------
    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'

        String containing all allowable amino acids in tokenization functions

    sequence_mutation_locations : np.array(bool)

        Array that stores boolean values with Trues indicating that the position is
        mutated relative to the seed sequence

     mutated_positions: np.array(int)

        Numpy array that stores the integers of each position that is mutated

    d_data : {distance : index_array}

        A dictionary where each distance is a key and the values are the indexes of nodes
        with that distance from the seed sequence

    tokens : {tuple(tokenized_sequence) : index}

        A dictionary that stores a tuple format of the tokenized string with the index
        of it within the data array as the value. Used to rapidly perform membership checks

    seed_seq : str

        Seed sequence as a string

    tokenized : np.array, shape(N,L+1)

        Array containing each sequence with fitness appended onto the end.
        For the shape, N is the number of samples, and L is the length of the seed sequence

    mutation_array : np.array, shape(L*20,L)

        Array containing all possible mutations to produce sequences 1 amino acid away.
        Used by maxima generator to accelerate the construction of the graph.

        L is sequence length.

    hammings : np.array(N,)

        Numpy array of length number of samples, where each value is the hamming
        distance of the species at that index from the seed sequence.

    max_distance : int

        The maximum distance from the seed sequence within the dataset.

    graph : {idx  : "tokenized"  : tuple(int)     - a Tuple representation of the tokenized protein sequence
                    "string"     : str            - string representation of protein sequence
                    "fitness"    : float          - The fitness value associated with this protein
                    "neighbours" : np.array[idxs] - A numpy array of indexes in the dataset that are neighbours

        A memory efficient storage of the graph that can be passed to graph visualisation packages

    Written by Adam Mater, last revision 23.2.21
    """

    def __init__(self,csv_path):

                """data=None,
                      seed_seq=None,
                      seed_id=0,
                      gen_graph=False,
                      csv_path=None,
                      custom_columns={"x_data" : "Sequence",
                                      "y_data" : "Fitness",
                                      "index_col" : None},
                      amino_acids='ACDEFGHIKLMNPQRSTVWY',
                      saved_file=None
                      ):

        if saved_file:
            try:
                print(saved_file)
                self.load(saved_file)
                return None
            except:
                raise Exception.FileNotFoundError("File could not be opened")

        if (not data and not csv_path):
            raise Exception.FileNotFoundError("Data must be provided as numpy array or csv file")

        if csv_path and data:
            print('''Both a filepath and a data array has been provided. The CSV is
                     given higher priority so the data will be overwritten by this
                     if possible''')
            self.csv_path = csv_path
            data = self.csvDataLoader(csv_path,**custom_columns)

        elif csv_path:

            self.csv_path = csv_path
            data = self.csvDataLoader(csv_path,**custom_columns)

        sequences            = data[:,0]
        fitnesses            = data[:,1]

        self.gen_graph       = gen_graph

        #### Tokenizing
        self.amino_acids     = amino_acids
        self.custom_columns  = custom_columns
        self.tokens          = {x:y for x,y in zip(self.amino_acids, list(range(len(self.amino_acids))))}

        self.graph = {idx : Protein(sequence) for idx,sequence in enumerate(sequences)}
        self.update_graph(fitnesses,"fitness")
        self.seq_idxs        = {seq : idx for idx, seq in enumerate(sequences)}
        self.len             = len(sequences)

        if seed_seq:
            self.seed_prot      = Protein(seed_seq)
        else:
            self.seed_id        = seed_id
            self.seed_prot      = Protein(sequences[self.seed_id])

        self.seed = self.seed_prot.seq

        setattr(self.seed_prot,"tokenized",tuple(self.tokenize(self.seed_prot.seq)))

        self.seq_len     = len(self.seed_prot.seq)
        self.tokenized = np.concatenate((self.tokenize_data(sequences),fitnesses.reshape(-1,1)),axis=1)
        self.update_graph([tuple(x) for x in self.tokenized[:,:-1]],"tokenized")
        self.token_dict = {tuple(seq) : idx for idx,seq in enumerate(self.tokenized[:,:-1])}

        self.mutated_positions = self.calc_mutated_positions()
        self.sequence_mutation_locations = self.boolean_mutant_array(self.seed_prot.seq)
        # Stratifies data into different hamming distances

        # Contains the information to provide all mutants 1 amino acid away for a given sequence
        self.mutation_arrays  = self.gen_mutation_arrays()

        self.d_data = self.gen_d_data()

        if self.gen_graph:
            self.update_graph(self.build_graph(sequences, fitnesses),"neighbours")

        self.learners = {}
        print(self)"""

    @abstractmethod
    def update_graph(self, data, label):
        for idx, protein in self.graph.items():
            setattr(protein, label, data[idx])
        return None

    @abstractmethod
    def __str__(self):
        # TODO Change print formatting for seed sequence so it doesn't look bad
        return """
        Protein Landscape class
            Number of Sequences : {0}
            Max Distance        : {1}
            Number of Distances : {2}
            Seed Sequence       : {3}
                Modified positions are shown in green
        """.format(len(self),
                   self.max_distance,
                   len(self.d_data),
                   self.coloured_seed_string())

    @abstractmethod
    def __repr__(self):
        # TODO Finish this
        return f"""Protein_Landscape(seed_seq='{self.seed_prot.seq}',
                                  gen_graph={self.gen_graph},
                                  csv_path='{self.csv_path}',
                                  custom_columns={self.custom_columns},
                                  amino_acids='{self.amino_acids}')"""

    @abstractmethod
    def __len__(self):
        return len(self.tokenized)

    @abstractmethod
    def __getitem__(self,idx):
        return self.graph.get(idx,"Not a valid index")

    @abstractmethod
    def label_iter(self, label):
        """
        Helper function that returns an iterable over a particular label for each
        Protein
        """
        # Returns a generator based around a particular label
        return (molecule[label] for molecule in self.graph.values())

    # WHAT ABOUT THIS ONE?
    def get_distance(self,dist,d_data=None):
        """ Returns all the index of all arrays at a fixed distance from the seed string

        Parameters
        ----------
        dist : int

            The distance that you want extracted

        d_data : dict, default=None

            A custom d_data dictionary can be provided. If none is provided, self.d_data
            is used.
        """
        if d_data is None:
            d_data = self.d_data

        assert dist in d_data.keys(), "Not a valid distance for this dataset"
        return d_data[dist]

    def gen_d_data(self,seq=None) -> dict:
        """
        Generates a dictionary of possible distances from a sequence and then
        populates it with boolean indexing arrays for each distance

        IMPORTANT Boolean indexing arrays are used here instead of arrays of integer
        indexes because it is simpler to manipulate groups of them and bitwise
        operations become available.

        Parameters
        ----------
        seq : str, int, tuple

            A protein sequence provided in any one of the formats that query can parse.
        """
        if seq is None:
            seq = self.seed

        else:
            seq = self.graph[self.query(seq)]["seq"]

        subsets = {x : [] for x in range(len(seq)+1)}

        self.hammings = self.hamming_array(seq=seq)

        for distance in range(len(seq)+1):
            subsets[distance] = np.where(np.equal(distance,self.hammings))[0]
            # Stores an indexing array that isolates only sequences with that Hamming distance

        d_data = {k : v for k,v in subsets.items() if v.any()}
        self.max_distance = max(d_data.keys())
        return d_data

    def get_mutated_positions(self,positions):
        """
        Function that returns the portion of the data only where the provided positions
        have been modified.

        Parameters
        ----------
        positions : np.array(ints)

            Numpy array of integer positions that will be used to index the data.
        """
        for pos in positions:
            assert pos in self.mutated_positions, "{} is not a position that was mutated in this dataset".format(pos)

        constants = np.setdiff1d(self.mutated_positions,positions)
        index_array = np.ones((self.seq_len),dtype=np.int8)
        index_array[positions] = 0
        mutated_indexes = np.all(np.invert(self.sequence_mutation_locations[:,constants]),axis=1)
        # This line checks only the positions that have to be constant, and ensures that they all are

        return mutated_indexes

    @staticmethod
    def csvDataLoader(csvfile,x_data,y_data,index_col):
        """Simple helper function to load NK landscape data from CSV files into numpy arrays.
        Supply outputs to sklearn_split to tokenise and split into train/test split.

        Parameters
        ----------

        csvfile : str

            Path to CSV file that will be loaded

        x_data : str, default="Sequence"

            String key used to extract relevant x_data column from pandas dataframe of
            imported csv file

        y_data : str, default="Fitness"

            String key used to extract relevant y_data column from pandas dataframe  of
            imported csv file

        index_col : int, default=None

            Interger value, if provided, will determine the column to use as the index column

        returns np.array (Nx2), where N is the number of rows in the csv file

            Returns an Nx2 array with the first column being x_data (sequences), and the second being
            y_data (fitnesses)
        """

        data         = pd.read_csv(csvfile,index_col=index_col)
        protein_data = data[[x_data,y_data]].to_numpy()

        return protein_data

    @abstractmethod
    def tokenize(self,seq,tokenizer=None):
        """
        Simple static method which tokenizes an individual sequence
        """
        if tokenizer == None:
            return [self.tokens[aa] for aa in seq]
        else:
            return "This feature is not ready yet"

    @abstractmethod
    def tokenize_data(self, sequences):
        """
        Takes an iterable of sequences provided as one amino acid strings and returns
        an array of their tokenized form.

        Note : The tokenize function is not called and the tokens value is regenerated
        as it removes a lot of function calls and speeds up the operation significantly.
        """
        tokens = self.tokens
        return np.array([[tokens[aa] for aa in seq] for seq in sequences])

    ############################################################################
    ##################### Graph Generation and Manipulation ####################
    ############################################################################

    @staticmethod
    @abstractmethod
    def distance(rep1, rep2):
        """
        Distance abstractmethod which can be used to calculate the distance between any two
        representations in the dataset. The default provided here is the Hamming distance
        between two strings.
        """
        return sum(rep1 != rep2 for rep1,rep2 in zip(rep1,rep2))

    @abstractmethod
    def calc_neighbours(self,rep,idxs=None,threshold=1):
        """
        Takes a representation and checks all other representations in the dataset
        to determine if any of them are neighbours. Note that this is a highly inefficient
        approach in many cases, and should be overwritten.

        Parameters:
        -----------

        rep : str

            A string representation of a dataset member

        idxs : np.array[np.int], default=None

            the indexes of possible neighbours to consider

        threshold : int, default=1

            Threshold distance in which representations are considered neighbours.
        """
        # TODO Add index functionality
        distances = []
        for rep2 in self.label_iter("rep"):
            distances.append(self.distance(rep,rep2))
        neighbours = np.intersect1d(np.where(0 < np.array(distances))[0],np.where(np.array(distances) <= threshold)[0])
        return neighbours

    @abstractmethod
    def build_graph(self, sequences, fitnesses,idxs=None,single_thread=False):
        """
        Efficiently builds the graph of the protein landscape. There are two ways to build
        graphs for protein networks. The first considers the entire data sequence and calculates
        the Hamming distance for each pair (once) and uses this information to identify the neighbours.

        The second is to explicitly produce the neighbours, and then check to see if any of them are
        in the dataset. The second approach is typically vastly quicker. The reason for this is that even for
        an enormous protein (500 AAs) with the full 20 canonical amino acids and possible mutants, there are only
        10000 neighbours to generate and then perform in membership checking on.

        Most protein datasets have 10s to 100s of thousands of entries, and as such, this approach leads to dramatic
        speed ups.

        Parameters:
        -----------
        idxs : np.array[int]

            An array of integers that are used to index the complete dataset
            and provide a subset to construct a subgraph of the full dataset.
        """

        if idxs is None:
            print("Building Protein Graph for entire dataset")
            token_dict = self.token_dict
            pool = mp.Pool(mp.cpu_count())
            indexes = np.array(range(len(self)))

        else:
            print("Building Protein Graph For subset of length {}".format(sum(idxs)))
            dataset = self.tokenized[idxs,:-1]
            token_dict = {key : value for key,value in self.token_dict.items() if value in idxs}
            if len(idxs) < 100000:
                pool = mp.Pool(4)
            else:
                pool = mp.Pool(mp.cpu_count())
            indexes = [x for x in idxs]

        # This section roughly estimates the cost of the two different approaches to calculating the graph
        # representation of the dataset, and determines which to use based around whichever will result in
        # fewer total operations.

        calculating_explicit_neighbours = len(self.amino_acids) * len(self.seed) * len(self)
        calculating_implicit_neighbours = len(self) ** 2

        if calculating_explicit_neighbours >= 10*calculating_implicit_neighbours:
            explicit_neighbours=False
        else:
            explicit_neighbours=True

        mapfunc = partial(self.calc_neighbours,token_dict=token_dict,explicit_neighbours=explicit_neighbours,idxs=indexes)
        neighbours = list(pool.map(mapfunc,tqdm.tqdm(indexes)))
        return neighbours # The indices are stored as the first value of the tuple

    def graph_to_networkx(self,labels=None,update_self=False):
        """
        Produces a networkx graph from the internally stored graph object

        Parameters
        ----------
        labels : [str], default=None

            A list of strings that will be used to generate label_iters using the self.label_iter
            method. This information will then be added to each node.
        """
        # So the problem here is that I do not know how to stick labels on with the
        # nodes. As a result, cytoscape won't be able to visualize the graphs properly.
        sequences = self.label_iter("seq")
        label_iters = []
        if labels is not None:
            for label in labels:
                label_iters.append(self.label_iter(label))
        prots = [prot for prot in sequences]
        g = nx.Graph()
        for idx,prot in enumerate(prots):
            g.add_node(prot, **{labels[x] : label_iters[x][idx] for x in range(len(label_iters))})
        for idx,neighbours in enumerate(tqdm.tqdm(self.label_iter("neighbours"))):
            g.add_edges_from([(sequences[idx], sequences[neighbour_idx]) for neighbour_idx in neighbours])
        if update_self:
            self.networkx_graph = g
            return None
        else:
            return g

    ############################################################################
    ################### Data Manipulation and Slicing ##########################
    ############################################################################

    def get_data(self,tokenized=False):
        """
        Simple function that returns a copy of the data stored in the class.

        Parameters
        ----------
        tokenized : Bool, default=False

            Boolean value that determines if the raw or tokenized data will be returned.
        """
        if tokenized:
            return np.array([x[["seq","fitness"]] for x in self.graph])
        else:
            return copy.copy(self.tokenized)

    ############################################################################
    ################################ Utilities #################################
    ############################################################################
    @abstractmethod
    def save(self,name=None,ext=".txt"):
        """
        Save function that stores the entire landscape so that it can be reused without
        having to recompute distances and tokenizations

        Parameters
        ----------
        name : str, default=None

            Name that the class will be saved under. If none is provided it defaults
            to the same name as the csv file provided.

        ext : str, default=".txt"

            Extension that the file will be saved with.
        """
        if self.csv_path:
            directory, file = self.csv_path.rsplit("/",1)
            directory += "/"
            if not name:
                name = file.rsplit(".",1)[0]
            file = open(directory+name+ext,"wb")
            file.write(pickle.dumps(self.__dict__))
            file.close()

    @abstractmethod
    def load(self,name):
        """
        Functions that instantiates the landscape from a saved file if one is provided

        Parameters
        ----------
        name: str

            Provides the name of the file. MUST contain the extension.
        """
        file = open(name,"rb")
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)
        return True

if __name__ == "__main__":
    test = Landscape(csv_path="../Data/NK/K4/V1.csv")
