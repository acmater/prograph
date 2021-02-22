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
import torch

from colorama import Fore
from colorama import Style

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from .protein import Protein

class Protein_Landscape():
    """
    Class that handles a protein dataset

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

    Written by Adam Mater, last revision 21.2.21
    """

    def __init__(self,data=None,
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
            print("""Both a filepath and a data array has been provided. The CSV is
                     given higher priority so the data will be overwritten by this
                     if possible""")
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

        print(self)

    def seed(self):
        return self.seed_prot.seq

    def update_graph(self, data, label):
        for idx, protein in self.graph.items():
            setattr(protein, label, data[idx])
        return None

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

    def __repr__(self):
        # TODO Finish this
        return f"""Protein_Landscape(seed_seq='{self.seed_prot.seq}',
                                  gen_graph={self.gen_graph},
                                  csv_path='{self.csv_path}',
                                  custom_columns={self.custom_columns},
                                  amino_acids='{self.amino_acids}'
                                  )"""

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self,idx):
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            return {i : self.graph[x] for i,x in enumerate(self.query(idx))}
            # TODO Should the above return the index of the original data, or a reindexed
            # iterable?
        else:
            return self.graph[self.query(idx)]

    def query(self,sequence,information=False):
        """
        Helper function that interprets a query sequence and accepts multiple formats.
        This object works by moving indexes of sequences around due to the large increase
        in computational efficiency, and as a result this function returns the index associated
        with this sequence

        Parameters
        ----------
        sequence : str, tuple, int

            The sequence to query against the dataset. Multiple valid input formats

        information : Bool, default=True

            Whether or not to return the information rich form of the protein, i.e
            its multiple representations, label, and neighbours (leverages the graph object)
        """
        if isinstance(sequence, int) or isinstance(sequence, np.integer):
            assert sequence <= self.len, "Index exceeds bounds of dataset"
            idx = sequence

        elif isinstance(sequence, np.ndarray) or isinstance(sequence, list):
            if isinstance(sequence[0], np.integer) or isinstance(sequence[0], int):
                idx = sequence
            elif isinstance(sequence[0], str):
                idx = [self.seq_idxs.get(seq, "This sequence is not in the dataset") for seq in sequence]
            else:
                print("Wrong data format in numpy array or list iterable")

        elif isinstance(sequence, str):
            idx = self.seq_idxs.get(sequence, "This sequence is not in the dataset")

        elif isinstance(sequence, tuple):
            assert len(sequence) == self.seq_len, "Tuple not valid length for dataset"
            check = np.where(np.all(sequence == self.tokenized[:,:-1],axis=1))[0]
            assert len(check) > 0, "Not a valid tuple representation of a protein in this dataset"
            idx = int(check)

        else:
            raise ValueError("Input format not understood")

        if information:
            assert self.graph is not None, "To provide additional information, the graph must be computed"
            if isinstance(idx, np.ndarray) or isinstance(idx, list):
                return {ix : self.graph[ix] for ix in idx}
            else:
                return self.graph[idx]

        else:
            return idx

    def positions(self,positions):
        return self.indexing(positions=positions)

    def distances(self,distances):
        return self.indexing(distances=distances)

    def indexing(self,reference_seq=None,distances=None,positions=None,percentage=None,Bool="or",complement=False):
        """
        Function that handles more complex indexing operations, for example wanting
        to combine multiple distance indexes or asking for a random set of indices of a given
        length relative to the overall dataset.

        An important note is that percentage works after distances and positions have been applied.
        Meaning that it will reduce the index array provided by those two operations first, rather
        than providing a subset of the data to distances and positions.

        Parameters
        ----------
        reference_seq : int, str, tuple

                Any of the valid query inputs that select a single string. This will be used
                to calculate all relative positions and distances.

        distances : [int], default=None

            A list of integer distances that the dataset will return.

        positions : [int], default=None

            The mutated positions that you want to manually inspect

        percentage : float, default=None, 0 <= split_point <= 1

            Will return a fraction of the data.

        Bool : str, default = "or", "or"/"and"

            A boolean switch that changes the logic used to combine mutated positions.

        complement : bool, default=False

            Whether or not the indexes of the complement should also be returned
        """
        idxs = []

        assert Bool == "or" or Bool == "and", "Not a valid boolean value."

        if reference_seq is None:
            reference_seq   = self.seed_prot.seq
            d_data          = self.d_data
        else:
            d_data          = self.gen_d_data(self.query(reference_seq))

        if distances is not None:
            if type(distances) == int:
                distances = [distances]
            assert type(distances) == list, "Distances must be provided as integer or list"
            for d in distances:
                assert d in d_data.keys(), f"{d} is not a valid distance"
            # Uses reduce from functools package and the union1d operation
            # to recursively combine the indexing arrays.
            idxs.append(reduce(np.union1d, [d_data[d] for d in distances]))

        if positions is not None:
            # This code uses bitwise operations to maximize speed.
            # It first uses an or gate to collect every one where the desired position was modified
            # It then goes through each position that shouldn't be changed, and uses three logic gates
            # to switch ones where they're both on to off, returning the indexes of strings where ONLY
            # the desired positions are changed
            not_positions = [x for x in range(len(self[reference_seq]["seq"])) if x not in positions]
            sequence_mutation_locations = self.boolean_mutant_array(reference_seq)
            if Bool == "or":
                working = reduce(np.logical_or,[sequence_mutation_locations[:,pos] for pos in positions])
            else:
                working = reduce(np.logical_and,[sequence_mutation_locations[:,pos] for pos in positions])
            for pos in not_positions:
                temp = np.logical_xor(working,sequence_mutation_locations[:,pos])
                working = np.logical_and(temp,np.logical_not(sequence_mutation_locations[:,pos]))
            idxs.append(np.where(working)[0])

        if len(idxs) > 0:
            idxs = reduce(np.intersect1d, idxs)
        else:
            idxs = np.array(range(len(self)))

        if percentage is not None:
            assert 0 <= percentage <= 1, "Percentage must be between 0 and 1"
            indexes = np.zeros((len(idxs)),dtype=bool)
            indexes[np.random.choice(np.arange(len(idxs)),size=int(len(idxs)*percentage),replace=False)] = 1
            return idxs[indexes]

        assert len(idxs) != 0, "No possible valid indices have been provided."

        if complement:
            return idxs, np.setdiff1d(np.arange(self.len), idxs)
        else:
            return idxs

    def neighbours(self, seq, keys=[["seq"]]):
        return self[self.graph[self.query(seq)]["neighbours"]]

    def label_iter(self, label):
        """
        Helper function that returns an iterable over a particular label for each
        Protein
        """
        return np.array([protein[label] for protein in self.graph.values()])

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
    def hamming(str1, str2):
        """Calculates the Hamming distance between 2 strings"""
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def hamming_array(self,seq=None,idxs=None):
        """
        Function to calculate the hamming distances of every array using vectorized
        operations

        Function operates by building an array of the (Nxlen(seed sequence)) with
        copies of the tokenized seed sequence.

        This array is then compared elementwise with the tokenized data, setting
        all places where they don't match to False. This array is then inverted,
        and summed, producing an integer representing the difference for each string.

        Parameters:
        -----------
        seq : str, int, tuple

            Any valid sequence representation (see query for valid formats)

        idxs : np.array[int]

            A numpy integer index array
        """
        if seq is None:
            tokenized_seq = self.tokenized[0,:-1]
        else:
            tokenized_seq = self.tokenized[self.query(seq),:-1]

        if idxs is not None:
            data = self.tokenized[idxs,:-1]
        else:
            data = self.tokenized[:,:-1]

        #hold_array     = np.zeros((len(self.sequences),len(tokenized_seq)))
        #for i,char in enumerate(tokenized_seq):
        #    hold_array[:,i] = char

        hammings = np.sum(np.invert(data == tokenized_seq),axis=1)

        return hammings

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

    def tokenize(self,seq,tokenizer=None):
        """
        Simple static method which tokenizes an individual sequence
        """
        if tokenizer == None:
            return [self.tokens[aa] for aa in seq]
        else:
            return "This feature is not ready yet"

    def tokenize_data(self, sequences):
        """
        Takes an iterable of sequences provided as one amino acid strings and returns
        an array of their tokenized form.

        Note : The tokenize function is not called and the tokens value is regenerated
        as it removes a lot of function calls and speeds up the operation significantly.
        """
        tokens = self.tokens
        return np.array([[tokens[aa] for aa in seq] for seq in sequences])

    def boolean_mutant_array(self,seq=None):
        return np.invert(self.tokenized[:,:-1] == self.tokenized[self.query(seq),:-1])

    def calc_mutated_positions(self):
        """
        Determines all positions that were modified experimentally and returns the indices
        of these modifications.

        Because the Numpy code is tricky to read, here is a quick breakdown:

            self.tokenized is called, and the fitness column is removed by [:,:-1]
            Each column is then tested against the first
        """
        mutated_bools = np.invert(np.all(self.tokenized[:,:-1] == self.tokenize(self.seed_prot.seq),axis=0)) # Calculates the indices all of arrays which are modified.
        mutated_idxs  = mutated_bools * np.arange(1,len(self.seed) + 1) # Shifts to the right so that zero can be counted as an idx
        return mutated_idxs[mutated_idxs != 0] - 1 # Shifts it back

    def coloured_seed_string(self):
        """
        Printing function that prints the original seed string and colours the positions that
        have been modified
        """
        strs = []
        idxs = self.mutated_positions
        for i,char in enumerate(self.seed_prot.seq):
            if i in idxs:
                strs.append(f"{Fore.GREEN}{char}{Style.RESET_ALL}")
            else:
                strs.append("{0}".format(char))
        return "".join(strs)

    def gen_mutation_arrays(self):
        leng = len(self.seed)
        xs = np.arange(leng*len(self.amino_acids))
        ys = np.array([[y for x in range(len(self.amino_acids))] for y in range(leng)]).flatten()
        modifiers = np.array([np.arange(len(self.amino_acids)) for x in range(leng)]).flatten()
        return (xs, ys, modifiers)


    def generate_mutations(self,seq):
        """
        Takes a sequence and generates all possible mutants 1 Hamming distance away
        using array substitution

        Parameters:
        -----------
        seq : np.array[int]

            Tokenized sequence array
        """
        seq = self.tokenized[self.query(seq),:-1]
        seed = self.seed
        hold_array = np.zeros(((len(seed)*len(self.amino_acids)),len(seed)))
        for i,char in enumerate(seq):
            hold_array[:,i] = char

        xs, ys, mutations = self.mutation_arrays
        hold_array[(xs,ys)] = mutations
        copies = np.invert(np.all(hold_array == seq,axis=1))
        return hold_array[copies]

    ############################################################################
    ##################### Graph Generation and Manipulation ####################
    ############################################################################

    def calc_neighbours(self,seq,token_dict=None,explicit_neighbours=True,idxs=None):
        """
        Takes a sequence and checks all possible neighbours against the ones that are actually present within the dataset.

        There is a particular design decision here that makes it fast in some cases, and slow in others.
        Instead of checking the entire dataset to see if there are any neighbours present, it calculates all
        possible neighbours, and checks to see if any of them match entries in the dataset.

        The number of possible neighbours for a given sequence is far smaller than the size of the dataset in almost
        all cases, which makes this approach significantly faster.

        Parameters:
        -----------

        seq : int, str, tuple

            A sequence in any of the valid formats.

        token_dict : {tuple(tokenized_representation) : int}, default=None

            A token dictionary that matches each tokenized sequence to its integer index.

        explicit_neighbours : Bool, default=True

            How the graph is calculated. Explicit neighbours means that it will generate all
            possible neighbours then check to see if they're in the dataset. Otherwise it will check
            each sequence in the dataset against each other sequence.
        """
        if token_dict is None:
            token_dict = self.token_dict

        if explicit_neighbours:
            possible_neighbours =self.tuple_seqs(self.generate_mutations(seq))
            actual_neighbours = []
            for key in possible_neighbours:
                if key in token_dict:
                    actual_neighbours.append(token_dict[key])
            actual_neighbours = np.sort(actual_neighbours)#[token_dict[tuple(key)] for key in possible_neighbours if tuple(key) in token_dict])

        else:
            actual_neighbours = np.where(self.hamming_array(self.query(seq),idxs=idxs) == 1)[0]

        return actual_neighbours

    @staticmethod
    def tuple_seqs(seqs):
        return [tuple(x) for x in seqs]

    # Graph Section
    def build_graph(self,sequences, fitnesses,idxs=None,single_thread=False):
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
        neighbours = list(map(mapfunc,tqdm.tqdm(indexes)))
        return neighbours # The indices are stored as the first value of the tuple

    def graph_to_networkx(self,labels=None):
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
        self.networkx_graph = g
        return None

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

    def sklearn_data(self, data=None,idxs=None,split=0.8,scaler=False,shuffle=True):
        """
        Parameters
        ----------
        data : np.array(NxM+1), default=None

            Optional data array that will be split. Added to the function to enable it
            to interface with lengthen sequences.

            Provided array is expected to be (NxM+1) where N is the number of data points,
            M is the sequence length, and the +1 captures the extra column for the fitnesses

        distance : int or [int], default=None

            The specific distance (or distances) from the seed sequence that the data should be sampled from.

        positions : [int], default=None

            The specific mutant positions that the data will be sampled from

        split : float, default=0.8, range [0-1]

            The split point for the training - validation data.

        shuffle : Bool, default=True

            Determines if the data will be shuffled prior to returning.

        returns : x_train, y_train, x_test, y_test

            All Nx1 arrays with train as the first 80% of the shuffled data and test
            as the latter 20% of the shuffled data.
        """

        assert (0 <= split <= 1), "Split must be between 0 and 1"

        if data is not None:
            data = data
        elif idxs is not None:
            data = copy.copy(self.tokenized[idxs])
        else:
            data = copy.copy(self.tokenized)

        if shuffle:
            np.random.shuffle(data)

        if scaler:
            fitnesses = data[:,-1].reshape(-1,1)
            scaler.fit(fitnesses)
            data[:,-1] = scaler.transform(fitnesses).reshape(-1)

        split_point = int(len(data)*split)

        # Y data selects only the last column of Data, X selects the rest
        train, test = data[:split_point], data[split_point:]

        x_train, x_test = train[:,:-1], test[:,:-1]
        y_train, y_test = train[:,-1], test[:,-1]

        return x_train.astype("int"), y_train.astype("float"), \
               x_test.astype("int"), y_test.astype("float")

    def gen_dataloaders(self,labels,keys,params,split_point):
        """
        Function that generates PyTorch dataloaders from a labels dictionary and a list of keys
        with an associated parameter dictionary.

        Parameters
        ----------
        labels : dict, {seq[could be tokenized] : fitness}

            Label dictionary that links all sequence labels with fitness values

        keys : list, [seqs]

            List of all sequences, used as keys in dataset

        params : dict, {param : value}

            Dictionary of dataloader parameters that are unpacked into PyTorch dataloader class

        split_point : float, default=0.8, 0 <= split_point <= 1

            Determines what fraction of data goes into training and test dataloaders

        Returns
        -------
            Returns a training and testing dataloader
        """
        partition = {"train" : keys[:int(len(keys)*split_point)],
                     "test"  : keys[int(len(keys)*split_point):]}
        training_set = Dataset(partition["train"], labels)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        test_set = Dataset(partition["test"], labels)
        test_generator = torch.utils.data.DataLoader(test_set, **params)
        return training_generator, test_generator

    def pytorch_dataloaders(self,
                            tokenize=True,
                            split_point=0.8,
                            idxs=None,
                            distance=False,
                            positions=None,
                            params={"batch_size"  : 500,
                                    "shuffle"     : True,
                                    "num_workers" : 8},
                            unsupervised=False,
                            real_label = 0 # Used for GANs, ignored for VAEs
                            ):
        """
        Function that wraps gen_dataloaders and determines what kind of dataloaders will be returned.

        Parameters:
        -----------
        tokenize : Bool, default=True

            Determines if the dataloaders should return tokenized values or not

        split_point : float, default=0.8, 0 <= split_point <= 1

            Determines what fraction of data goes into training and test dataloaders

        idxs : np.array[int], default=None

            Indexes which will be used to create a subset of the data before the other operations are applied.

        distance : int or [int], default=False

            A single, or list of integers that specify the distances from the seed sequence
            that will be returned in the dataloader

        positions : [int], default=None

            A list of integers which specify mutated positions in the sequence.

        params : dict, {param : value}

            Dictionary of dataloader parameters that are unpacked into PyTorch dataloader class

        unsupervised : Bool, default=False

            Determines if the dataloaders that should be returned will have placeholder fitnesses

        real_label : Bool, default=0

            Value used to assign ground truth status for algorithms such as GANs. 0 is the default
            to enable better gradient movement
        """
        if tokenize:
            stored_data = self.tokenized
        else:
            stored_data = self.data

        if idxs is not None:
            data = copy.copy(stored_data[idxs])
        else:
            data = copy.copy(stored_data)

        if unsupervised:
            labels = {torch.Tensor(x[:-1].astype('int8')).long() : real_label for x in data}
        else:
            labels = {torch.Tensor(x[:-1].astype('int8')).long() : x[-1] for x in data}

        keys   = list(labels.keys())

        return self.gen_dataloaders(labels=labels, keys=keys, params=params, split_point=split_point)

    ############################################################################
    ############################ Machine Learning ##############################
    ############################################################################

    def fit(self, model, model_args, save_model=False,**kwargs):
        """
        Uses ths sklearn syntax to fit the data to model that is provided as a few arguments.

        Parameters
        ----------
        model : sklearn or skorch model architecture.

        model_args : {keyword : argument}

            Arguments that will be unpacked when the model is instantiated.

        save_model : bool, default=False

            The boolean value for whether or not the model will be saved to self.learner

        kwargs : {keyword : argument}

            Arguments that will be provided to sklearn_data to perform the necessary splits
        """
        x_train, y_train, x_test, y_test = self.sklearn_data(**kwargs)
        print(len(x_train))
        model = model(**model_args)
        if model.__class__.__name__ == "NeuralNetRegressor":
            y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)
        print(f"Training model {model}")
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        print(f"Model score on training data: {train_score}")
        test_score = model.score(x_test, y_test)
        print(f"Score of {model} on testing data is {test_score}")
        if save_model:
            self.learners[f"{model}"] = model
        return train_score, test_score

    ############################################################################
    ################################ Utilities #################################
    ############################################################################

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
    test = Protein_Landscape(csv_path="../Data/NK/K4/V1.csv")
