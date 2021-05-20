import os
import copy
import time
import torch
import random
import pickle
import operator
import numpy as np
import tqdm as tqdm
import pandas as pd
import networkx as nx
import sklearn.utils as skutils
import multiprocessing as mp
from colorama import Fore, Style
from functools import partial, reduce
from .utils import Dataset, load, save
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .protein import Protein
from .distance import hamming

class Prograph():
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

        Storage of the graph that can be passed to graph visualisation packages

    Written by Adam Mater, last revision 18.5.21
    """
    def __init__(self,data=None,
                      seed_seq=None,
                      gen_graph=False,
                      csv_path=None,
                      seqs_col="Sequence",
                      columns=["Fitness"],
                      amino_acids='ACDEFGHIKLMNPQRSTVWY',
                      saved_file=None):
        if saved_file:
            try:
                print(f"Trying to load {saved_file}")
                self = load(saved_file,pgraph=self)
                return None
            except:
                raise FileNotFoundError("File could not be opened")

        if (not data and not csv_path):
            print("Initializing empty protein graph")
            return

        if csv_path and data:
            print(f"""Both a filepath ({csv_path}) and a data array has been provided. The CSV is
                     given higher priority so the data will be overwritten by this.""")
            self.csv_path = csv_path
            sequences, prot_data = self.csvDataLoader(csv_path,seqs_col=seqs_col,columns=columns)

        elif csv_path:
            self.csv_path = csv_path
            sequences, prot_data = self.csvDataLoader(csv_path,seqs_col=seqs_col,columns=columns)

        self.amino_acids     = amino_acids
        self.gen_graph       = gen_graph
        self.columns         = columns
        self.tokens          = {x:y for x,y in zip([x.encode("utf-8") for x in self.amino_acids], range(1,len(self.amino_acids)+1))}
        self.graph = {idx : Protein(sequence) for idx,sequence in enumerate(sequences)}

        self.df_graph

        for axis in columns:
            self.update_graph(prot_data[axis],axis)
        # This is a reverse of the graph dictionary to support querying by sequence instead of index
        self.seq_idxs        = {seq : idx for idx, seq in enumerate(sequences)}

        if seed_seq:
            self.seed        = Protein(seed_seq)
        else:
            self.seed        = self.graph[0]
        self.seq_len = len(self.seed)
        self.len = len(self)

        self.tokenized = self.tokenize(sequences)
        self.update_graph([tuple(x) for x in self.tokenized],"tokenized")
        self.token_dict = {tuple(seq) : idx for idx,seq in enumerate(self.tokenized)}

        self.mutated_positions = self.calc_mutated_positions()
        self.sequence_mutation_locations = self.boolean_mutant_array(self.seed.seq)
        # Stratifies data into different hamming distances

        self.mutation_arrays  = self.gen_mutation_arrays()
        self.mode = "graph"

        # Contains the information to provide all mutants 1 amino acid away for a given sequence
        self.len = len(self)

        if self.gen_graph:
            self.update_graph(self.build_graph(),"neighbours")

        self.learners = {}
        print(self)

    def __str__(self):
        distances = hamming(self.tokenized,self.tokenized[self.query(self.seed.seq)].reshape(1,-1))
        # TODO Change print formatting for seed sequence so it doesn't look bad
        return f"""
            Protein Landscape class
            Number of Sequences : {len(self)}
            Max Distance        : {torch.max(distances)}
            Longest Sequence    : {np.max([len(x) for x in self("seq")])}
            Number of Distances : {len(np.unique(distances))}
            Seed Sequence       : {self.coloured_seed_string()}
                Modified positions are shown in green"""

    def __repr__(self):
        return f"""Protein_Landscape(seed_seq='{self.seed.seq}',
                                  gen_graph='{self.gen_graph}'
                                  csv_path='{self.csv_path}',
                                  columns={self.columns},
                                  amino_acids='{self.amino_acids}')"""

    def __len__(self):
        return len(self.graph)

    def __getitem__(self,idx):
        """
        Customised __getitem__ that supports both integer and array indexing.
        """
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            return {x : self.graph[x] for x in self.query(idx)}
        else:
            return self.graph[self.query(idx)]

    def __call__(self,label,**kwargs):
        return self.label_iter(label,**kwargs)

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
                idx = [self.seq_idxs.get(seq, "This sequence is not in the dataset.") for seq in sequence]
            else:
                print("Wrong data format in numpy array or list iterable.")

        elif isinstance(sequence, str):
            idx = self.seq_idxs.get(sequence, "This sequence is not in the dataset.")

        elif isinstance(sequence, tuple):
            assert len(sequence) == self.seq_len, "Tuple not valid length for dataset."
            check = np.where(np.all(sequence == self.tokenized,axis=1))[0]
            assert len(check) > 0, "Not a valid tuple representation of a protein in this dataset."
            idx = int(check)

        else:
            raise ValueError("Input format not understood.")

        if information:
            assert self.graph is not None, "To provide additional information, the graph must be computed"
            if isinstance(idx, np.ndarray) or isinstance(idx, list):
                return {ix : self.graph[ix] for ix in idx}
            else:
                return self.graph[idx]

        else:
            return idx

    def gen_mutation_arrays(self):
        """
        Generates mutation arrays

        TODO finish writing docstrings
        """
        xs = np.arange(self.seq_len*len(self.amino_acids))
        ys = np.array([[y for x in range(len(self.amino_acids))] for y in range(self.seq_len)]).flatten()
        modifiers = np.array([np.arange(len(self.amino_acids)) for x in range(self.seq_len)]).flatten()
        return (xs, ys, modifiers)

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
            reference_seq = self.seed.seq
        d_data          = hamming(self.tokenized,self.tokenized[self.query(reference_seq)].reshape(1,-1))

        if distances is not None:
            if type(distances) == int:
                distances = [distances]
            assert type(distances) == list, "Distances must be provided as integer or list"
            for d in distances:
                assert d in np.unique(d_data), f"{d} is not a valid distance"
            # Uses reduce from functools package and the union1d operation
            # to recursively combine the indexing arrays.
            idxs.append(reduce(np.union1d, [np.where(d_data == d)[1] for d in distances]))

        if positions is not None:
            # This code uses bitwise operations to maximize speed.
            # It first uses an or gate to collect every one where the desired position was modified
            # It then goes through each position that shouldn't be changed, and uses three logic gates
            # to switch ones where they're both on to off, returning the indexes of strings where ONLY
            # the desired positions are changed.
            not_positions = [x for x in range(len(self[reference_seq])) if x not in positions]
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

    def generate_mutations(self,seq):
        """
        Takes a sequence and generates all possible mutants 1 Hamming distance away
        using array substitution

        Parameters:
        -----------
        seq : np.array[int]
            Tokenized sequence array
        """
        seq = self.tokenized[self.query(seq)]
        seed = self.seed
        hold_array = np.zeros(((self.seq_len*len(self.amino_acids)),self.seq_len))
        for i,char in enumerate(seq):
            hold_array[:,i] = char

        xs, ys, mutations = self.mutation_arrays
        hold_array[(xs,ys)] = mutations
        copies = np.invert(np.all(hold_array == seq,axis=1))
        return hold_array[copies]

    def neighbours(self, seq, keys=[["seq"]]):
        """ A simple wrapper function that will return the dictionary containing neighbours for a particular sequence """
        return self[self.graph[self.query(seq)]["neighbours"]]

    def label_iter(self, label, **kwargs):
        """
        Helper function that generates an iterable from the protein graph.

        Parameters
        ----------
        label : str
            A string identifier for the label that will have an iterable generated for it.
        """
        if label == "pytorch":
            return self.pytorch_dataloaders(**kwargs)
        elif label == "sklearn":
            return self.sklearn_data(**kwargs)
        else:
            return np.array(list((protein[label] for protein in self.graph.values())))

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
    def csvDataLoader(csvfile,seqs_col,columns="all",index_col=None,):
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

        Returns
        -------
        np.array (Nx2), where N is the number of rows in the csv file
            Returns an Nx2 array with the first column being x_data (sequences), and the second being
            y_data (fitnesses)
        """
        data         = pd.read_csv(csvfile,index_col=index_col)
        sequences    = data[seqs_col].to_numpy()
        if columns == "all":
            columns = list(data.keys()).remove(seqs_col)
        protein_data = data[columns]
        return sequences, protein_data

    def custom_tokenize(self,seq,tokenizer=None):
        """
        Simple static method which tokenizes an individual sequence

        Parameters
        ----------
        seq : str
            The sequence to be tokenized.

        tokenizer : func, default=None
            The function that will be used to tokenize the string.
        """
        if tokenizer == None:
            return np.array([self.tokens[aa.encode("utf-8")] for aa in seq])
        else:
            return "This feature is not ready yet"

    def tokenize(self, sequences):
        """
        Takes any number of sequences provided as one amino acid strings and returns
        an array of their tokenized form.

        Sequences can be of varying length and the shorter sequences will be padded
        with zeros.

        Parameters
        ----------
        sequences : str, iterable, np.array
            The sequences that will be tokenized using self.tokens
        """
         # Convert list  or array of strings into individual char byte array
        sequences = np.array(sequences,dtype="bytes").reshape(-1,1).view("S1")
        # Initiailise empty tokenized array
        tokenized = np.zeros(sequences.shape,dtype=int)
        for char, token in self.tokens.items():
            tokenized[np.where(sequences == char)] = token

        return tokenized

    def boolean_mutant_array(self,seq=None):
        return self.tokenized != self.tokenized[self.query(seq)]

    def calc_mutated_positions(self):
        """
        Determines all positions that were modified experimentally and returns the indices
        of these modifications.

        Because the Numpy code is tricky to read, here is a quick breakdown:
            self.tokenized is called, and the fitness column is removed by [:,:-1]
            Each column is then tested against the first
        """
        mutated_bools = np.invert(np.all(self.tokenized == self.tokenize(self.seed.seq),axis=0)) # Calculates the indices all of arrays which are modified.
        mutated_idxs  = mutated_bools * np.arange(1,len(self.seed) + 1) # Shifts to the right so that zero can be counted as an idx
        return mutated_idxs[mutated_idxs != 0] - 1 # Shifts it back

    def coloured_seed_string(self):
        """
        Printing function that prints the original seed string and colours the positions that
        have been modified
        """
        strs = []
        idxs = self.mutated_positions
        for i,char in enumerate(self.seed.seq):
            if i in idxs:
                strs.append(f"{Fore.GREEN}{char}{Style.RESET_ALL}")
            else:
                strs.append("{0}".format(char))
        return "".join(strs)


    ############################################################################
    ##################### Graph Generation and Manipulation ####################
    ############################################################################

    def calc_neighbours(self,seq,eps=1,distance=hamming,comp=operator.eq):
        """
        Calcualtes the neighbours for a given sequence in the presence of a cutoff value.

        Parameters
        ----------
        seq : str, int, tup
            An identifier for the string of interest

        eps : int, float
            The distance cutoff for the two to be considered neighbours

        distance : <function>, default=hamming
            A distance function that must match the syntax of those in the distance module.

        comp : <function _operator>, default=operator.eq
            An operator function that will be used to compare the epsilon value to the comparison vector.
        """
        return np.where(comp(hamming(self.tokenized,self.tokenized[self.query(seq)].reshape(1,-1)),eps))[1] # Select the columns indexes.

    def nearest_neighbour(self,seq,distance=hamming,batch_size=8):
        # Todo correctly implement batching code - maybe make it into its own function.
        """
        Calculates the nearest neighbour a sequence to the dataset in accordance with a given distance metric.

        Parameters
        ----------
        seq : str, int, tup
            An identifier for the string of interest

        distance : <function>, default=hamming
            A distance function that must match the syntax of those in the distance module.

        Returns
        -------
        (nearest Protein, distance)
        """
        results = []
        for batch in tqdm.tqdm(list(self.get_every_n(self.tokenize(seq),n=batch_size))):
            results.append([x.cpu().numpy() for x in torch.where(comp(hamming(gpu_tokenized,batch),eps))])

        distances = distance(self.tokenized,self.tokenize(seq)).numpy()
        idx = np.argmin(distances,axis=1)
        return self[idx], np.min(distances)

    @staticmethod
    def get_every_n(a, n=2):
        """
        Splits an array (numpy or pytorch) into chunks of size n and returns a generator.
        """
        for i in range(a.shape[0] // n):
            yield a[n*i:n*(i+1)]

    @staticmethod
    def prod_neighbours(index,out):
        """
        Function to convert output of np.where into arrays of neighbours.
        """
        results = {}
        idxs,neighbours = out
        idxs = idxs + (index * 8)
        for idx in np.unique(idxs):
            results[idx] = neighbours[np.where(idxs == idx)]
        return results

    def build_graph(self,
                    idxs=None,
                    batch_size=8,
                    eps=1,
                    distance=hamming,
                    comp=operator.eq):
        """
        Function to build the protein graph using GPU accelerated pairwise distance calculations.

        Parameters
        ----------
        idxs : np.array, default=None
            Integers of the graph to consider as a subgraph of the entire dataset.

        batch_size : int, default=8
            The size of the batches that will be used to compute pairwise distances on the GPU.
            The batch chunks will have size (batch_size x len(self)), and as such batch_size should
            not be too large as otherwise it will not fit into memory.

        eps : numeric, default = 1
            The epsilon value that defines the local neighbourhood around the value of interest.

        distance : function, default=hamming
            The vectorized distance function to use.

        comp : <function _operator>, default=operator.eq
            The operator that will be used to compare the epsilon value and the batch of distances.
        """
        if idxs is None:
            idxs = torch.arange(len(self))
        gpu_tokenized = torch.as_tensor(self.tokenized[idxs,:].astype(np.float16),dtype=torch.float16,device=torch.device("cuda:0"))

        results = []
        for batch in tqdm.tqdm(list(self.get_every_n(gpu_tokenized,n=batch_size))):
            results.append([x.cpu().numpy() for x in torch.where(comp(distance(gpu_tokenized,batch),eps))])

        final = []
        for idx, result in enumerate(results):
            final.append(self.prod_neighbours(idx,result))

        neighbour_dict = {k: v for d in final for k, v in d.items()}
        completed = []
        for i in range(len(gpu_tokenized)):
            completed.append(neighbour_dict.get(i,np.array([],dtype=np.int)))

        return completed


    def update_graph(self, data, label):
        """
        Function that updates the internal graph structure. The data array must have the same
        order as protein sequences in the original graph.

        Parameters
        ----------
        data : iterable
            An iterable that contains the labels that will be used to update the graph.
            The iterable must support positional indexing as its values will be accessed
            as data[idx], where index is an integer.

        label : str
            A string that will be used as the key for this property in each Protein object
        """
        for idx, protein in self.graph.items():
            setattr(protein, label, data[idx])
        return None

    def graph_to_networkx(self,labels=None,update_self=False,iterable="seq"):
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
        sequences = self.label_iter(iterable)
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

    def degree(self):
        """
        Method to calculate the degree of each node in the graph.

        Returns
        -------
        degrees : np.array,
            A numpy array where each value is the degree of the corresponding protein node in the graph.
        """
        degrees = np.zeros((len(self),),dtype=np.int)
        for i in range(len(self)):
            degrees[i] = len(self[i].neighbours)
        return degrees

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

    def sklearn_data(self,
                     data=None,
                     idxs=None,
                     token_form="tokenized",
                     labels=["Fitness"],
                     split=[0.8,0,0.2],
                     scaler=False,
                     shuffle=True,
                     random_state=0):
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

        split : [float], default=[0.8,0,0.2], range [0-1,0-1,0-1]
            The data percentage in the [training,validation,testing] data.

        shuffle : Bool, default=True
            Determines if the data will be shuffled prior to returning.

        random_state : int, default=0
            The random state seed used to shuffle the arrays.

        returns : x_train, y_train, x_val, y_val, x_test, y_test
            All Nx1 arrays with train as the first 80% of the shuffled data and test
            as the latter 20% of the shuffled data.
        """
        if isinstance(split,int):
            split = [split,0,1-split]
        assert sum(split) <= 1, "The sum of the split terms must be between 0 and 1"

        if data is not None:
            data = data
        elif idxs is not None:
            tokenized,labels = self(token_form)[idx], np.vstack([self(label)[idx] for label in labels]).T
        else:
            tokenized,labels = self(token_form), np.vstack([self(label) for label in labels]).T

        if shuffle:
            tokenized, labels = skutils.shuffle(tokenized,labels,random_state=random_state)

        if scaler:
            fitnesses = labels.reshape(-1,1)
            scaler.fit(fitnesses)
            labels = scaler.transform(fitnesses).reshape(-1)

        split1, split2 = int(len(tokenized)*split[0]), int(len(tokenized)*sum(split[:2]))

        x_train, x_val, x_test = tokenized[:split1], tokenized[split1:split2], tokenized[split2:]
        y_train, y_val, y_test = labels[:split1],   labels[split1:split2],  labels[split2:]

        return x_train.astype("int"), y_train.astype("float"), \
               x_val.astype("int"),   y_val.astype("float"), \
               x_test.astype("int"),  y_test.astype("float")

    def gen_dataloaders(self,labels,keys,params,split_points):
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
        split1, split2 = split_points

        partition = {"train" : keys[:split1],
                     "val"   : keys[split1:split2],
                     "test"  : keys[split2:]}
        training_set = Dataset(partition["train"], labels)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        test_set = Dataset(partition["test"], labels)
        test_generator = torch.utils.data.DataLoader(test_set, **params)
        return training_generator, test_generator

    def pytorch_dataloaders(self,
                            tokenize=True,
                            split=[0.8,0,0.2],
                            idxs=None,
                             token_form="tokenized",
                             labels=["Fitness"],
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
        if idxs is not None:
            tokenized,labels = self(token_form)[idx], np.vstack([self(label)[idx] for label in labels]).T
        else:
            tokenized,labels = self(token_form), np.vstack([self(label) for label in labels]).T

        if unsupervised:
            labels = {torch.Tensor(tokenize.astype('int8')).long() : real_label for tokenize in tokenized}
        else:
            labels = {torch.Tensor(tokenize.astype('int8')).long() : label for tokenize,fit in zip(tokenized,labels)}

        keys   = list(labels.keys())

        split_points = [int(len(tokenized)*split[0]), int(len(tokenized)*sum(split[:2]))]

        return self.gen_dataloaders(labels=labels, keys=keys, params=params, split_points=split_points)

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

        Example Syntax - landscape.fit(LinearRegressor,{"fit_intercept" : True})
            Fits a sklearn linear regressor with the fit intercept attribute set to True.
        """
        x_train, y_train, x_test, y_test = self("sklearn",**kwargs)
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
