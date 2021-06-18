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
from scipy import sparse
from colorama import Fore, Style
from functools import partial, reduce
from .utils import Dataset, save
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .protein import Protein
from .distance import hamming, minkowski
from .utils import flatten

distances = {"hamming" : hamming,
             "minkowski" : minkowski}

class Prograph():
    """
    Class that handles a protein dataset. The graph is initialized by passing a csv
    file to the file argument. By default, the code recognises the sequences by a column
    titled "Sequence". To modify this pass a new string argument to seqs_col. This information
    is combined with an arbitrary number of labels columns. The default is just a column entitled "Fitness"
    but any number can be passed and the labels will be associated with each node.

    Finally, if neighbours are already present in the csv file they will be recognised and not
    recalculated.

    Parameters
    ----------
    file : str,default=None
        Path to the dataframe file that should be imported using CSV loader function

    seed_seq : str, default=None
        Enables the user to explicitly provide the seed sequence as a string. Otherwise it
        will default to the first sequence in the dataset.

    seqs_col : str, default="Sequence"
        The string that identifies the sequence data within the csv file.

    columns : ["Fitness"]
        The columns to extract from the provided csv file.

    index_col : str
        The index column of the csv datafile.

    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'
        String containing all allowable amino acids for tokenization functions

    Attributes
    ----------
    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'
        String containing all allowable amino acids in tokenization functions

    sequence_mutation_locations : np.array(bool)
        Array that stores boolean values with Trues indicating that the position is
        mutated relative to the seed sequence

     mutated_positions: np.array(int)
        Numpy array that stores the integers of each position that is mutated

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

    graph : pd.DataFrame
        Storage efficient form of the graph as a pandas dataframe.
        The graph must contain three columns - ["Sequence","Tokenized","Neighbours"].
        When saved, the csv will have the tokenized form removed as this is quite
        a cheap calculation and requires a large amount of storage, and thus not worth it.

        Any neighbour column calculated through either epsilon or kNN methods will have the values
        stored as tuples (neighbours,weights).

    Written by Adam Mater, last revision 10.6.21
    """
    def __init__(self,file,
                      seed_seq=None,
                      seqs_col="Sequence",
                      columns=["Fitness"],
                      index_col=0,
                      amino_acids='ACDEFGHIKLMNPQRSTVWY'):
        try:
            ext = file.split(".")[-1]
            if ext  == "csv":
                self.graph = self.csvDataLoader(file,seqs_col=seqs_col,columns=columns,index_col=index_col)
            elif ext == "pkl":
                self.graph = pd.read_pickle(file)

        except:
            raise FileNotFoundError("File could not be opened")

        self.file            = file
        self.seed_seq        = seed_seq
        self.seqs_col        = seqs_col
        self.columns         = columns
        self.index_col       = index_col
        self.amino_acids     = amino_acids

        if seed_seq:
            self.seed        = Protein(seed_seq)
        else:
            self.seed        = Protein(**self.graph.loc[0])
        self.seq_len = len(self.seed)
        self.len = len(self)

        # The amino acids are encoded as utf-8 to facilitate fact array substitution in the tokenize function.
        self.tokens          = {x:y for x,y in zip([x.encode("utf-8") for x in self.amino_acids], range(1,len(self.amino_acids)+1))}
        self.tokenized = self.tokenize(self.graph[seqs_col])

        # This is a reverse of the graph dictionary to support querying by sequence instead of index
        self.token_dict = {tuple(seq) : idx for idx,seq in enumerate(self.tokenized)}
        self.seq_idxs        = {seq : idx for idx, seq in enumerate(self.graph[seqs_col])}

        self.mutated_positions = self.calc_mutated_positions()
        self.sequence_mutation_locations = self.boolean_mutant_array(self.seed.Sequence)
        self.mutation_arrays  = self.gen_mutation_arrays()

        if "Tokenized" not in self.graph:
            self.graph["Tokenized"] = [x for x in self.tokenized]
        if "Neighbours" not in self.graph:
            self.graph["Neighbours"] = [x for x in self.build_graph(eps=1)]

        self.learners = {}
        print(self)

    def __str__(self):
        distances = hamming(self.tokenized,self.tokenized[self.query(self.seed.Sequence)].reshape(1,-1))
        # TODO Change print formatting for seed sequence so it doesn't look bad
        return f"""
            Prograph
            Number of Sequences : {len(self)}
            Max Distance        : {torch.max(distances)}
            Longest Sequence    : {np.max([len(x) for x in self("Sequence")])}
            Number of Distances : {len(np.unique(distances))}
            Seed Sequence       : {self.coloured_seed_string()}
                Modified positions are shown in green"""

    def __repr__(self):
        return f"""Prograph(file={self.file},
                            seed_seq='{self.seed.Sequence}',
                            seqs_col='{self.seqs_col}',
                            columns={self.columns},
                            index_col={self.index_col},
                            amino_acids='{self.amino_acids}')"""

    def __len__(self):
        """
        Calculates the length of the self object by calculating the number of entries in the internal
        self.graph dataframe.
        """
        return len(self.graph)

    def __getitem__(self,idx):
        """
        Customised __getitem__ that supports both integer and array indexing.
        """
        return self.graph.iloc[self.query(idx)]

    def __call__(self,label=None,**kwargs):
        """
        Overloaded call function to pass arguments to label_iter.
        """
        return self.label_iter(label,**kwargs)

    def label_iter(self, label, **kwargs):
        """
        Helper function that generates an iterable from the protein graph. A copy of corresponding
        portion of the dataframe is returned in order to ensure that the underlying data remains untouched.

        Parameters
        ----------
        label : str
            A string identifier for the label that will have an iterable generated for it.
        """
        if label == "pytorch":
            return self.pytorch_dataloaders(**kwargs)
        elif label == "sklearn":
            return self.sklearn_data(**kwargs)
        elif label == None:
            return self.graph.copy()
        else:
            return self.graph[label].copy()

    def query(self,sequence):
        """
        Helper function that interprets a query sequence and accepts multiple formats.
        This object works by moving indexes of sequences around due to the large increase
        in computational efficiency, and as a result this function returns the index associated
        with this sequence

        Parameters
        ----------
        sequence : str, tuple, int
            The sequence to query against the dataset. Multiple valid input formats
        """
        if isinstance(sequence, int) or isinstance(sequence, np.integer):
            assert sequence <= self.len, "Index exceeds bounds of dataset"
            idx = sequence

        elif isinstance(sequence, np.ndarray) or isinstance(sequence, list):
            if isinstance(sequence[0], int) or isinstance(sequence[0],np.integer) or isinstance(sequence[0], np.bool_):
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

        return idx

    def positions(self,positions):
        """
        Helper function that passes a set of positions to indexing.
        """
        return self.indexing(positions=positions)

    def distances(self,distances):
        """
        Helper function that passes a set of distances to indexing.
        """
        return self.indexing(distances=distances)

    def indexing(self,
                 reference_seq=None,
                 distances=None,
                 positions=None,
                 percentage=None,
                 Bool="or",
                 complement=False):
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
            reference_seq = self.seed.Sequence

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
            not_positions = [x for x in range(len(self[reference_seq]["Sequence"])) if x not in positions]
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

    def neighbours(self, seq, keys=[["Sequence"]]):
        """ A simple wrapper function that will return the dictionary containing neighbours for a particular sequence """
        return self[self.graph[self.query(seq)]["Neighbours"]]

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

    def gen_mutation_arrays(self):
        """
        Generates mutation arrays. These arrays can be combined to generate all single mutants
        from a given single sequence.
        """
        xs = np.arange(self.seq_len*len(self.amino_acids))
        ys = np.array([[y for x in range(len(self.amino_acids))] for y in range(self.seq_len)]).flatten()
        modifiers = np.array([np.arange(len(self.amino_acids)) for x in range(self.seq_len)]).flatten()
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
        seq = self.tokenized[self.query(seq)]
        seed = self.seed
        hold_array = np.zeros(((self.seq_len*len(self.amino_acids)),self.seq_len))
        for i,char in enumerate(seq):
            hold_array[:,i] = char

        xs, ys, mutations = self.mutation_arrays
        hold_array[(xs,ys)] = mutations
        copies = np.invert(np.all(hold_array == seq,axis=1))
        return hold_array[copies]

    @staticmethod
    def csvDataLoader(csvfile,seqs_col,columns="all",index_col=None):
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
        if columns == "all":
            columns = list(data.keys())
            columns.remove(seqs_col)
        columns = [seqs_col] + columns
        if "Neighbours" in data:
            columns += ["Neighbours"]
        return data[columns]

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

    def embedding(self, embedded, name):
        """
        Flexible function to add an embedded representation of each sequence in the dataset.

        embedded : np.array()
            The array of embedded representations. Order must match the original sequence order.

        name : str
            A name that will be used to identify the embedding in the graph structure.
        """
        self.graph[f"{name}_embedded"] = embedded

    def boolean_mutant_array(self,seq=None):
        """
        Generates an array that highlights all mutated positions in the dataset relative to a seed sequence.
        """
        return self.tokenized != self.tokenized[self.query(seq)]

    def calc_mutated_positions(self):
        """
        Determines all positions that were modified experimentally and returns the indices
        of these modifications.

        Because the Numpy code is tricky to read, here is a quick breakdown:
            self.tokenized is called, and the fitness column is removed by [:,:-1]
            Each column is then tested against the first
        """
        mutated_bools = np.invert(np.all(self.tokenized == self.tokenize(self.seed.Sequence),axis=0)) # Calculates the indices all of arrays which are modified.
        mutated_idxs  = mutated_bools * np.arange(1,len(self.seed) + 1) # Shifts to the right so that zero can be counted as an idx
        return mutated_idxs[mutated_idxs != 0] - 1 # Shifts it back

    def coloured_seed_string(self):
        """
        Printing function that prints the original seed string and colours the positions that
        have been modified
        """
        strs = []
        idxs = self.mutated_positions
        for i,char in enumerate(self.seed.Sequence):
            if i in idxs:
                strs.append(f"{Fore.GREEN}{char}{Style.RESET_ALL}")
            else:
                strs.append("{0}".format(char))
        return "".join(strs)


    ############################################################################
    ##################### Graph Generation and Manipulation ####################
    ############################################################################

    def calc_neighbours(self,seq,eps=1,distance=hamming,comp=operator.eq,weights=False):
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
        return np.where(comp(distance(self.tokenized,self.tokenized[self.query(seq)].reshape(1,-1)),eps))[1] # Select the columns indexes.

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

    def neighbourhood(self,seq,eps,distance=hamming):
        """
        A neighbourhood function that takes a sequence and epsilon value and produces all sequences which fall within
        this neighbourhood.

        Parameters
        ----------
        seq : idx, str, tuple
            The sequence that will form the locus of the neighbourhood.

        eps : numeric
            The episolon value that defines the radius of the neighbourhood.

        distance : prograph.distance.function, default=hamming
            The distance function uses to calculate the distances from the locus sequence.
        """
        distances = hamming(self.tokenized,np.atleast_2d(self.tokenized[self.query(seq)])) <= eps
        return self[distances.numpy().flatten()]

    def neighbourhood_clustering(self,eps,distance=hamming):
        """
        Utilises the neighbourhood function to compute clusters within the dataset.

        Parameters
        ----------
        seq : idx, str, tuple
            The sequence that will form the locus of the neighbourhood.

        eps : numeric
            The episolon value that defines the radius of the neighbourhood.

        distance : prograph.distance.function, default=hamming
            The distance function uses to calculate the distances from the locus sequence.

        # I am not sure if this is actually clustering, as entities are capable of being in multiple clusters simultaneously.
        """
        sequences = self("Sequence")
        clusters = {}
        skips = set()
        for i,seq in tqdm.tqdm(enumerate(sequences)):
            if i not in skips:
                neighbours = self.neighbourhood(seq,eps,distance)
                clusters[i] = neighbours
                skips = skips | set(neighbours.index)
        return clusters

    @staticmethod
    def get_every_n(a, n=2):
        """
        Splits an array (numpy or pytorch) into chunks of size n and returns a generator.
        """
        # Line below performs integer division but rounds result up so that final chunk is also returned.
        for i in range((a.shape[0] // n) + (a.shape[0] % n > 0)):
            yield a[n*i:n*(i+1)]

    @staticmethod
    def prod_neighbours(index,out,batch_size,weights=None):
        """
        Function to convert output of np.where into arrays of neighbours.

        Parameters
        ----------
        index : int
            The index of the batch in which the neighbours were calculated.

        out : [torch.Tensor, torch.Tensor]
            The row,column indexes respectively for the neighbours in the batches.
            Is unpacked within the code into row,col objects.

        weights : np.array
            The numpy array of weights that will be assigned to each edge.

        batch_size : int
            The size of the batch as an integer.
        """
        results = {}
        row,col = out
        row = row + (index * batch_size) # Scale row indexes up to batch index.
        if weights is None:
            weights = np.ones(col.shape)

        for idx in np.unique(row):
            results[idx] = (col[np.where(row == idx)],weights[np.where(row == idx)]) # Isolate the correct elements of the neighbours array
        return results

    def build_graph(self,
                    idxs=None,
                    batch_size=8,
                    eps=None,
                    k=None,
                    weighted=False,
                    similarity=False,
                    representation="Tokenized",
                    distance=hamming,
                    comp=operator.le):
        """
        Function to build the protein graph using GPU accelerated pairwise distance calculations.
        The function contains two different generation methods - knn graph and epsilon neigbourhood.

        kNN Graph generation involves calculating all pairwise distances, sorting the values, and selecting
        the smallest k values as neighbours.

        .. math::
            \sum_i^k {x_i * y_i}

        Epsilon neighbourhood graph generation calculates all pairwise distances and compares them against a
        threshold value (epsilon). All nodes that are less than or equal to this threshold are added as
        neighbours of the sequence in question.

        .. math::
            x = 3

        Parameters
        ----------
        idxs : np.array, default=None
            Integers of the graph to consider as a subgraph of the entire dataset.

        batch_size : int, default=8
            The size of the batches that will be used to compute pairwise distances on the GPU.
            The batch chunks will have size (batch_size x len(self)), and as such batch_size should
            not be too large as otherwise it will not fit into memory.

        eps : numeric, default=None
            The epsilon value that defines the local neighbourhood around the value of interest.

        k : int, default=None
            The number of k neighbours for k-nearest neighbour graph construction.

        weighted : bool, default=False
            Whether or not the graph that is generated contains the weights of the connections.

        similariity : bool, default=False
            Whether or not to convert the distances into a similarity metric.

        distance : <prograph.distance function>, default=hamming
            The vectorized distance function to use.

        comp : <function _operator>, default=operator.le
            The operator that will be used to compare the epsilon value and the batch of distances.
        """
        if operator.xor(bool(eps),bool(k)) is False:
            raise ValueError("Epsilon or K must be provided, but both cannot be as they are different methods of graph construction.")
        if k is not None:
            if (not isinstance(k,int)):
                raise TypeError("K must be provided as an integer.")

        if similarity and eps:
            eps = 1/(1+eps) # Convert epsilon euclidean distance into similarity

        if idxs is None:
            idxs = Ellipsis # Used to ensure that a slice is indexed instead of a copy.

        gpu_tokenized = torch.as_tensor(self(representation),dtype=torch.float16,device=torch.device("cuda:0"))[idxs,:]

        weights   = []
        edge_idxs = []
        if eps:
            for batch in tqdm.tqdm(list(self.get_every_n(gpu_tokenized,n=batch_size))):
                distances = distance(gpu_tokenized,batch,similarity=similarity)
                if similarity:
                    locations = torch.where((comp(eps,distances) & (distances < 1))) # Order swapped as similarity is an inverse relationship.
                else:
                    locations = torch.where((comp(distances,eps) & (distances > 0)))

                weights.append(distances[locations].cpu().numpy())
                edge_idxs.append([x.cpu().numpy() for x in locations])
                # Checks two boolean conditions to ensure that nodes are not considered their own neighbours (i.e their distance is >0)

            # Iterate over each batch and use prod_neighbours to alter indices so that they are correct.
            final = []
            for idx, result in enumerate(edge_idxs):
                final.append(self.prod_neighbours(idx,result,batch_size,weights=weights[idx]))

            # Merge dictionaries in final
            neighbour_dict = {k: v for d in final for k, v in d.items()}

            # Make a final pass and assign empty neighbour arrays to and sequences that do not have any neighbours.
            completed = []
            for i in range(len(gpu_tokenized)):
                completed.append(neighbour_dict.get(i,(np.array([],dtype=int),np.array([],dtype=int))))

        else:
            for batch in tqdm.tqdm(list(self.get_every_n(gpu_tokenized,n=batch_size))):
                if similarity:
                    sorted_ds = torch.sort(distance(gpu_tokenized,batch,similarity=similarity),dim=1,descending=True)
                else:
                    sorted_ds = torch.sort(distance(gpu_tokenized,batch,similarity=similarity),dim=1)
                weights.append([x.cpu().numpy() for x in sorted_ds[0][:,1:k+1]])
                edge_idxs.append([x.cpu().numpy() for x in sorted_ds[1][:,1:k+1]])
                # The [:,1:k+1] selects all rows, ignores the first element as the self distance is always 0, and then selects up to k+1 elements
            completed = list(zip(flatten(edge_idxs),flatten(weights)))
        return completed

    def graph_to_networkx(self,labels=None,update_self=False,iterable="Sequence"):
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
        neighbours_iter = [x[0] for x in self("Neighbours")]
        for idx,neighbours in enumerate(tqdm.tqdm(neighbours_iter)):
            g.add_edges_from([(sequences[idx], sequences[neighbour_idx]) for neighbour_idx in neighbours])
        if update_self:
            self.networkx_graph = g
            return None
        else:
            return g

    def degree(self,graph="Neighbours",boolean_weights=False):
        """
        Method to calculate the degree of each node in the graph. If the graph is directed,
        then this method calculates the outdegree of each node.

        Parameters
        ----------
        graph : str, default="Neighours"
            The graph that will be used to calculate the degree

        boolean_weights : bool, default=False
            If True, will replace all weight values with a 1.

        Returns
        -------
        degrees : np.array,
            A numpy array where each value is the degree of the corresponding protein node in the graph.
        """
        degrees = np.zeros((len(self),),dtype=np.float32)
        if boolean_weights:
            for i,edges in enumerate(self(graph)):
                degrees[i] = len(edges[0]) # Access just the number of nodes
        else:
            for i,edges in enumerate(self(graph)):
                degrees[i] = np.sum(edges[1].astype(np.float32)) # Access just the weights term
        return degrees

    def get_neighbour_coords(self,graph="Neighbours",boolean_weights=False):
        """
        Gets the I,J neighbours coordinates to enable sparse matrix construction and operations.

        Parameters
        ----------
        graph : str, default="Neighbours"
            The graph to be used when accessing the matrix construction.

        boolean_weights : bool, default=False
            If True, will replace all weight values with a 1.

        Returns
        -------
        I : np.array
            The row indices of the numpy neighbour positions

        J : np.array
            The column indices of the numpy neighbour positions

        weights : bool, default=False
            The weight associated with each edge.
        """
        I = []
        neighbours,weights = zip(*self(graph))
        for i,J in enumerate(neighbours):
            I.append(np.zeros(len(J),dtype=int) + 1*i)
        I = np.concatenate(I)
        J = np.concatenate(neighbours)
        if boolean_weights:
            return I,J,np.ones(I.shape)
        else:
            weights = np.concatenate(weights)
            return I,J,weights.astype(np.float32)

    def adjacency(self,graph="Neighbours",boolean_weights=False):
        """
        Exports the neighbour data to a sparse adjacency matrix that can be used for later computation.

        Parameters
        ----------
        graph : str, default="Neighbours"
            The graph to be used when accessing the matrix construction.

        boolean_weights : bool, default=False
            If True, will replace all weight values with a 1.
        """
        I, J, V = self.get_neighbour_coords(graph=graph,boolean_weights=boolean_weights)
        return sparse.coo_matrix((V,(I,J)),shape=(len(self),len(self)))

    def laplacian(self, graph="Neighbours",boolean_weights=False,mode="outdegree"):
        """
        Exports the graph laplacian as a sparse matrix for later computation.

        Parameters
        ----------
        graph : str, default="Neighbours"
            The graph to be used when accessing the matrix construction.

        boolean_weights : bool, default=False
            If True, will replace all weight values with a 1.

        mode : str \u2208 ["indegree","outdegree"], default="outdegree"
            Whether or not to use indegree or outdegree.
        """
        L = (-1) * self.adjacency(graph,boolean_weights)
        if mode == "outdegree":
            D = self.degree(graph,boolean_weights)
        elif mode == "indegree":
            D = (-1) * np.array(L.sum(0)).reshape(-1,)
        else:
            raise ValueError("Not a valid degree mode.")
        L.setdiag(D)
        return L

    def dirichlet(self,graph="Neighbours",boolean_weights=False,scaler=MinMaxScaler,mode="outdegree"):
        """
        Calculates the Dirichlet energy of the graph representation.

        Parameters
        ----------
        graph : str, default="Neighbours"
            The graph to be used when accessing the matrix construction.

        boolean_weights : bool, default=False
            If True, will replace all weight values with a 1.

        scaler : sklearn.base.BaseEstimator, default=MinMaxScaler
            A scaler for the fitness values. Defaults to minmax scaler with default bounds (0,1)

        mode : str \u2208 ["indegree","outdegree"], default="outdegree"
            Whether or not to use indegree or outdegree.
        """
        fitness = self("Fitness").to_numpy().reshape(-1,1)
        if scaler is not None:
            scaler = scaler()
            fitness = scaler.fit_transform(fitness)
        L = self.laplacian(graph=graph,boolean_weights=boolean_weights,mode=mode)
        return fitness.T @ L @ fitness

    def local_variance(self,graph="Neighbours",boolean_weights=False,scaler=MinMaxScaler):
        """
        Calculates the local variance for each node in the protein graph.

        Parameters
        ----------
        graph : str, default="Neighbours"
            The graph to be used when accessing the matrix construction.

        boolean_weights : bool, default=False
            If True, will replace all weight values with a 1.

        scaler : sklearn.base.BaseEstimator, default=MinMaxScaler
            A scaler to scale the fitness values. Defaults to minmax scaler with default bounds (0,1)
        """
        variances = np.zeros(len(self))
        fitnesses = self("Fitness").to_numpy()
        scaler = scaler()
        fitnesses = scaler.fit_transform(fitnesses.reshape(-1,1))
        for i,(neighbours,weights) in enumerate(self(graph)):
            I = np.zeros(len(neighbours),dtype=int) + 1*i
            variances[i] = np.mean(fitnesses[I] - fitnesses[neighbours])
        return variances

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
            return np.array([x[["Sequence","Fitness"]] for x in self.graph])
        else:
            return copy.copy(self.tokenized)

    def sklearn_data(self,
                     data=None,
                     idxs=None,
                     representation="Tokenized",
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
            tokenized,labels = np.vstack(self(representation)[idx]), np.vstack([self(label)[idx] for label in labels]).T
        else:
            tokenized,labels = np.vstack(self(representation)), np.vstack([self(label) for label in labels]).T

        if shuffle:
            tokenized, labels = skutils.shuffle(tokenized,labels,random_state=random_state)

        if scaler:
            fitnesses = labels.reshape(-1,1)
            labels = scaler.fit_transform(fitnesses).reshape(-1)

        split1, split2 = int(len(tokenized)*split[0]), int(len(tokenized)*sum(split[:2]))

        x_train, x_val, x_test = tokenized[:split1], tokenized[split1:split2], tokenized[split2:]
        y_train, y_val, y_test = labels[:split1],   labels[split1:split2],  labels[split2:]

        return x_train.astype("float"), y_train.astype("float"), \
               x_val.astype("float"),   y_val.astype("float"), \
               x_test.astype("float"),  y_test.astype("float")

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
        data_generators = {}
        for key in partition:
            if len(partition[key]) == 0:
                continue # skip empty partitions.
            data_set = Dataset(partition[key], labels)
            data_generators[key] = torch.utils.data.DataLoader(data_set, **params)
        return data_generators

    def pytorch_dataloaders(self,
                            split=[0.8,0,0.2],
                            idxs=None,
                            representation="Tokenized",
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
        split : [float], default=[0.8,0,0.2], 0 <= sum(split) <= 1
            Determines what fraction of data goes into training, validation, and test dataloaders

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
            tokenized,labels = np.vstack(self(representation)[idxs]), np.vstack([self(label)[idxs] for label in labels]).T
        else:
            tokenized,labels = np.vstack(self(representation)), np.vstack([self(label) for label in labels]).T

        if unsupervised:
            data_labels = {torch.Tensor(tokenize.astype('float32')).long() : real_label for tokenize in tokenized}
        else:
            data_labels = {torch.Tensor(tokenize.astype('float32')).long() : label for tokenize,label in zip(tokenized,labels)}

        keys   = list(data_labels.keys())

        split_points = [int(len(tokenized)*split[0]), int(len(tokenized)*sum(split[:2]))]

        return self.gen_dataloaders(labels=data_labels, keys=keys, params=params, split_points=split_points)

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
