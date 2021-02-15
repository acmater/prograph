import numpy as np
import pandas as pd
import copy
import time
import random
import pickle
import tqdm as tqdm
import networkx as nx
import multiprocessing as mp
from sklearn_utils import collapse_concat
from functools import partial, reduce
from Dataset import Dataset
import torch

from colorama import Fore
from colorama import Style

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

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

    data : np.array

        Full, untokenized data. Two columns, first is sequences as strings, and second is fitnesses

    tokens : {tuple(tokenized_sequence) : index}

        A dictionary that stores a tuple format of the tokenized string with the index
        of it within the data array as the value. Used to rapidly perform membership checks

    sequences : np.array(str)

        A numpy array containing all sequences as strings

    seed_seq : str

        Seed sequence as a string

    tokenized : np.array, shape(N,L+1)

        Array containing each sequence with fitness appended onto the end.
        For the shape, N is the number of samples, and L is the length of the seed sequence

    mutation_array : np.array, shape(L*20,L)

        Array containing all possible mutations to produce sequences 1 amino acid away.
        Used by maxima generator to accelerate the construction of the graph.

        L is sequence length.

    self.hammings : np.array(N,)

        Numpy array of length number of samples, where each value is the hamming
        distance of the species at that index from the seed sequence.

    max_distance : int

        The maximum distance from the seed sequence within the dataset.

    graph : {idx  : "tokenized"  : tuple(int)     - a Tuple representation of the tokenized protein sequence
                    "string"     : str            - string representation of protein sequence
                    "fitness"    : float          - The fitness value associated with this protein
                    "neighbours" : np.array[idxs] - A numpy array of indexes in the dataset that are neighbours

        A memory efficient storage of the graph that can be passed to graph visualisation packages

    num_minima : int

        The number of minima within the dataset

    num_maxima : int

        The number of maxima within the dataset

    extrema_ruggedness : float32

        The floating point ruggedness of the landscape calculated as the normalized
        number of maxima and minima.

    Written by Adam Mater, last revision 15.2.21
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
                "File could not be loaded"
                raise Exception.FileNotFoundError("File could not be opened")

        if (not data and not csv_path):
            raise Exception.FileNotFoundError("Data must be provided as numpy array or csv file")

        if csv_path and data:
            print("""Both a filepath and a data array has been provided. The CSV is
                     given higher priority so the data will be overwritten by this
                     if possible""")
            self.csv_path = csv_path
            self.data = self.csvDataLoader(csv_path,**custom_columns)

        elif csv_path:

            self.csv_path = csv_path
            self.data = self.csvDataLoader(csv_path,**custom_columns)

        else:
            self.data      = data

        self.amino_acids     = amino_acids
        self.custom_columns  = custom_columns
        self.gen_graph       = gen_graph

        self.tokens      = {x:y for x,y in zip(self.amino_acids, list(range(len(self.amino_acids))))}

        self.sequences   = self.data[:,0]
        self.fitnesses   = self.data[:,1]

        if seed_seq:
            self.seed_seq      = seed_seq

        else:
            self.seed_id        = seed_id
            self.seed_seq       = self.sequences[self.seed_id]

        seq_len        = len(self.seed_seq)

        self.tokenized = np.concatenate((self.tokenize_data(),self.fitnesses.reshape(-1,1)),axis=1)

        self.token_dict = {tuple(seq) : idx for idx,seq in enumerate(self.tokenized[:,:-1])}

        self.mutated_positions = self.calc_mutated_positions()
        self.sequence_mutation_locations = self.boolean_mutant_array()
        # Stratifies data into different hamming distances

        # Contains the information to provide all mutants 1 amino acid away for a given sequence
        self.mutation_arrays  = self.gen_mutation_arrays()

        self.d_data = self.gen_d_data()

        # FIX THE CODE BELOW

        if self.gen_graph is False:
            self.graph = None

            self.num_minima,self.num_maxima = "Not Done", "Not Done"
            self.extrema_ruggedness = "Not Done"
            self.linear_slope, self.linear_RMSE, self.RS_ruggedness = "Not Done", "Not Done", "Not Done"

        else:
            self.graph = self.build_graph()

            self.num_minima,self.num_maxima = self.calculate_num_extrema()
            self.extrema_ruggedness = self.calc_extrema_ruggedness()
            self.linear_slope, self.linear_RMSE, self.RS_ruggedness = self.rs_ruggedness()

        print(self)

    def seed(self):
        return self.seed_seq

    def __str__(self):
        # TODO Change print formatting for seed sequence so it doesn't look bad
        return """
        Protein Landscape class
            Number of Sequences : {0}
            Max Distance        : {1}
            Number of Distances : {2}
            Seed Sequence       : {3}
                Modified positions are shown in green
            Number of minima : {4}
            Number of maxima : {5}
            Normalized Extrema Ruggedness : {6}
            R/S Ruggedness : {7}
        """.format(len(self),
                   self.max_distance,
                   len(self.d_data),
                   self.coloured_seed_string(),
                   self.num_minima,
                   self.num_maxima,
                   self.extrema_ruggedness,
                   self.RS_ruggedness)

    def __repr__(self):
        # TODO Finish this
        return f"""Protein_Landscape(seed_seq='{self.seed_seq}',
                                  gen_graph={self.gen_graph},
                                  csv_path='{self.csv_path}',
                                  custom_columns={self.custom_columns},
                                  amino_acids='{self.amino_acids}'
                                  )"""

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self,idx):
        return self.data[self.query(idx,information=False)]

    def query(self,sequence,information=True) -> int:
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
        if type(sequence) == str:
            assert sequence in self.sequences,"This sequence is not in the dataset"
            idx = int(np.where(self.sequences == sequence)[0])

        elif type(sequence) == tuple:
            assert len(sequence) == len(self.seed()), "Tuple not valid length for dataset"
            check = np.where(np.all(sequence == self.tokenized[:,:-1],axis=1))[0]
            assert len(check) > 0, "Not a valid tuple representation of a protein in this dataset"
            idx = int(check)

        elif type(sequence) == int:
            assert sequence in range(len(self)), "Index exceeds bounds of dataset"
            idx = sequence

        else:
            raise ValueError("Input format not understood")

        if information:
            assert self.graph is not None, "To provide this information, the graph must be computed"
            return self.graph[idx]

        else:
            return idx

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
            seq = self.seed()

        else:
            seq = self.sequences[self.query(seq,information=False)]

        subsets = {x : [] for x in range(len(seq)+1)}

        self.hammings = self.hamming_array(seq=seq)

        for distance in range(len(seq)+1):
            subsets[distance] = np.equal(distance,self.hammings)
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
        index_array = np.ones((len(self.seed_seq)),dtype=np.int8)
        index_array[positions] = 0
        mutated_indexes = np.all(np.invert(self.sequence_mutation_locations[:,constants]),axis=1)
        # This line checks only the positions that have to be constant, and ensures that they all are

        return mutated_indexes

    @staticmethod
    def hamming(str1, str2):
        """Calculates the Hamming distance between 2 strings"""
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def hamming_array(self,seq=None,data=None):
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
        seq : np.array[int], default=None

            Sequence which will be compared to the entire dataset.
        """
        if seq is None:
            tokenized_seq = np.array(self.tokenize(self.seed_seq))
        else:
            tokenized_seq = np.array(self.tokenize(seq))

        if data is None:
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

    def tokenize(self,seq):
        """
        Simple static method which tokenizes an individual sequence
        """
        return [self.tokens[aa] for aa in seq]

    def boolean_mutant_array(self):
        return np.invert(self.tokenized[:,:-1] == self.tokenize(self.seed_seq))

    def calc_mutated_positions(self):
        """
        Determines all positions that were modified experimentally and returns the indices
        of these modifications.

        Because the Numpy code is tricky to read, here is a quick breakdown:

            self.tokenized is called, and the fitness column is removed by [:,:-1]
            Each column is then tested against the first
        """
        mutated_bools = np.invert(np.all(self.tokenized[:,:-1] == self.tokenize(self.seed_seq),axis=0)) # Calculates the indices all of arrays which are modified.
        mutated_idxs  = mutated_bools * np.arange(1,len(self.seed()) + 1) # Shifts to the right so that zero can be counted as an idx
        return mutated_idxs[mutated_idxs != 0] - 1 # Shifts it back

    def coloured_seed_string(self):
        """
        Printing function that prints the original seed string and colours the positions that
        have been modified
        """
        strs = []
        idxs = self.mutated_positions
        for i,char in enumerate(self.seed_seq):
            if i in idxs:
                strs.append(f"{Fore.GREEN}{char}{Style.RESET_ALL}")
            else:
                strs.append("{0}".format(char))
        return "".join(strs)


    def gen_mutation_arrays(self):
        leng = len(self.seed())
        xs = np.arange(leng*len(self.amino_acids))
        ys = np.array([[y for x in range(len(self.amino_acids))] for y in range(leng)]).flatten()
        modifiers = np.array([np.arange(len(self.amino_acids)) for x in range(leng)]).flatten()
        return (xs, ys, modifiers)

    def lengthen_sequences(self, seq_len, AAs=None, mut_indices=False):
        """
        Vectorized function that takes a seq len and randomly inserts the tokenized
        integers into this new sequence framework with a fixed length.

        Parameters
        ----------
        seq_len : int

            Interger that determines how long the new sequences will be.

        AAs : str, default=None

            The string of all possible amino acids that should be considered.
            If no value is provided, it defaults to the landscapes amino acid
            list

        mut_indices : Bool

            If you want to specify the indices at which the old sequences will
            be embedded.
        """

        if not AAs:
            AAs = self.amino_acids

        sequences = self.tokenized[:,:-1]

        new = np.random.choice(len(AAs),seq_len)

        lengthened_data = np.zeros((sequences.shape[0],seq_len),dtype=np.int8)

        for i,val in enumerate(new):
            lengthened_data[:,i] = val

        if mut_indices.__class__.__name__ == "ndarray":
            assert len(mut_indices) == sequences.shape[1], "The index array must have the same length as the original sequence"
            idxs = mut_indices

        else:
            idxs = np.random.choice(seq_len,sequences.shape[1],replace=False)

        for i, idx in enumerate(idxs):
            lengthened_data[:,idx] = sequences[:,i]

        fitnesses = self.tokenized[:,-1]

        return np.concatenate((lengthened_data,fitnesses.reshape(-1,1)), axis=1 )

    def return_lengthened_data(self,seq_len, AAs=None, mut_indices=False,split=0.8,shuffle=True):
        """
        Helper function that passes the result of lengthen sequences to sklearn_data.
        Argument signature is a combination of self.lengthen_sequences and self.sklearn_data
        """
        return self.sklearn_data(data=(self.lengthen_sequences(seq_len,AAs,mut_indices)),split=split,shuffle=shuffle)


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

    # Ruggedness Section
    def is_extrema(self,idx,graph=None):
        """
        Takes the ID of a sequence and determines whether or not it is a maxima given its neighbours

        Parameters:
        -----------
        idx : int

            Integer index of the sequence that will be checked as a maxima.
        """
        if graph is None:
            graph = self.graph

        neighbours = graph[idx]["neighbours"]
        #print(self.fitnesses[neighbours])
        max_comparisons = np.greater(self.fitnesses[idx],self.fitnesses[neighbours])
        min_comparisons = np.less(self.fitnesses[idx],self.fitnesses[neighbours])
        if np.all(max_comparisons):
            return 1
        elif np.all(min_comparisons): # Checks to see if the point is a minima
            return -1
        else:
            return 0

    def generate_mutations(self,seq):
        """
        Takes a sequence and generates all possible mutants 1 Hamming distance away
        using array substitution

        Parameters:
        -----------
        seq : np.array[int]

            Tokenized sequence array
        """
        seed = self.seed()
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

    def calc_neighbours(self,idx,token_dict=None):
        """
        Takes a sequence and checks all possible neighbours against the ones that are actually present within the dataset.

        There is a particular design decision here that makes it fast in some cases, and slow in others.
        Instead of checking the entire dataset to see if there are any neighbours present, it calculates all
        possible neighbours, and checks to see if any of them match entries in the dataset.

        The number of possible neighbours for a given sequence is far smaller than the size of the dataset in almost
        all cases, which makes this approach significantly faster.

        Parameters:
        -----------

        seq : np.array[int]

            Tokenized sequence array
        """
        if token_dict is None:
            token_dict = self.token_dict

        possible_neighbours = self.generate_mutations(self.tokenized[idx,:-1])
        actual_neighbours = [token_dict[tuple(key)] for key in possible_neighbours if tuple(key) in token_dict]
        return idx, actual_neighbours

    # Graph Section
    def build_graph(self,idxs=None):
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

        else: # Check to see if this still works
            print("Building Protein Graph For subset of length {}".format(sum(idxs)))
            dataset = self.tokenized[:,:-1][idxs]
            integer_indexes = np.where(idxs)[0]
            token_dict = {key : value for key,value in self.token_dict.items() if value in integer_indexes}
            if len(integer_indexes) < 100000:
                pool = mp.Pool(4)
            else:
                pool = mp.Pool(mp.cpu_count())

        mapfunc = partial(self.calc_neighbours,token_dict=token_dict)
        results = pool.map(mapfunc,tqdm.tqdm(range(len(self))))
        neighbours = {idx :        {"tokenized"   : tuple(self.tokenized[idx,:-1]),
                                    "string"      : self.sequences[idx],
                                    "fitness"       : self.fitnesses[idx],
                                    "neighbours"  : neighbours} for idx, neighbours in results}
        return neighbours

    def graph_to_networkx(self):
        """
        Produces a networkx graph from the internally stored graph object
        """
        # So the problem here is that I do not know how to stick labels on with the
        # nodes. As a result, cytoscape won't be able to visualize the graphs properly.
        g = nx.Graph()
        g.add_nodes_from(self.sequences)#,attr_dict={"fitness" : self.fitnesses[idx]})
        for node in tqdm.tqdm(self.sequences):
            # I could just shift this to an enumerate operation, I just want to be absolutely
            # positive that there is no strange changes to the ordering.
            idx = int(np.where(self.sequences == node)[0])
            g.add_edges_from([(node, self.sequences[neighbour_idx]) for neighbour_idx in self.graph[idx]["neighbours"]])
        self.networkx_graph = g

    ############################################################################
    ######################## Ruggedness Estimation #############################
    ############################################################################

    def calculate_num_extrema(self,idxs=None):
        """
        Calcaultes the number of maxima across a given dataset or array of indices
        """
        if idxs is None:
            idxs = range(len(self))
            graph = self.graph
        else:
            graph = self.build_graph(idxs=idxs)
            idxs = np.where(idxs)[0]

        print("Calculating the number of extrema")
        mapfunc = partial(self.is_extrema, graph=graph)
        results = np.array(list(map(mapfunc,tqdm.tqdm(idxs))))
        minima = -1*np.sum(results[results<0])
        maxima = np.sum(results[results>0])
        return minima, maxima

    def calc_extrema_ruggedness(self):
        """
        Simple function that returns a normalized ruggedness value
        """
        ruggedness = (self.num_minima+self.num_maxima)/len(self)
        return ruggedness

    def extrema_ruggedness_subset(self,idxs):
        """
        Function that calculates the extrema ruggedness based on a subset of the
        full protein graph
        """
        minima, maxima = self.calculate_num_extrema(idxs=idxs)
        return (minima+maxima)/sum(idxs)

    def indexing(self,distances=None,percentage=None,positions=None):
        """
        Function that handles more complex indexing operations, for example wanting
        to combine multiple distance indexes or asking for a random set of indices of a given
        length relative to the overall dataset
        """
        if distances is not None:
            # Uses reduce from functools package and the bitwise or operation
            # to recursively combine the indexing arrays, returning a final array
            # where all Trues are collated into one array
            return reduce(np.logical_or, [self.d_data[d] for d in distances])

        if percentage is not None:
            assert 0 <= percentage <= 1, "Percentage must be between 0 and 1"
            idxs = np.zeros((len(self)))
            for idx in np.random.choice(np.arange(len(self)),size=int(len(self)*percentage),replace=False):
                idxs[idx] = 1
            return idxs.astype(np.bool)

        if positions is not None:
            # This code uses bitwise operations to maximize speed.
            # It first uses an or gate to collect every one where the desired position was modified
            # It then goes through each position that shouldn't be changed, and uses three logic gates
            # to switch ones where they're both on to off, returning the indexes of strings where ONLY
            # the desired positions are changed
            not_positions = [x for x in range(len(self.seed_seq)) if x not in positions]
            working = reduce(np.logical_or,[self.sequence_mutation_locations[:,pos] for pos in positions])
            for pos in not_positions:
                temp = np.logical_xor(working,self.sequence_mutation_locations[:,pos])
                working = np.logical_and(temp,np.logical_not(self.sequence_mutation_locations[:,pos]))
            return working


    def rs_ruggedness(self, log_transform=False, distance=None, split=1.0):
        """
        Returns the rs based ruggedness estimate for the landscape.

        Parameters
        ----------
        log_transform : bool, default=False

            Boolean value that determines if the base 10 log transform will be applied.
            The application of this was suggested in the work by Szengdo

        distance : int, default=None

            Determines the distance for data that will be sampled

        split : float, default=1.0, range [0-1]

            How much of the data is used to determine ruggedness
        """
        if distance:
            x_train, y_train, _, _ = self.sklearn_data(split=split,distance=distance)
        else:
            x_train, y_train, _, _ = self.sklearn_data(split=split)
        if log_transform:
            y_train = np.log10(y_train)

        lin_model = LinearRegression(n_jobs=mp.cpu_count).fit(x_train,y_train)
        y_preds = lin_model.predict(x_train)
        coefs   = lin_model.coef_
        rmse_predictions = np.sqrt(mean_squared_error(y_train, y_preds))
        slope = (1/len(self.seed_seq)*sum([abs(i) for i in coefs]))
        return [slope, rmse_predictions, rmse_predictions/slope]

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
            return copy.copy(self.data)
        else:
            return copy.copy(self.tokenized)


    def tokenize_data(self):
        """
        Takes an iterable of sequences provided as one amino acid strings and returns
        an array of their tokenized form.

        Note : The tokenize function is not called and the tokens value is regenerated
        as it removes a lot of function calls and speeds up the operation significantly.
        """
        tokens = self.tokens
        return np.array([[tokens[aa] for aa in seq] for seq in self.sequences])

    def sklearn_data(self, data=None,distance=None,positions=None,split=0.8,shuffle=True):
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
        elif distance:
            if type(distance) == int:
                data = copy.copy(self.tokenized[self.get_distance(distance)])
            else:
                data = collapse_concat([copy.copy(self.tokenized[self.get_distance(d)]) for d in distance])

        elif positions is not None:
            data = copy.copy(self.tokenized[self.get_mutated_positions(positions)])
        else:
            data = copy.copy(self.tokenized)

        if shuffle:
            np.random.shuffle(data)

        split_point = int(len(data)*split)

        train = data[:split_point]
        test  = data[split_point:]

        # Y data selects only the last column of Data
        # X selects the rest

        x_train = train[:,:-1]
        y_train = train[:,-1]
        x_test  = test[:,:-1]
        y_test  = test[:,-1]

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
            data = stored_data[idxs]

        elif distance:
            if type(distance) == int:
                data = stored_data[self.get_distance(distance)]
            else:
                data = collapse_concat([stored_data[self.get_distance(d)] for d in distance])

        elif positions is not None:
            data = stored_data[self.get_mutated_positions(positions)]

        else:
            data = stored_data

        if unsupervised:
            labels = {torch.Tensor(x[:-1].astype('int8')).long() : real_label for x in data}
        else:
            labels = {torch.Tensor(x[:-1].astype('int8')).long() : x[-1] for x in data}

        keys   = list(labels.keys())

        return self.gen_dataloaders(labels=labels, keys=keys, params=params, split_point=split_point)

    def shotgun_data(self,num_samples=1000):
        """
        Shotgun data randomly samples the entirety of the dataset

        Parameters
        ----------
        num_samples : int, default=1000

            Determines how many datapoints will be sampled
        """
        idxs = np.random.choice(len(self), size=(num_samples,))
        return idxs

    @staticmethod
    def mc_criterion(state1, state2, T):
        preference_for_state2 = 1 - (1 / (1 + np.exp((state2 - state1) * 1/T)))
        return np.array([1-preference_for_state2, preference_for_state2]).reshape(-1,)

    def evolved_trajectory_data(self,initial_seq=None,
                                     num_steps=1000,
                                     scaled = True,
                                     mc_criterion=None,
                                     T=0.01,
                                     remove_duplicates=False):
        """
        Evolves a trajectory from an initial seed sequence using a Monte Carlo Criterion
        that can be provided as a generic python function.

        IMPORTANT : It is allowed to sample the same position multiple times,
        with duplicate entries being removed before the indices are returned.

        Parameters
        ----------
        initial_seq : str, default=None

            The initial sequence from which the random walk will begin. If none,
            will use the seed sequence of the dataset.

        num_steps : int, default=1000

            The number of steps that the algorithm is allowed to take through the
            graph.

        scaled : Bool, default=True

            Whether or not the data should be scaled to be within a range of 0 to 1
            to remove scale based artefacts of the Monte Carlo Criterion

        mc_criterion : func, default=None

            A Python function to calculate the probability of choosing a new state (state2)
            over the previous state (state1). Must use the same signature as the
            mc_criterion static method above

            def mc_criterion(state1 : float32, state2 : float32, T : float32) -> np.array[p(state1),p(state2)]

            The state inputs are the fitness labels assocaiated with the states

        T : float, default=0.01

            A sensitivity (or "temperature") parameter, with lower values increasing the
            preference for lower energy states, i.e cooling produces a more ordered system.

            Higher temperatures make this procedure analogous to a random walk.

        remove_duplicates : Bool, default=False

            Whether or not to remove duplicate indexes from the list. By default the answer is no.

        Returns
        -------

        idxs : np.array(num_steps,)

            A numpy array of all indices sampled during the walk.
        """
        if scaled:
            scaler = MinMaxScaler()
            scaler.fit(self.data[:,-1].reshape(-1,1))
            labels = scaler.transform(self.data[:,-1].reshape(-1,1))
        else:
            labels = self.data[:,-1].reshape(-1,1)

        if initial_seq is not None:
            initial_seq = self.query(initial_seq,information=False)
        else:
            initial_seq = self.seed_id

        idxs = [initial_seq]
        state = initial_seq
        if mc_criterion is None:
            mc_criterion = self.mc_criterion

        for i in range(num_steps):
            idx = random.choice(self.graph[state]["neighbours"]) # Choose random possible neighbour
            state = np.random.choice([state, idx], p = mc_criterion(labels[state],labels[idx],T))
            idxs.append(state)
        return idxs


    def random_walk_data(self,T=10000,
                              **kwargs):
        """
        Helper method that generates a random walk by performing a Monte Carlo
        walk with a very high temperature. See self.evolved_trajectory_data for
        complete call signature.
        """
        return self.evolved_trajectory_data(T=T,**kwargs)

    def deep_sequence_data(self,initial_seq=None,max_distance=1):
        """
        Uses the gen_d_data function to provide all sequences a certain hamming
        distance away

        Parameters
        ----------
        initial_seq : str, default=None

            The initial sequence from which the random walk will begin. If none,
            will use the seed sequence of the dataset.

        max_distance : int, default=1

            The maximum hamming distance from the central sequence that the data
            will be extracted from.
        """
        if initial_seq is None:
            initial_seq = tuple(self.tokenize(self.seed_seq))
            d_data      = self.d_data
        else:
            seq         = self.query(initial_seq,information=False)
            d_data      = self.gen_d_data(seq=seq)

        idxs = reduce(np.add,[d_data[d] for d in range(1,max_distance+1)])
        return idxs

if __name__ == "__main__":
    test = Protein_Landscape(csv_path="../Data/NK/K4/V1.csv")
