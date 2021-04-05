# Molecular Spaces

Package that interacts with structured molecular datasets and provides a large
range of functionalities to integrate it with machine learning workflows.

## Features

1. Complex indexing operations that allow the user to isolate mutational regimes, select a portion of mutated positions or any combination.
2. Sample a given dataset using a variety of regimes commonly seen in protein engineering including deep sequence, shotgun, random walk, and evolved trajectory.
3. The software will use multiprocessing to calculate the graph of the protein dataset, which is then leveraged by a wide
variety of other methods to accelerate computation.
4. The capacity to return data (indexed or not) in formats that are pre-prepared for a variety of machine learning tasks including Scikit-learn, Pytorch, and (To do) Tensorflow

## How it works

The code relies on the interaction of two abstract objects: molecular.py and space.py. These abstract classes provide the user with an API to customize the behaviour of the interactions, resulting in different graphs that can be indexed and traversed in a variety of ways.

The code has two fundamental subcomponents:
1.   The ability to slice the data in a wide variety of ways.
2.   The ability to return the data to you in the form of useful objects.

The package strives to achieve high speeds wherever possible by leveraging numpy's abilities.

### Todo

1. Completely restructuring code to provide abstract base classes that can be readily modified for different examples.
   1. Currently producing an abstract landscape class that has the minimal level of functionality to generate a graph from a series of molecular representations that are provided.
   2. Biggest challenge at this point is rebuilding the build_graph function
   3. Also need to add generalized indexing functions that allow people to meaningfully extract data for different molecular types.
   4. Need to completely rebuild indexing for abstract base class
   5. Need to decide if I want machine learning functionality to be built into base class
2. Add support for the generation of tensorflow data loaders (https://www.tensorflow.org/tutorials/load_data/csv)
4. Need to figure out how to build an extensible tokenization system so that it can integrate with methods like TAPE.
5. Add feature to write out its own graphml object
6. So when I index it. It makes most sense to return another protein_landscape class, instead of just a dictionary. Then all of then custom functinoality comes with it. Two ways to do this - either use a view of the original graph dictory, or literally instantiate a new class.
7. Need to add feature that allows you to specify indices for train, validation, and test for sklearn and pytorch

The system needs to be rebuilt to maximize generalizability, making it work with a wide variety of different systems.
