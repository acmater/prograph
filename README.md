# Protein Landscape

Package that interacts with structured protein datasets and provides a large
range of functionalities to integrate it with machine learning workflows.

## Features

1. Complex indexing operations that allow the user to isolate mutational regimes, select a portion of mutated positions or any combination.
2. Sample a given dataset using a variety of regimes commonly seen in protein engineering including deep sequence, shotgun, random walk, and evolved trajectory.
3. The software will use multiprocessing to calculate the graph of the protein dataset, which is then leveraged by a wide
variety of other methods to accelerate computation.
4. The capacity to return data (indexed or not) in formats that are pre-prepared for a variety of machine learning tasks including Scikit-learn, Pytorch, and (To do) Tensorflow

## How it works

The code has two fundamental subcomponents:
1.   The ability to slice the data in a wide variety of ways.
2.   The ability to return the data to you in the form of useful objects.

Modifying the code should work to maintain this syntax. In general, what is passed between
functions are indexes stored as numpy arrays. They are stored on one of two formats,
the first is as an array of integer indexes, the second is as a Boolean index array.

The former is more memory efficient as the index arrays can become highly sparse data. However
the latter has advantages when calculating the union.

I am fairly confident that I can standardize it to just the integer index arrays by employing
the np.union1d operation when merging different index operations.

The package strives to achieve high speeds wherever possible by leveraging numpy's abilities.

### Todo

1. Work out how to handle the seed sequence. Should it be removed or not?
2. Add functionality to export protein landscape graph to cytoscape
3. Need to make all data acquisition schemes utilize the same syntax which allows you to manually specify a data array
4. Add support for the generation of tensorflow data loaders
