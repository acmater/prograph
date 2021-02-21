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

The package strives to achieve high speeds wherever possible by leveraging numpy's abilities.

### Todo

1. Work out how to handle the seed sequence. Should it be removed or not?
    No, it shouldn't be removed, however I need to deal with the fact that it often isn't explicitly included with a fitness value.
2. Add support for the generation of tensorflow data loaders (https://www.tensorflow.org/tutorials/load_data/csv)
3. Should I consider updating the array operations to use the cupy package?
4. Add scaler feature to Sklearn data and by extension, fit.
5. Need to figure out how to build an extensible tokenization system so that it can integrate with methods like TAPE.
6. I want to update the __getitem__ method so that you can extra only sequences, or only fitnesses, if that is what you want
7. Basically I need to update self.graph so that it uses a custom dictionary that emulates the inbuilt dictionaries, but with a custom __getitem__ syntax which means that if you go self.graph['seq'] it will return all of the sequences. This is the only way to get the behaviour I want. I think.
8. Add feature to write out its own graphml object

The system needs to be rebuilt to maximize generalizability, making it work with a wide variety of different systems.
