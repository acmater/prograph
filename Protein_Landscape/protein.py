import numpy as np

class Protein():
    """
    Python class which handles instances of individual proteins. Protein is
    initialized with a variety of properties and utilises dictionary type syntaxing
    to make it compliant with the remainder of the code.
    """
    def __init__(self, seq,
                       fitness=None,
                       tokenized=None,
                       neighbours=None):
        self.seq        = seq
        self.fitness    = fitness
        self.tokenized  = tokenized
        self.neighbours = neighbours

    def __repr__(self):
        return f"""Protein(seq='{self.seq}',
        fitness={self.fitness},
        tokenized={self.tokenized},
        neighbours=np.array({list(self.neighbours)}))"""

    def __getitem__(self,keys):
        if isinstance(keys, list):
            # customized behaviour to allow user to specify tuple of keys
            return tuple([self.__dict__[x] for x in keys])
        return self.__dict__[keys]

if __name__ == "__main__":
    a = Protein("AAC")
    b = Protein("ACA")
