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
        return f"""Protein(seq  ='{self.seq}',
        fitness = {self.fitness},
        tokenized ='{self.tokenized}',
        neighbours ={self.neighbours}"""

    def __getitem__(self,idx):
        if isinstance(idx, list):
            return tuple([self.__dict__[x] for x in idx])
        return self.__dict__[idx]

if __name__ == "__main__":
    a = Protein("AAC")
    b = Protein("ACA")
