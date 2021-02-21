class Protein():
    def __init__(self, seq,
                       fitness=None,
                       tokenized=None,
                       neighbours=None):
        self.seq        = seq
        self.fitness    = fitness
        self.tokenized  = tokenized
        self.neighbours = neighbours



if __name__ == "__main__":
    a = Protein("AAC")
    b = Protein("ACA")
