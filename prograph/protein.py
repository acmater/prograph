import numpy as np

class Protein():
    """
    Python class which handles instances of individual proteins. Protein is
    initialized with a variety of properties and utilises dictionary type syntaxing
    to make it compliant with the remainder of the code.

    Parameters
    ----------
    seq : str

        The only required argument. Is used to calculate length and to determine if
        two proteins are equivalent.
    """
    def __init__(self, seq,**kwargs):
        self.seq = seq
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        parsed = []
        for key, value in vars(self).items():
            if isinstance(value,np.ndarray):
                parsed.append(f"{key}=np.array({list(value)})")
            elif isinstance(value, str):
                parsed.append(f"{key}='{value}'")
            else:
                parsed.append(f"{key}={value}")
        return f"Protein(" + ",".join(parsed) + ")"

    def __getitem__(self,keys):
        if isinstance(keys, list):
            # customized behaviour to allow user to specify tuple of keys
            return tuple([self.__dict__[x] for x in keys])
        return self.__dict__[keys]

    def __len__(self):
        return len(self.seq)

    def __eq__(self,other):
        return True if self.seq == other.seq else False

if __name__ == "__main__":
    a = Protein("AAC",fitness=0.3)
    print(len(a))
    print(a.seq)
    b = Protein("ACA")
    c = Protein("AAC")
    print(a==b)
    print(a==c)
    print(a["fitness"])
    print(a.__repr__())


"""def __eq__(self,prot):
    for attr in vars(self).keys():
        try:
            comp = self[attr] == prot[attr]
            if hasattr(comp, '__iter__'):
                if all(comp) is not True:
                    return False
            else:
                if comp is not True:
                    return False
        except:
            return False
    return True
    """
