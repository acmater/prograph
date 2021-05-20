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
    def __init__(self, Sequence,**kwargs):
        self.Sequence = Sequence
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
        return len(self.Sequence)

    def __eq__(self,other):
        return True if self.Sequence == other.Sequence else False
