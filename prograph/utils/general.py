"""
Some general python utility functions.
"""
import numpy as np

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Uses np.allclose to determine whether or not an array is symmetric by comparing
    it to its transpose.
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def esm_sequence_embeddings(data,model,alphabet,layer):
    """
    ESM sequence embedding function. Isolates the sequence representations.

    Parameters
    ----------
    data : [(name, sequence)] or [sequence]
        The data to be processed. Can either be passed with sequence names as in fasta files,
        or can simply be passed the sequences, in which case it will assign each one a numeric
        index.

    model : esm.model.ProteinBertModel
        The esm protein transformer model to use

    alphabet : esm.data.Alphabet
        The protein alphabet to use.

    layer : int
        The layer to extract the representation out of.

    return
    """
    print(f"Will extract information from layer {layer}")
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    print("Ensuring data is of correct type")
    if type(data[0]) != tuple:
        data = [(f"seq_{idx}",seq) for idx, seq in enumerate(data)]

    print("Tokenizing sequences")
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    print("Passing sequences through transformer")
    results = []
    with torch.no_grad():
        for seq,token in tqdm.tqdm(zip(batch_strs,batch_tokens)):
            tensor = model(token.reshape(1,-1).cuda(), repr_layers=[layer])["representations"][layer].cpu()
            results.append(np.array(tensor[0,1 : len(seq) + 1].mean(0))) # Take the mean of the sequences to lower stress on memory.

    return results

def flatten(lst):
    """
    Flattens a list of lists using a list comprehension.
    """
    return [item for sublist in lst for item in sublist]
