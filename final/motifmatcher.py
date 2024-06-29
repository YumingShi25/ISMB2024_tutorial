from collections import defaultdict
import argparse  # for command line interface
import re  # for regular expressions
import numpy as np  # for typed arrays
from numba import njit, uint64, uint8  # for just-in-time compilation


def make_toupper_table():
    T = np.arange(256, dtype=np.uint8)
    T[ord('a')] = ord('A')
    T[ord('c')] = ord('C')
    T[ord('g')] = ord('G')
    T[ord('t')] = ord('T')
    T[ord('u')] = ord('T')
    T[ord('n')] = ord('N')
    return T


@njit
def seq_to_upper(seq, T=make_toupper_table()):
    n = len(seq)
    for i in range(n):
        seq[i] = T[seq[i]]


def _fasta_reads_from_filelike(f, COMMENT=b';'[0], HEADER=b'>'[0]):
    """internal function that yields facta records as (header: bytes, seq: bytearray)"""
    strip = bytes.strip
    header = seq = None
    for line in f:
        line = strip(line)
        if len(line) == 0:
            continue
        if line[0] == COMMENT:
            continue
        if line[0] == HEADER:
            if header is not None:
                yield (header, seq)
            header = line[1:]
            seq = bytearray()
            continue
        seq.extend(line)
    if header is not None:
        yield (header, seq)


def fasta_items(filename):
    """
    generator function that yields each (header, sequence) pair from a FASTA file.
    The header is given as an immutable 'bytes' object;
    The sequence is given as a mutable numpy array of dtype uint8.
    """
    with open(filename, "rb") as f:
        for (header, seq) in _fasta_reads_from_filelike(f):
            seqb = np.frombuffer(seq, dtype=np.uint8)
            seq_to_upper(seqb)  # translate in-place
            yield (header, seqb)


def parse_spacer(spacer):
    """parse a string of the form N(minlen,maxlen) and return maxlen, optionals"""
    match = re.match(r"N\((\d+),(\d+)\)$", spacer)
    minlen = int(match.group(1))
    maxlen = int(match.group(2))
    optionals = maxlen - minlen
    return maxlen, optionals


# Define the IUPAC alphabet
_IUPAC = defaultdict(list,
    A=[ord('A')],
    C=[ord('C')],
    G=[ord('G')],
    T=[ord('T')],
    R=[ord('A'), ord('G')],
    Y=[ord('C'), ord('T')],
    S=[ord('C'), ord('G')],
    W=[ord('A'), ord('T')],
    K=[ord('G'), ord('T')],
    M=[ord('A'), ord('C')],
    B=[ord('C'), ord('G'), ord('T')],
    D=[ord('A'), ord('G'), ord('T')],
    H=[ord('A'), ord('C'), ord('T')],
    V=[ord('A'), ord('C'), ord('G')],
    N=[ord('A'), ord('C'), ord('G'), ord('T')],
    )


def build_nfa(motif, iupac=_IUPAC):
    """Build an NFA from a IUPAC motif with additonal N(low,high) elements"""
    if not motif or not isinstance(motif, str):
        raise ValueError(f"Error: empty or invalid {motif=}")
    motif = motif.upper()
    mlist = []
    masks = np.zeros(256, dtype=np.uint64)
    I = F = 0
    # First , find the N(*,*) elements, replace them by maximal runs of Ns;
    # and set the corresponding I, F bits.
    spacer_allowed = False
    parts = re.split(r"(N\(\d+,\d+\))", motif)
    for part in parts:
        if not part:
            continue
        if part.startswith("N("):
            if not spacer_allowed:
                raise ValueError(f"Error: spacer not allowed here: {part}")
            maxlen, optionals = parse_spacer(part)
            bit_I = len(mlist) - 1
            I |= (1 << bit_I)
            F |= (1 << (bit_I + optionals))
            mlist.extend(["N"] * maxlen)
            spacer_allowed = False
        else:
            mlist.extend(list(part))
            spacer_allowed = True
    if not spacer_allowed:
        raise ValueError(f"Error: spacer not allowed at end ({motif=})")
    mstring = "".join(mlist)  # maximal motif as string
    if len(mstring) > 64:
        raise ValueError(f"Error: maximal motif length is {len(mstring)} > 64: {mstring}")
    for bit, c in enumerate(mstring):
        value = (1 << bit)
        for a in iupac[c]:
            masks[a] |= np.uint64(value)
    accept = value
    return masks, I, F, accept


def find_matches_slow(mask, I, F, accept, sequence, *_):
    """yield each end position of a match of the NFA against the sequence"""
    A = 0
    for i, c in enumerate(sequence):
        A = ((A << 1) | 1) & int(mask[c])  # int vs. numpy.uint64
        A = A | ((F - (A & I)) & ~F)
        if A & accept:
            yield i  # makes this a generator function


@njit(locals=dict(k=uint64, A=uint64, i=uint64, c=uint8))
def find_matches_fast(mask, I, F, accept, sequence, results):
    """
    just-in-time compiled version of the NFA matcher 
    that writes end positions of matches in to an array 'results'
    and returns the number of matches.
    """
    k = 0
    N = results.size
    A = 0
    for i, c in enumerate(sequence):
        A = ((A << 1) | 1) & mask[c]
        A = A | ((F - (A & I)) & ~F)
        if A & accept:
            if k < N:
                results[k] = i
            k += 1
    return k


def main(args):
    if not args.slow:
        NRESULTS = args.maxresults
        results = np.zeros(NRESULTS, dtype=np.uint64)
    nfa = build_nfa(args.motif)
    for header, sequence in fasta_items(args.fasta):
        print("#", header.decode("ASCII"))
        if args.slow:
            for pos in find_matches_slow(*nfa, sequence):
                print(pos)
        else:
            nresults = find_matches_fast(*nfa, sequence, results)
            if nresults > NRESULTS:
                print(f"! Too many results, showing first {NRESULTS}")
                nresults = NRESULTS
            print(*list(results[:nresults]), sep="\n")


def get_argument_parser():
    p = argparse.ArgumentParser(description="DNA Motif Searcher")
    p.add_argument("--motif", "-m",
        default="RCTGTGYRN(17,23)CYTCTCTG",  # Nucl. Acid Res. 47(2): p. 707, 2019
        help="DNA motif (IUPAC) with optional N(min,max) elements")
    p.add_argument("--fasta", "-f", required=True,
        help="FASTA file of genome")
    p.add_argument("--maxresults", "-R", type=int, default=10_000_000,
        help="maximum number of output positions per chromosome [10 mio] when using the numba-compiles version")
    p.add_argument("--slow", action="store_true",
        help="use a slow pure Python implementation instead of the fast numba implementation")
    return p


if __name__ == "__main__":
    main(get_argument_parser().parse_args())
