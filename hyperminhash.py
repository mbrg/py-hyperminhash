from typing import List, Tuple, Union

import numpy as np

import metrohash

import logging


p = 14
m = np.uint32(1 << p)  # 16384
max = 64 - p
maxX = np.iinfo(np.uint64).max >> max
alpha = 0.7213 / (1 + 1.079 / np.float64(m))
q = 6  # the number of bits for the LogLog hash
r = 10  # number of bits for the bbit hash
_2q = 1 << q
_2r = 1 << r
c = 0.169919487159739093975315012348


def metro_hash_128(val: bytes, seed: int):
    h: bytes = metrohash.metrohash128(val, seed)

    h1 = int.from_bytes(h, byteorder="little", signed=False)
    h2 = int.from_bytes(h[8:], byteorder="little", signed=False)

    h1 = np.uint64(h1 % np.iinfo(np.uint64).max)
    h2 = np.uint64(h2 % np.iinfo(np.uint64).max)

    return h1, h2


def beta(ez: np.float64) -> np.float64:
    zl = np.log(ez + 1)
    return -0.370393911 * ez + \
        0.070471823 * zl + \
        0.17393686 * np.power(zl, 2) + \
        0.16339839 * np.power(zl, 3) + \
        -0.09237745 * np.power(zl, 4) + \
        0.03738027 * np.power(zl, 5) + \
        -0.005384159 * np.power(zl, 6) + \
        0.00042419 * np.power(zl, 7)


class Register:
    def __init__(self, val: int = 0, *args, **kwargs):
        logging.debug(f"New Register({val}).")
        self.val = np.uint16(val, *args, **kwargs)

    @classmethod
    def from_tuple(cls, lz: np.uint8 = 0, sig: np.uint16 = 0, *args, **kwargs):
        val = (np.uint16(lz) << r) | sig
        return Register(val, *args, **kwargs)

    def lz(self) -> np.uint8:
        return np.uint8(np.uint16(self.val) >> (16 - q))

    def __str__(self):
        return self.val.__str__()

    def __repr__(self):
        return self.val.__repr__()

    def __eq__(self, other):
        return isinstance(other, Register) and self.val.__eq__(other.val)

    def __ge__(self, other):
        return isinstance(other, Register) and self.val.__ge__(other.val)

    def __le__(self, other):
        return isinstance(other, Register) and self.val.__le__(other.val)

    def __gt__(self, other):
        return isinstance(other, Register) and self.val.__gt__(other.val)

    def __lt__(self, other):
        return isinstance(other, Register) and self.val.__lt__(other.val)

    def __neg__(self, other):
        return isinstance(other, Register) and self.val.__neg__(other.val)

    def __ne__(self, other):
        return isinstance(other, Register) and self.val.__ne__(other.val)

    def __rmul__(self, other):
        return isinstance(other, Register) and self.val.__rmul__(other.val)

    def __mul__(self, other):
        return isinstance(other, Register) and self.val.__mul__(other.val)

    def __add__(self, other):
        return isinstance(other, Register) and self.val.__add__(other.val)

    def __bool__(self, other):
        return isinstance(other, Register) and self.val.__bool__(other.val)

    def __sub__(self, other):
        return isinstance(other, Register) and self.val.__sub__(other.val)


def leading_zeros64(x: np.uint64, num_bits: int = 64) -> int:
    """
    LeadingZeros64 returns the number of leading zero bits in x; the result is 64 for x == 0.
    """
    res = 0
    while (x & (np.uint64(1) << (np.uint64(num_bits) - np.uint64(1)))) == 0:
        x = (x << np.uint64(1))
        res += 1

    return res


def reg_sum_and_zeros(registers: List[Register]) -> Tuple[np.float64, np.float64]:
    sm: np.float64 = np.float64(0)
    ez: np.float64 = np.float64(0)

    for val in registers:
        lz = val.lz()
        if lz == 0:
            ez += 1
        sm += 1 / np.power(2, np.float64(lz))

    return sm, ez


class HyperMinHash:
    """
    HyperMinHash is a sketch for cardinality estimation based on LogLog counting
    """
    def __init__(self, ln: int = m):
        logging.debug(f"New HyperMinHash({ln}).")
        self.reg = [Register() for _ in range(ln)]

    def add_hash(self, x: np.uint64, y: np.uint64) -> None:
        """
        AddHash takes in a "hashed" value (bring your own hashing)
        """
        k = x >> np.uint32(max)
        lz = np.uint8(leading_zeros64((x << np.uint64(p)) ^ np.uint64(maxX))) + 1
        sig = y << (np.uint64(64) - np.uint64(r)) >> (np.uint64(64) - np.uint64(r))
        sig = np.uint16(sig % np.iinfo(np.uint16).max)
        reg = Register.from_tuple(lz, sig)
        if self.reg[k] is None or self.reg[k] < reg:
            self.reg[k] = reg

    def add(self, value: Union[bytes, str, int]) -> None:
        """
        Add inserts a value into the sketch
        """
        if isinstance(value, int):
            value = str(value)
        if isinstance(value, str):
            value = str.encode(value)

        logging.debug(f"HyperMinHash.add({value}).")

        h1, h2 = metro_hash_128(value, 1337)
        self.add_hash(h1, h2)

    def cardinality(self) -> np.uint64:
        """
        Cardinality returns the number of unique elements added to the sketch
        """
        sm, ez = reg_sum_and_zeros(self.reg)
        res = np.uint64(alpha * np.float64(m) * (np.float64(m) - ez) / (beta(ez) + sm))
        logging.debug(f"HyperMinHash.cardinality sm={sm}, ez={ez}, res={res}.")
        return res

    def __len__(self):
        return int(self.cardinality())

    def merge(self, other: "HyperMinHash") -> "HyperMinHash":
        """
        Merge returns a new union sketch of both sk and other
        """
        if len(self.reg) != len(other.reg):
            raise ValueError(f"self / other have different lengths: {len(self.reg)} / {len(other.reg)}.")

        for i in range(len(self.reg)):
            if self.reg[i] < other.reg[i]:
                self.reg[i] = other.reg[i]

        return self

    def similarity(self, other: "HyperMinHash") -> np.float64:
        """
        Similarity return a Jaccard Index similarity estimation
        """
        c = np.float64(0)
        n = np.float64(0)

        for i in range(len(self.reg)):
            if self.reg[i] != 0 and self.reg[i] == other.reg[i]:
                c += 1
            if self.reg[i] != 0 or other.reg[i] != 0:
                n += 1
        if c == 0:
            return np.float64(0)

        crd_slf = np.float64(self.cardinality())
        crd_otr = np.float64(other.cardinality())
        ec = self.approximate_expected_collisions(crd_slf, crd_otr)

        # FIXME: must be a better way to predetect this
        if c < ec:
            return np.float64(0)

        res = np.float64((c - ec) / n)
        logging.debug(f"HyperMinHash.similarity c={c}, n={n}, crd_slf={crd_slf}, crd_otr={crd_otr}, ec={ec}, res={res}.")

        return res

    def approximate_expected_collisions(self, n: np.float(64), m: np.float(64)) -> np.float(64):
        if n < m:
            n, m = m, n

        if n > np.power(2, np.power(2, q) + r):
            return np.iinfo(np.uint64).max
        elif n > np.power(2, p + 5):
            d = (4 * n / m) / np.power((1 + n) / m, 2)
            return c * np.power(2, p - r) * d + 0.5
        else:
            return self.expected_collision(n, m) / np.float64(p)

    def expected_collision(self, n: np.float(64), m: np.float(64)) -> np.float(64):
        x = np.float64(0)

        for i in range(1, _2q + 1):
            for j in range(1, _2r + 1):
                if i != _2q:
                    den = np.power(2, p + r + i)
                    b1: np.float64 = (_2r + j) / den
                    b2: np.float64 = (_2r + j + 1) / den
                else:
                    den = np.power(2, p + r + i - 1)
                    b1: np.float64 = j / den
                    b2: np.float64 = (j + 1) / den

                prx = np.power(1 - b2, n) - np.power(1 - b1, n)
                pry = np.power(1 - b2, m) - np.power(1 - b1, m)
                x += (prx * pry)

        return (x * np.float64(p)) + 0.5

    def intersection(self, other: "HyperMinHash") -> np.uint64:
        """
        Intersection returns number of intersections between sk and other
        """
        sim = self.similarity(other)
        return np.uint64((sim * np.float64(self.merge(other).cardinality()) + 0.5))
