from typing import Tuple, Union

import numpy as np

import metrohash

import logging


class HyperMinHash:
    """
    HyperMinHash is a sketch for cardinality estimation based on LogLog counting
    """
    def __init__(self, p: int = 14, q: int = 6, r: int = 10, c: float = 0.169919487159739093975315012348):
        """
        :param p: number of bits for each register
        :param q: number of bits for the LogLog hash
        :param r: number of bits for the bbit hash
        """
        self.p = p
        self.q = q
        self.r = r
        self._c = c

        logging.debug(f"New HyperMinHash({self._m}).")
        self.reg = np.zeros(self._m, dtype=np.uint16)

    @property
    def _m(self):
        return np.uint32(1 << self.p)

    @property
    def _max(self):
        return 64 - self.p

    @property
    def _maxX(self):
        return np.iinfo(np.uint64).max >> self._max

    @property
    def _alpha(self):
        return 0.7213 / (1 + 1.079 / np.float64(self._m))

    @property
    def _2q(self):
        return 1 << self.q

    @property
    def _2r(self):
        return 1 << self.r

    @staticmethod
    def _leading_zeros64(x: np.uint64, num_bits: int = 64) -> int:
        """
        LeadingZeros64 returns the number of leading zero bits in x; the result is 64 for x == 0.
        """
        res = 0
        while (x & (np.uint64(1) << (np.uint64(num_bits) - np.uint64(1)))) == 0:
            x = (x << np.uint64(1))
            res += 1

        return res

    @staticmethod
    def _metro_hash_128(val: bytes, seed: int):
        h: bytes = metrohash.metrohash128(val, seed)

        h1 = int.from_bytes(h, byteorder="little", signed=False)
        h2 = int.from_bytes(h[8:], byteorder="little", signed=False)

        h1 = np.uint64(h1 % np.iinfo(np.uint64).max)
        h2 = np.uint64(h2 % np.iinfo(np.uint64).max)

        return h1, h2

    def from_tuple(self, lz: np.uint8, sig: np.uint16):
        val = (np.uint16(lz) << self.r) | sig
        return np.uint16(val)

    def lz(self, val: np.uint16) -> np.uint8:
        return np.uint8(val >> (16 - self.q))

    def _add_hash(self, x: np.uint64, y: np.uint64) -> None:
        """
        AddHash takes in a "hashed" value (bring your own hashing)
        """
        k = x >> np.uint32(self._max)
        lz = np.uint8(self._leading_zeros64((x << np.uint64(self.p)) ^ np.uint64(self._maxX))) + 1
        sig = y << (np.uint64(64) - np.uint64(self.r)) >> (np.uint64(64) - np.uint64(self.r))
        sig = np.uint16(sig % np.iinfo(np.uint16).max)
        reg = self.from_tuple(lz, sig)
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

        h1, h2 = self._metro_hash_128(value, 1337)
        self._add_hash(h1, h2)

    @staticmethod
    def _beta(ez: np.float64) -> np.float64:
        zl = np.log(ez + 1)
        val = np.polyval([0.00042419, -0.005384159, 0.03738027, -0.09237745, 0.16339839, 0.17393686, 0.070471823, -0.370393911 * ez], zl)
        return np.float64(val)

    def reg_sum_and_zeros(self, registers: np.ndarray) -> Tuple[np.float64, np.float64]:
        sm: np.float64 = np.float64(0)
        ez: np.float64 = np.float64(0)

        for val in registers:
            lz = self.lz(val)
            if lz == 0:
                ez += 1
            sm += 1 / np.power(2, np.float64(lz))

        return sm, ez

    def cardinality(self) -> np.uint64:
        """
        Cardinality returns the number of unique elements added to the sketch
        """
        sm, ez = self.reg_sum_and_zeros(self.reg)
        res = np.uint64(self._alpha * np.float64(self._m) * (np.float64(self._m) - ez) / (self._beta(ez) + sm))
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
        ec = self._approximate_expected_collisions(crd_slf, crd_otr)

        # FIXME: must be a better way to predetect this
        if c < ec:
            return np.float64(0)

        res = np.float64((c - ec) / n)
        logging.debug(f"HyperMinHash.similarity "
                      f"c={c}, n={n}, crd_slf={crd_slf}, crd_otr={crd_otr}, ec={ec}, res={res}.")

        return res

    def _approximate_expected_collisions(self, n: np.float(64), m: np.float(64)) -> np.float(64):
        if n < m:
            n, m = m, n

        if n > np.power(2, np.power(2, self.q) + self.r):
            return np.iinfo(np.uint64).max
        elif n > np.power(2, self.p + 5):
            d = (4 * n / m) / np.power((1 + n) / m, 2)
            return self._c * np.power(2, self.p - self.r) * d + 0.5
        else:
            return self._expected_collision(n, m) / np.float64(self.p)

    def _expected_collision(self, n: np.float(64), m: np.float(64)) -> np.float(64):
        x = np.float64(0)

        for i in range(1, self._2q + 1):
            for j in range(1, self._2r + 1):
                if i != self._2q:
                    den = np.power(2, self.p + self.r + i)
                    b1: np.float64 = (self._2r + j) / den
                    b2: np.float64 = (self._2r + j + 1) / den
                else:
                    den = np.power(2, self.p + self.r + i - 1)
                    b1: np.float64 = j / den
                    b2: np.float64 = (j + 1) / den

                prx = np.power(1 - b2, n) - np.power(1 - b1, n)
                pry = np.power(1 - b2, m) - np.power(1 - b1, m)
                x += (prx * pry)

        return (x * np.float64(self.p)) + 0.5

    def intersection(self, other: "HyperMinHash") -> np.uint64:
        """
        Intersection returns number of intersections between sk and other
        """
        sim = self.similarity(other)
        return np.uint64((sim * np.float64(self.merge(other).cardinality()) + 0.5))
