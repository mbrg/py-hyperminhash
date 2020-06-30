from typing import Dict

import numpy as np

from hyperminhash import Register, m, reg_sum_and_zeros, Sketch


LETTER_BYTES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def estimate_error(got, exp: int) -> float:

	if got > exp:
		delta = got - exp
	else:
		delta = exp - got

	return delta / exp


def test_zeros(ln: int = m):
	registers = []
	exp = 0.0

	for i in range(ln):
		val = Register(np.random.randint(0, np.iinfo(np.uint16).max))
		if val.lz() == 0:
			exp += 1
		registers.append(val)
	_, got = reg_sum_and_zeros(registers)

	assert got == exp, f"expected {exp:.2f}, got {got:.2f}"


def test_all_zeros():
	registers = [Register() for _ in range(m)]
	exp = 16384.0

	_, got = reg_sum_and_zeros(registers)
	assert got == exp, f"expected {exp:.2f}, got {got:.2f}"


def rand_string_bytes_mask_impr_src(n: np.uint32) -> str:
	return "".join(
		[LETTER_BYTES[np.random.randint(0, len(LETTER_BYTES))] for 	_ in range(n)])


def test_cardinality():
	sk = Sketch()
	step = 10000
	unique: Dict[str, bool] = {}

	for i in range(1, 1000000 + 1):
		st = rand_string_bytes_mask_impr_src(np.uint32(np.random.randint(0, 32)))
		b = str.encode(st)
		sk.add(b)
		unique[st] = True

		if len(unique) % step == 0:
			exact = np.uint64(len(unique))
			res = np.uint64(sk.cardinality())
			step *= 10

			ratio = 100 * estimate_error(res, exact)
			assert ratio <= 2, f"Exact {exact}, got {res} which is {ratio:.2f} error"
