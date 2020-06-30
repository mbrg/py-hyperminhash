from typing import Dict

import numpy as np
import string

from hyperminhash import Register, m, reg_sum_and_zeros, Sketch


def estimate_error(got, exp: int) -> float:

	if got > exp:
		delta = got - exp
	else:
		delta = exp - got

	return 100 * delta / exp


def rnd_str(size: int):
	arr = np.random.choice(string.ascii_letters, size)
	return "".join(list(arr))


def test_zeros(ln: int = m, exp: float = 0.0):
	registers = []

	for i in range(ln):
		val = Register(np.random.randint(0, np.iinfo(np.uint16).max))
		if val.lz() == 0:
			exp += 1
		registers.append(val)
	_, got = reg_sum_and_zeros(registers)

	assert got == exp, f"expected {exp:.2f}, got {got:.2f}"


def test_all_zeros(exp: float = 16384.0):
	registers = [Register() for _ in range(m)]

	_, got = reg_sum_and_zeros(registers)
	assert got == exp, f"expected {exp:.2f}, got {got:.2f}"


def test_cardinality():
	sk = Sketch()
	step = 10000
	unique: Dict[str, bool] = {}

	for _ in range(1000000):
		st = rnd_str(32)
		b = str.encode(st)
		sk.add(b)
		unique[st] = True

		if len(unique) % step == 0:
			exact = np.uint64(len(unique))
			res = np.uint64(sk.cardinality())
			step *= 10

			ratio = estimate_error(res, exact)
			assert ratio <= 2, f"Exact {exact}, got {res} which is {ratio:.2f} error. String: {st}."


def test_merge():
	sk1 = Sketch()
	sk2 = Sketch()
	unique: Dict[str, bool] = {}

	for _ in range(3500000):
		for sk in (sk1, sk2):
			st = rnd_str(32)
			b = str.encode(st)
			sk.add(b)
			unique[st] = True

	for _sk1, _sk2 in ((sk1, sk2), (sk2, sk1)):
		msk = _sk1.merge(_sk2)
		exact = np.uint64(len(unique))
		res = msk.cardinality()

		ratio = estimate_error(res, exact)
		assert ratio <= 2, f"Exact {exact}, got {res} which is {ratio:.2f} error."


def test_intersection():
	iters = 20
	k = 1000000

	for j in range(1, iters + 1):
		sk1 = Sketch()
		sk2 = Sketch()
		unique: Dict[str, bool] = {}

		frac = np.float64(j) / np.float64(iters)

		for i in range(k):
			st = str(i)
			b = str.encode(st)
			sk1.add(b)
			unique[st] += 1

		for i in range(int(np.float64(k) * frac), 2 * k):
			st = str(i)
			b = str.encode(st)
			sk2.add(b)
			unique[st] += 1

		col = 0
		for count in unique.values():
			if count > 1:
				col += 1

		exact = np.uint64(k - int(np.float64(k) * frac))
		res = sk1.intersection(sk2)

		ratio = estimate_error(res, exact)
		assert ratio <= 100, f"Exact {exact}, got {res} which is {ratio:.2f} error."


def test_no_intersection():
	sk1 = Sketch()
	sk2 = Sketch()

	for i in range(1000000):
		st = str(i)
		b = str.encode(st)
		sk1.add(b)

	for i in range(1000000, 2000000):
		st = str(i)
		b = str.encode(st)
		sk2.add(b)

	got = sk1.intersection(sk2)
	assert got == 0, f"Expected no intersection, got {got}."
