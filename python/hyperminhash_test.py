import numpy as np

from hyperminhash import Register, m, reg_sum_and_zeros


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

