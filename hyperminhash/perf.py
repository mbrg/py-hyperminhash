def estimate_error(got, exp: int) -> float:
    if got == exp == 0:
        return 0.0
    if got != 0 and exp == 0:
        return 100.0

    if got > exp:
        delta = got - exp
    else:
        delta = exp - got

    return 100 * delta / exp
