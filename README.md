# HyperMinSketch

![Build Status](https://travis-ci.org/mibarg/py-hyperminhash.svg?branch=master)

This repository is a Python>=3.6 port of golang [hyperminhash](https://github.com/axiomhq/hyperminhash):

> Besides being a compact and pretty speedy HyperLogLog implementation for cardinality counting, this modified HyperLogLog allows **intersection** and **similarity** estimation of different HyperLogLogs.

## Install
```
pip install hyperminhash
```

## Example Usage
```python
from hyperminhash import HyperMinHash

sk1 = HyperMinHash()
sk2 = HyperMinHash()

for i in range(10000):
    sk1.add(i)

print(len(sk1))
# 10001 (should be 10000)

for i in range(3333, 23333):
    sk2.add(i)

print(len(sk2))         
# 19977 (should be 20000)

print(sk1.similarity(sk2))
# 0.284589082 (should be 0.2857326533)

print(sk1.intersection(sk2))
# 6623 (should be 6667)

sk1.merge(sk2)
print(sk1.cardinality())
# 23271 (should be 23333)
```