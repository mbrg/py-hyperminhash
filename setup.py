from setuptools import setup, find_packages

setup(
    name="hyperminhash",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/mibarg/hyperminhash",
    license="MIT",
    author="mibarg",
    author_email="mibarg@users.noreply.github.com",
    install_requires=["metrohash-python", ],
    python_requires=">=3.6",
)
