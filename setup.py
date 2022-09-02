from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fsspec-chfs',
    version='0.0.1',
    author='Kazuki Obata, Sohei Koyama',
    author_email='obata@hpcs.cs.tsukuba.ac.jp, skoyama@hpcs.cs.tsukuba.ac.jp',
    description='Pythonic file system for CHFS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/donkomura/fsspec-chfs',
    packages=['chfs', 'chfs.dask'],
    entry_points={
        'fsspec.specs': [
            'chfs=chfs.CHFSClient',
            'chfs_stub=chfs.CHFSStubClient'
        ]
    },
    install_requires = [
        'pychfs>=0.0.1',
    ],
)
