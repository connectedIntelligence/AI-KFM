import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ai-kfm",
    version = "0.0.1",
    author = "Shivam Pandey",
    author_email = "pandeyshivam2017robotics@gmail.com",
    description = ("Pytorch implementation for AI-KFM challenge."),
    license = "AGPLv3",
    keywords = "DeepLearning Pytorch AI-KFM",
    url = "https://github.com/connectedIntelligence/AI-KFM.git",
    packages=['aikfm', 'aikfm.models'],
    install_requires=[
        'roifile',
        'pillow',
        'pytorch'
    ],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: Alpha",
        "Topic :: Research",
        "License :: OSI Approved :: AGPLv3 License",
    ],
)
