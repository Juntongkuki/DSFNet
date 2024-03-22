.. try_docs documentation master file, created by
   sphinx-quickstart on Sun Mar  3 01:45:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TNLearn: Task-based Neurons for Learning
======================================
.. image:: ./_static/logo.png
   :target: https://github.com/NewT123-WM/tnlearn


.. image:: https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg
   :target: https://github.com/NewT123-WM/tnlearn
.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
   :target: https://github.com/NewT123-WM/tnlearn
.. image:: https://img.shields.io/badge/pypi-v0.1.0-orange?logo=PyPI
   :target: https://github.com/NewT123-WM/tnlearn
.. image:: https://img.shields.io/github/stars/NewT123-WM/tnlearn?style=flat&logo=github
   :target: https://github.com/NewT123-WM/tnlearn

-------------------------------------------------------------------

.. image:: https://img.shields.io/badge/ news: -F75D5D

``tnlearn`` is published in JMLR! Please cite `our paper <https://github.com/NewT123-WM/tnlearn>`_ if our tools are useful in your research!

-------------------------------------------------------------------

TNLearn
--------
``tnlearn`` , a Python package for implementing task-based neurons that aims to be easy to use, versatile for different data, and performant on different tasks.

- **Easy-to-use**: It provides a zero-barrier package for novices and a state-of-the-art benchmark for experienced researchers. Users can get results in 8 lines of code.
- **Versatile**: TNLearn constructs task-based neurons which are versatile for different data such as tabular data, images, and time-series. Users only need to collect the input and output data.
- **Performant**: Because task-based neurons capture the useful prior knowledge from task-related data, the network that is made up of task-based neurons can integrate the task-driven forces, which given the same structure should outperform the network of generic neurons.


Installation
-------------
This page provides a brief introduction to graph matching and some guidelines for using pygmtools. If you are seeking some background information, this is the right place!

.. important:: Please ensure that the versions of packages meet the requirements:

.. code-block:: linux
   :linenos:

   h5py~=3.10.0
   numpy~=1.26.2
   tnlearn~=0.1
   torch~=2.1.0
   sympy~=1.12
   setuptools~=68.0.0
   scikit-learn~=1.4.0
   scipy~=1.12.0
   joblib~=1.3.2
   requests~=2.31.0
   networkx~=3.2.1
   matplotlib~=3.8.3
   pandas~=2.2.0
   packaging~=23.2
   ipython~=8.18.1
   tqdm~=4.66.2

Run the following command to install `TNLearn <https://github.com/NewT123-WM/tnlearn>`_ from PyPI:

.. code-block:: linux
   :linenos:

   pip install tnlearn



.. toctree::
   :maxdepth: 2
   :caption: Documentation

   README_Page_1.md
   Page_2
   Page_3
   Page_4


The Team
-------------
``tnlearn`` is a work by:


- Meng Wang (`NewT123-WM <https://github.com/NewT123-WM>`_)
- Fenglei Fan (`FengleiFan Fan <https://github.com/FengleiFan>`_)
- Juntong Fan (`Juntongkuki <https://github.com/Juntongkuki>`_)


Citing
---------
If you find ``tnlearn`` useful, please cite it in your publications.

.. code-block:: linux
   :linenos:
   @article{


   }

License
-------
``tnlearn`` is released under the BSD 3-Clause License.


About Version Update
---------------------
We plan to keep this package up-to-date by including more architectures such as transformer and Mamba.