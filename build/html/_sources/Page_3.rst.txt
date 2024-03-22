
Get Started
===========================

Packages Required
-----------------

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



Basic Install by pip
---------------------
Run the following command to install `tnlearn <https://github.com/NewT123-WM/tnlearn>`_ from PyPI:

.. code-block:: linux
   :linenos:

    pip install tnlearn


Build Shared Libraries from Source
-----------------------------------
Firstly, you can run the following command in ``Git Bash`` to clone the library ``tnlearn`` to a specified local folder:

.. code-block:: linux
   :linenos:

    git clone https://github.com/NewT123-WM/tnlearn.git

Secondly, you can run the following command in ``Anaconda Prompt`` to install ``tnlearn`` to the specified virtual environment:

.. code-block:: linux
   :linenos:

    cd tnlearn
    pip3 install .

.. tip::
   After entering the above installation command, the **required packages** will be automatically installed. If there is any error with a required package during automatic installation, you can try to manually install the problematic package separately and then execute the ``pip3 install tnlearn`` again.




What's Next
------------

The next page :doc:`API Reference <Page_4>` provides detailed information about ``tnlearn``, along with simple examples to help users quickly get started.