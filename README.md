# MAHA

MAHA is an in-progress ETL package which uses machine learning to clean your dataset with one line command. Features of MAHA include :-

  - Drop all the index columns
  - Drop columns with too many missing values
  - Using Regression to find the missing values in the data and then replacing them

# Prerequisites

  - Data is in pandas DataFrame format
  - All the categorical variables are label encoded
  - All the columns are in the desired data type of the output

You can also:
  - Find the mean and mode of every column
  - Fill the NA values with mean and mode of the columnns depending on the datatype
  - Find a model for every column with all other columns being the independent variables 

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
### Dependencies

MAHA uses a number of open source projects to work properly:

* [NumPy] - NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
* [Pandas] - Pandas is a software library written for the Python programming language for data manipulation and analysis.
* [Sklearn] - Machine Learning library which includes various classification, regression and clustering algorithms

### Installation

MAHA requires pandas, numpy and sklearn

Use pip to install the packages

```sh
$ pip3 install pandas
```
```sh
$ pip3 install numpy
```
```sh
$ pip3 install sklearn
```

If you have not installed pip, you can do it by

```sh
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
Then run the following command where you have installed get-pip.py
```
$ python get-pip.py
```

### Development

Developed By :-
[Mithesh R] - GitHub 
[Arth Akhouri] - GitHub
[Heetansh Jhaveri] - GitHub
[Ayaan Khan] - GitHub

Want to contribute? Navigate to our GitHub for more information
GitHub Repository - [MAHA]

License
----

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [MAHA]: <https://github.com/FlintyTub49/MAHA>
   [NumPy]: <https://numpy.org>
   [Pandas]: <https://pandas.pydata.org>
   [Sklearn]: <https://scikit-learn.org/stable/>
   [Arth Akhouri]: <https://github.com/user/FlintyTub49>
   [Mithesh R]: <https://github.com/user/259-mit>
   [Heetansh Jhaveri]: <https://github.com/user/hjj31>
   [Ayaan Khan]: <https://github.com/user/ayaan-27>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
