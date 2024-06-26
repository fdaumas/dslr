# dslr
## Description
Subject created by the 42AI association. Discover Data Science in the projects where you re-constitute Poudlard’s Sorting Hat. Warning: this is not a subject on cameras.

## Prerequisites
ft_linear_regression

## Team or Solo Project
Tean project of 2 students : lburnet and fdaumas

## Skills
* DB & Data
* Algorithms & AI 

## Keywords
* DataScience
* Logistic Regression 

## Cursus
* for RNCP level 7 
* Module AI

## Personnal notes
### useful links
* argc argv in python : https://realpython.com/python-command-line-arguments/#:~:text=argc%20is%20an%20integer%20representing,remaining%20elements%20of%20the%20array
* pandas count : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.count.html
* nan values : https://www.turing.com/kb/nan-values-in-python
* drop line with na values : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
* create df : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
* histogram : https://matplotlib.org/stable/gallery/statistics/hist.html
* scatter plot : https://matplotlib.org/stable/plot_types/basic/scatter_plot.html#sphx-glr-plot-types-basic-scatter-plot-py
* pair plot : 
	* https://seaborn.pydata.org/generated/seaborn.pairplot.html
	* https://www.geeksforgeeks.org/python-seaborn-pairplot-method/
* Logistic Regression :
	* https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5
	* https://medium.com/analytics-vidhya/logistic-regression-from-scratch-multi-classification-with-onevsall-d5c2acf0c37c
    * https://www.youtube.com/watch?v=QqAUqRAWEV8
* subject of bootcamp 42AI Module 08 : https://github.com/42-AI/bootcamp_machine-learning/releases

### Notes
* we choose Ancient Runes and Herbology for logistic regression
* for predicting in 2 times :
	* G and S -> bool
	* G and H -> bool
	* So if :
		* G & H got 1 and G & S got 1 then result G
		* G & H got 1 and G & S got 0 then result H
		* G & H got 0 and G & S got 1 then result S
		* G & H got 0 and G & S got 0 then result R