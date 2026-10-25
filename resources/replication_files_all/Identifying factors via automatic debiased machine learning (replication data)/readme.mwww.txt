The R codes reproduce the main results as in Identifying Factors via Automatic Debiased Machine Learning (Journal of Applied Econometrics, forthcoming)
coauthored by Esfandiar Maasoumi, Jianqiu Wang, Zhuo Wang, and Ke Wu


DATA SOURCES

The tables and figures of this paper are produced using factor library and test portfolios from Feng, Giglio, and Xiu (2020)

factors.xls: 150 risk factors from June 1976 to December 2017

summary.xls: Detailed description of 150 factors

factors_cn.xls: 132 risk factors of Chinese stock market from January 2000 to December 2021

port_3x2.xls:  a set of 3×2 portfolios from Feng, Giglio, and Xiu (2020), include 3×2 portfolios sorted by size and book-to-market ratio, 3×2 portfolios sorted by size and operating profitability, 3 × 2 portfolios sorted by size and investment, 3 × 2 portfolios sorted by size and short-term reversal on prior (1-1) return, 3 × 2 portfolios sorted by size and momentum on prior (2-12) return, and 3 × 2 portfolios sorted by size and long-term reversal on prior (13-60) return and 714 additional 3 × 2 portfolios cover additional characteristics.

port_3x2_id.xls: information to retain portfolios with more than 10 stocks

port_5x5.xls: a set of 5×5 portfolios from Feng, Giglio, and Xiu (2020)

port_5x5_id.xls: information to retain portfolios with more than 10 stocks

port202.xls:  202 portfolios employed by Giglio and Xiu (2021)

testing_assets_cn.xls: 762 3 × 2 bivariate-sorted portfolios in Chinese stock market


CODE DESCRIPTION

dataimport.R: Read factor and test asset data

Recursive_neural_network_3x2.R: ADML estimates with neural network to estimate γ0 and 3×2 bivariate-sorted portfolios as test assets. The result of the last row is an estimate of the whole sample. 

Recursive_neural_network_5x5.R: ADML estimates with neural network to estimate γ0 and 5×5 bivariate-sorted portfolios as test assets.

Recursive_neural_network_202.R: ADML estimates with neural network to estimate γ0 and 202 bivariate-sorted portfolios as test assets.

Recursive_random_forest_3x2.R:  ADML estimates with random forest to estimate γ0 and 3×2 bivariate-sorted portfolios as test assets.

Recursive_random_forest_5x5.R: ADML estimates with random forest to estimate γ0 and 5×5 bivariate-sorted portfolios as test assets.

Recursive_random_forest_202.R: ADML estimates with random forest to estimate γ0 and 202 bivariate-sorted portfolios as test assets.

Recursive_ols.R: Recursive estimates of all factors by OLS.

Recursive_double_selection_lasso.R: Recursive estimates of all factors by Double-selection Lasso of Feng, Giglio, and Xiu (2020).

Plot_factor_significance_rate.R: plot the factor significance rate of expanding window ADML estimates.

Time_series_performance.R: output the time series performance in each category.

Robustness_tuning.R：calculate ADML absolute t-statistics with different tuning parameters and numbers of sample-splitting subsets.

Plot_heatmap.R: plot the heat maps for ADML absolute t-statistics of the top 10 significant factors with different tuning parameters and numbers of sample-splitting subsets.

Plot_factor_importance.R: plot the factor importance with different specifications.

Spanning_test.R: spanning test result

Identify_Factors_Chinese_market.R: ADML estimates of 132 risk factors in Chinese stock market.

simulation_**.R: simulation code with various parameters settings


REPLICATION PROCESS

TABLE 1: Run Recursive_neural_network_3x2.R (the result of neural network with 3 units in the hidden layer)

TABLE 2 : Run Spanning_test.R and obtain the results of the Spanning test for each factor model

TABLE 3: Run Identify_Factors_Chinese_market.R. 

FIGURE 1: Run simulation_p100_n150_t480.R, simulation_p100_n300_t480.R, simulation_p100_n450_t480.R

FIGURE 2: Run Recursive_neural_network_3x2.R, and then run Plot_factor_significance_rate.R

FIGURE 3: Run Recursive_neural_network_3x2.R, Recursive_double_selection_lasso.R, Recursive_ols.R, and then run Time_series_performance.R

FIGURE 4: Run Robustness_tuning.R, and then run Plot_heatmap

FIGURE 5: Run Recursive_neural_network_3x2.R, Recursive_neural_network_5x5.R, Recursive_neural_network_202.R, Recursive_random_forest_3x2.R, Recursive_random_forest_5x5.R, Recursive_random_forest_202.R, and then run  Plot_factor_importance.R


Note: For speed, we use R package "foreach" and "doParallel" for parallel operation in some codes. You need to adjust parameter "cores" to match your computer for achieving optimal computational efficiency. Running each code for ADML estimation takes approximately a few hours, depending on your computer.


If you find any errors, please get in touch with us. If you use the code, please cite the paper.
