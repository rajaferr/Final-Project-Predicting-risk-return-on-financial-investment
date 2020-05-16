#!/usr/bin/env python
# coding: utf-8

# ## 1. Meet Professor William Sharpe
# <p>An investment may make sense if we expect it to return more money than it costs. But returns are only part of the story because they are risky - there may be a range of possible outcomes. How does one compare different investments that may deliver similar results on average, but exhibit different levels of risks?</p>
# <p>Enter William Sharpe. He introduced the <a href="https://web.stanford.edu/~wfsharpe/art/sr/sr.htm"><em>reward-to-variability ratio</em></a> in 1966 that soon came to be called the Sharpe Ratio. It compares the expected returns for two investment opportunities and calculates the additional return per unit of risk an investor could obtain by choosing one over the other. In particular, it looks at the difference in returns for two investments and compares the average difference to the standard deviation (as a measure of risk) of this difference. A higher Sharpe ratio means that the reward will be higher for a given amount of risk. It is common to compare a specific opportunity against a benchmark that represents an entire category of investments.</p>
# <p>The Sharpe ratio has been one of the most popular risk/return measures in finance, not least because it's so simple to use. It also helped that Professor Sharpe won a Nobel Memorial Prize in Economics in 1990 for his work on the capital asset pricing model (CAPM).</p>
# <p>Let's learn about the Sharpe ratio by calculating it for the stocks of the two tech giants Facebook and Amazon. As a benchmark, we'll use the S&amp;P 500 that measures the performance of the 500 largest stocks in the US.</p>

# In[5]:


# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings to produce nice plots in a Jupyter notebook
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading in the data
stock_data = pd.read_csv('./input/stock_data.csv',
                          parse_dates=['Date'], 
                          index_col=['Date']).dropna()

benchmark_data = pd.read_csv('./input/benchmark_data.csv', 
                             parse_dates=['Date'], 
                             index_col=['Date']).dropna()


# ## 2. A first glance at the data
# <p>Let's take a look the data to find out how many observations and variables we have at our disposal.</p>

# In[6]:


# Display summary for stock_data
print('Stocks\n')
stock_data.info()
stock_data.head()

# Display summary for benchmark_data
print('\nBenchmarks\n')
benchmark_data.info()
benchmark_data.head()


# ## 3. Plot & summarize daily prices for Amazon and Facebook
# <p>Before we compare an investment in either Facebook or Amazon with the index of the 500 largest companies in the US, let's visualize the data, so we better understand what we're dealing with.</p>

# In[7]:


# visualize the stock_data
stock_data.plot(subplots=True)
plt_title = 'Stock Data'
plt.show()


# summarize the stock_data
stock_data.describe()


# ## 4. Visualize & summarize daily values for the S&P 500
# <p>Let's also take a closer look at the value of the S&amp;P 500, our benchmark.</p>

# In[8]:


# plot the benchmark_data
benchmark_data.plot()
plt_title = 'S&P 500'
plt.show()


# summarize the benchmark_data
benchmark_data.describe()


# ## 5. The inputs for the Sharpe Ratio: Starting with Daily Stock Returns
# <p>The Sharpe Ratio uses the difference in returns between the two investment opportunities under consideration.</p>
# <p>However, our data show the historical value of each investment, not the return. To calculate the return, we need to calculate the percentage change in value from one day to the next. We'll also take a look at the summary statistics because these will become our inputs as we calculate the Sharpe Ratio. Can you already guess the result?</p>

# In[9]:


# calculate daily stock_data returns
stock_returns = stock_data.pct_change()

# plot the daily returns
stock_returns.plot()
plt.show()


# summarize the daily returns
stock_returns.describe()


# ## 6. Daily S&P 500 returns
# <p>For the S&amp;P 500, calculating daily returns works just the same way, we just need to make sure we select it as a <code>Series</code> using single brackets <code>[]</code> and not as a <code>DataFrame</code> to facilitate the calculations in the next step.</p>

# In[10]:


# calculate daily benchmark_data returns
# ... YOUR CODE FOR TASK 6 HERE ...
sp_returns = benchmark_data['S&P 500'].pct_change()

# plot the daily returns
sp_returns.plot()
plt.show()


# summarize the daily returns
sp_returns.describe()


# ## 7. Calculating Excess Returns for Amazon and Facebook vs. S&P 500
# <p>Next, we need to calculate the relative performance of stocks vs. the S&amp;P 500 benchmark. This is calculated as the difference in returns between <code>stock_returns</code> and <code>sp_returns</code> for each day.</p>

# In[11]:


# calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plot the excess_returns
excess_returns.plot()
plt.show()


# summarize the excess_returns
excess_returns.describe()


# ## 8. The Sharpe Ratio, Step 1: The Average Difference in Daily Returns Stocks vs S&P 500
# <p>Now we can finally start computing the Sharpe Ratio. First we need to calculate the average of the <code>excess_returns</code>. This tells us how much more or less the investment yields per day compared to the benchmark.</p>

# In[12]:


# calculate the mean of excess_returns 
# ... YOUR CODE FOR TASK 8 HERE ...
avg_excess_return = excess_returns.mean()

# plot avg_excess_returns
avg_excess_return.plot.bar()
plt_title = 'Mean of the Return Difference'
plt.show()


# ## 9. The Sharpe Ratio, Step 2: Standard Deviation of the Return Difference
# <p>It looks like there was quite a bit of a difference between average daily returns for Amazon and Facebook.</p>
# <p>Next, we calculate the standard deviation of the <code>excess_returns</code>. This shows us the amount of risk an investment in the stocks implies as compared to an investment in the S&amp;P 500.</p>

# In[13]:


# calculate the standard deviations
sd_excess_return = excess_returns.std()

# plot the standard deviations
sd_excess_return.plot()
plt_title = 'Standard Deviation of the Return Difference'
plt.show()


# ## 10. Putting it all together
# <p>Now we just need to compute the ratio of <code>avg_excess_returns</code> and <code>sd_excess_returns</code>. The result is now finally the <em>Sharpe ratio</em> and indicates how much more (or less) return the investment opportunity under consideration yields per unit of risk.</p>
# <p>The Sharpe Ratio is often <em>annualized</em> by multiplying it by the square root of the number of periods. We have used daily data as input, so we'll use the square root of the number of trading days (5 days, 52 weeks, minus a few holidays): √252</p>

# In[14]:


# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plot the annualized sharpe ratio
annual_sharpe_ratio.plot()
plt_title = 'Annualized Sharpe Ratio: Stocks vs S&P 500'
plt.show()


# ## 11. Conclusion
# <p>Given the two Sharpe ratios, which investment should we go for? In 2016, Amazon had a Sharpe ratio twice as high as Facebook. This means that an investment in Amazon returned twice as much compared to the S&amp;P 500 for each unit of risk an investor would have assumed. In other words, in risk-adjusted terms, the investment in Amazon would have been more attractive.</p>
# <p>This difference was mostly driven by differences in return rather than risk between Amazon and Facebook. The risk of choosing Amazon over FB (as measured by the standard deviation) was only slightly higher so that the higher Sharpe ratio for Amazon ends up higher mainly due to the higher average daily returns for Amazon. </p>
# <p>When faced with investment alternatives that offer both different returns and risks, the Sharpe Ratio helps to make a decision by adjusting the returns by the differences in risk and allows an investor to compare investment opportunities on equal terms, that is, on an 'apples-to-apples' basis.</p>

# In[ ]:





# In[ ]:




