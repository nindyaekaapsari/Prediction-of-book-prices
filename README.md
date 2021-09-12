# Import Dataset


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
dfr=pd.read_csv(r'D:/Coolyeah/LOMBA/Find It/public dataset/public-train.csv', delimiter = '|')
dft=pd.read_csv(r'D:/Coolyeah/LOMBA/Find It/public dataset/public-test.csv', delimiter = '|')
```


```python
dfr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>author2106</td>
      <td>Just after the Second World War, in the small ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>309.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>author1018</td>
      <td>Blame it on Hawaii’s rainbows, sparkling beach...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>author1087</td>
      <td>The Pulitzer Prize–winning, bestselling author...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>author1295</td>
      <td>THINGS ARE ABOUT TO GET SERIOUS FOR HARRY DRES...</td>
      <td>Hardcover</td>
      <td>First Edition</td>
      <td>418.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.30</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>author2622</td>
      <td>The Romanovs were the most successful dynasty ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>784.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.30</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
dft.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>author2305</td>
      <td>Rachel Friedman has always been the consummate...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>295.0</td>
      <td>March 29, 2011</td>
      <td>publisher034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>129789.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>author0204</td>
      <td>As Dr. Marina Singh embarks upon an uncertain ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>353.0</td>
      <td>June 7, 2011</td>
      <td>publisher155</td>
      <td>NaN</td>
      <td>990</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>262465.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>author2300</td>
      <td>From the moment she took a job on Captain Cald...</td>
      <td>Paperback</td>
      <td>US edition</td>
      <td>373.0</td>
      <td>April 22, 2014</td>
      <td>publisher261</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.16</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>182195.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>author1746</td>
      <td>#1 New York Times bestseller Lisa Gardner, aut...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>423.0</td>
      <td>February 5, 2013</td>
      <td>publisher105</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.18</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>288596.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>author1716</td>
      <td>This is not your mother’s memoir. In The Chron...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>310.0</td>
      <td>April 1, 2011</td>
      <td>publisher166</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.24</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>230270.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



# Data Understanding


```python
print("train:",dfr.shape)
print("test:", dft.shape)
```

    train: (3550, 39)
    test: (500, 39)
    


```python
print("train:",dfr.dtypes)
print("----------------",)
print("test:",dft.dtypes )
```

    train: author_id          object
    description        object
    bookformat         object
    bookedition        object
    pages             float64
    published_date     object
    publisher_id       object
    reading_age        object
    lexile_measure     object
    grade_level        object
    weight            float64
    rating_value_0    float64
    rating_value_1    float64
    rating_count_0      int64
    rating_count_1      int64
    dimension_0       float64
    dimension_1       float64
    dimension_2       float64
    genre_0            object
    genre_1            object
    genre_2            object
    genre_3            object
    genre_4            object
    genre_5            object
    genre_6            object
    genre_7            object
    genre_8            object
    genre_9            object
    genre_0_weight    float64
    genre_1_weight    float64
    genre_2_weight    float64
    genre_3_weight    float64
    genre_4_weight    float64
    genre_5_weight    float64
    genre_6_weight    float64
    genre_7_weight    float64
    genre_8_weight    float64
    genre_9_weight    float64
    price             float64
    dtype: object
    ----------------
    test: author_id          object
    description        object
    bookformat         object
    bookedition        object
    pages             float64
    published_date     object
    publisher_id       object
    reading_age        object
    lexile_measure     object
    grade_level        object
    weight            float64
    rating_value_0    float64
    rating_value_1    float64
    rating_count_0      int64
    rating_count_1      int64
    dimension_0       float64
    dimension_1       float64
    dimension_2       float64
    genre_0            object
    genre_1            object
    genre_2            object
    genre_3            object
    genre_4            object
    genre_5            object
    genre_6            object
    genre_7            object
    genre_8            object
    genre_9            object
    genre_0_weight    float64
    genre_1_weight    float64
    genre_2_weight    float64
    genre_3_weight    float64
    genre_4_weight    float64
    genre_5_weight    float64
    genre_6_weight    float64
    genre_7_weight    float64
    genre_8_weight    float64
    genre_9_weight    float64
    price             float64
    dtype: object
    


```python
dfr.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3550 entries, 0 to 3549
    Data columns (total 39 columns):
    author_id         3540 non-null object
    description       3512 non-null object
    bookformat        3537 non-null object
    bookedition       231 non-null object
    pages             3451 non-null float64
    published_date    568 non-null object
    publisher_id      568 non-null object
    reading_age       126 non-null object
    lexile_measure    88 non-null object
    grade_level       100 non-null object
    weight            519 non-null float64
    rating_value_0    3540 non-null float64
    rating_value_1    553 non-null float64
    rating_count_0    3550 non-null int64
    rating_count_1    3550 non-null int64
    dimension_0       512 non-null float64
    dimension_1       512 non-null float64
    dimension_2       499 non-null float64
    genre_0           3400 non-null object
    genre_1           3353 non-null object
    genre_2           3322 non-null object
    genre_3           3302 non-null object
    genre_4           3270 non-null object
    genre_5           3240 non-null object
    genre_6           3212 non-null object
    genre_7           3172 non-null object
    genre_8           3136 non-null object
    genre_9           3100 non-null object
    genre_0_weight    3400 non-null float64
    genre_1_weight    3353 non-null float64
    genre_2_weight    3322 non-null float64
    genre_3_weight    3302 non-null float64
    genre_4_weight    3270 non-null float64
    genre_5_weight    3240 non-null float64
    genre_6_weight    3212 non-null float64
    genre_7_weight    3172 non-null float64
    genre_8_weight    3136 non-null float64
    genre_9_weight    3100 non-null float64
    price             543 non-null float64
    dtypes: float64(18), int64(2), object(19)
    memory usage: 1.1+ MB
    


```python
dft.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 39 columns):
    author_id         500 non-null object
    description       492 non-null object
    bookformat        500 non-null object
    bookedition       54 non-null object
    pages             480 non-null float64
    published_date    469 non-null object
    publisher_id      469 non-null object
    reading_age       119 non-null object
    lexile_measure    97 non-null object
    grade_level       95 non-null object
    weight            438 non-null float64
    rating_value_0    500 non-null float64
    rating_value_1    444 non-null float64
    rating_count_0    500 non-null int64
    rating_count_1    500 non-null int64
    dimension_0       437 non-null float64
    dimension_1       437 non-null float64
    dimension_2       419 non-null float64
    genre_0           459 non-null object
    genre_1           453 non-null object
    genre_2           445 non-null object
    genre_3           437 non-null object
    genre_4           432 non-null object
    genre_5           427 non-null object
    genre_6           423 non-null object
    genre_7           420 non-null object
    genre_8           418 non-null object
    genre_9           413 non-null object
    genre_0_weight    459 non-null float64
    genre_1_weight    453 non-null float64
    genre_2_weight    445 non-null float64
    genre_3_weight    437 non-null float64
    genre_4_weight    432 non-null float64
    genre_5_weight    427 non-null float64
    genre_6_weight    423 non-null float64
    genre_7_weight    420 non-null float64
    genre_8_weight    418 non-null float64
    genre_9_weight    413 non-null float64
    price             500 non-null float64
    dtypes: float64(18), int64(2), object(19)
    memory usage: 152.5+ KB
    


```python
dfr.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pages</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>dimension_1</th>
      <th>dimension_2</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>3451.000000</td>
      <td>519.000000</td>
      <td>3540.000000</td>
      <td>553.000000</td>
      <td>3.550000e+03</td>
      <td>3550.000000</td>
      <td>512.000000</td>
      <td>512.000000</td>
      <td>499.000000</td>
      <td>3400.000000</td>
      <td>3353.000000</td>
      <td>3322.000000</td>
      <td>3302.000000</td>
      <td>3270.000000</td>
      <td>3240.00000</td>
      <td>3212.000000</td>
      <td>3172.000000</td>
      <td>3136.000000</td>
      <td>3100.000000</td>
      <td>543.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>331.894234</td>
      <td>2372.200829</td>
      <td>4.025017</td>
      <td>4.519530</td>
      <td>5.504540e+04</td>
      <td>370.759155</td>
      <td>15.438125</td>
      <td>3.686113</td>
      <td>22.091182</td>
      <td>0.427418</td>
      <td>0.198375</td>
      <td>0.119124</td>
      <td>0.077935</td>
      <td>0.054976</td>
      <td>0.04166</td>
      <td>0.032699</td>
      <td>0.026434</td>
      <td>0.021875</td>
      <td>0.018174</td>
      <td>231296.762431</td>
    </tr>
    <tr>
      <td>std</td>
      <td>147.472358</td>
      <td>2232.405524</td>
      <td>0.560412</td>
      <td>0.284086</td>
      <td>1.836514e+05</td>
      <td>2046.639890</td>
      <td>3.438329</td>
      <td>4.004741</td>
      <td>3.236897</td>
      <td>0.165297</td>
      <td>0.070470</td>
      <td>0.046482</td>
      <td>0.031668</td>
      <td>0.022640</td>
      <td>0.01698</td>
      <td>0.013457</td>
      <td>0.011010</td>
      <td>0.009474</td>
      <td>0.008107</td>
      <td>138233.508496</td>
    </tr>
    <tr>
      <td>min</td>
      <td>20.000000</td>
      <td>400.070000</td>
      <td>0.000000</td>
      <td>3.200000</td>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>1.470000</td>
      <td>0.030000</td>
      <td>1.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>256.000000</td>
      <td>612.350000</td>
      <td>3.880000</td>
      <td>4.400000</td>
      <td>3.565500e+03</td>
      <td>1.000000</td>
      <td>13.970000</td>
      <td>2.125000</td>
      <td>20.950000</td>
      <td>0.310000</td>
      <td>0.150000</td>
      <td>0.090000</td>
      <td>0.060000</td>
      <td>0.040000</td>
      <td>0.03000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>128922.500000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>330.000000</td>
      <td>839.150000</td>
      <td>4.080000</td>
      <td>4.600000</td>
      <td>1.308100e+04</td>
      <td>1.000000</td>
      <td>15.240000</td>
      <td>2.790000</td>
      <td>22.860000</td>
      <td>0.390000</td>
      <td>0.200000</td>
      <td>0.120000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.04000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>212946.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>400.000000</td>
      <td>4036.970000</td>
      <td>4.260000</td>
      <td>4.700000</td>
      <td>4.271300e+04</td>
      <td>1.000000</td>
      <td>16.360000</td>
      <td>3.580000</td>
      <td>24.130000</td>
      <td>0.500000</td>
      <td>0.240000</td>
      <td>0.150000</td>
      <td>0.100000</td>
      <td>0.070000</td>
      <td>0.05000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>287224.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1248.000000</td>
      <td>7212.110000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.803071e+06</td>
      <td>40409.000000</td>
      <td>30.480000</td>
      <td>27.430000</td>
      <td>28.910000</td>
      <td>1.000000</td>
      <td>0.600000</td>
      <td>0.330000</td>
      <td>0.250000</td>
      <td>0.150000</td>
      <td>0.12000</td>
      <td>0.110000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>978395.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Proportion Data


```python
import seaborn as sns

sns.countplot(dfr['price'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2f2d554ba48>




![png](output_12_1.png)



```python
sns.countplot(dft['price'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1767d838308>




![png](output_13_1.png)



```python
plt.figure(figsize=(10,5))
dfr['bookformat'].value_counts().plot.bar()
plt.title("Jumlah Format Book")
plt.show()
```


![png](output_14_0.png)



```python
plt.figure(figsize=(10,5))
dfr['genre_0'].value_counts().plot.bar()
plt.title("Jumlah Genre")
plt.show()
```


![png](output_15_0.png)



```python
plt.figure(figsize=(10,5))
dfr['reading_age'].value_counts().plot.bar()
plt.title("Reading Age")
plt.show()
```


![png](output_16_0.png)



```python
plt.figure(figsize=(10,5))
dfr['lexile_measure'].value_counts().plot.bar()
plt.title("Lexile Measure")
plt.show()
```


![png](output_17_0.png)



```python
plt.figure(figsize=(10,5))
dfr['grade_level'].value_counts().plot.bar()
plt.title("Grade Level")
plt.show()
```


![png](output_18_0.png)



```python
dfr['bookedition'].value_counts()
```




    First Edition                       76
    1st Edition                         20
    Trade                               13
    First Edition (U.S.)                11
    1st                                  9
                                        ..
    Boxed Set                            1
    Deluxe Edition                       1
    First Scribner Hardcover Edition     1
    Advanced Readers Copy                1
    Illustrated                          1
    Name: bookedition, Length: 68, dtype: int64




```python
dfr['reading_age'].value_counts()
```




    8 - 12 years       14
    18 years and up    14
    14 years and up    12
    4 - 8 years        11
    13 years and up    10
    12 - 15 years       7
    13 - 17 years       7
    10 - 14 years       7
    14 - 17 years       6
    12 - 17 years       4
    15 years and up     4
    12 - 18 years       3
    16 years and up     3
    3 - 7 years         2
    2 - 5 years         2
    3 - 5 years         2
    11 years and up     2
    12 years and up     2
    10 - 13 years       2
    4 - 7 years         1
    3 - 8 years         1
    8 - 11 years        1
    4 - 6 years         1
    9 - 12 years        1
    13 - 18 years       1
    3 - 6 years         1
    10 years            1
    5 - 8 years         1
    10 - 12 years       1
    10 years and up     1
    6 - 11 years        1
    Name: reading_age, dtype: int64



# Preprocessing


```python
#Cek missing value
miss_r = dfr.isnull().sum()
print(miss_r)
```

    author_id           10
    description         38
    bookformat          13
    bookedition       3319
    pages               99
    published_date    2982
    publisher_id      2982
    reading_age       3424
    lexile_measure    3462
    grade_level       3450
    weight            3031
    rating_value_0      10
    rating_value_1    2997
    rating_count_0       0
    rating_count_1       0
    dimension_0       3038
    dimension_1       3038
    dimension_2       3051
    genre_0            150
    genre_1            197
    genre_2            228
    genre_3            248
    genre_4            280
    genre_5            310
    genre_6            338
    genre_7            378
    genre_8            414
    genre_9            450
    genre_0_weight     150
    genre_1_weight     197
    genre_2_weight     228
    genre_3_weight     248
    genre_4_weight     280
    genre_5_weight     310
    genre_6_weight     338
    genre_7_weight     378
    genre_8_weight     414
    genre_9_weight     450
    price             3007
    dtype: int64
    


```python
#Duplicate data
dfr.duplicated().sum()
```




    109




```python
dft.duplicated().sum()
```




    8




```python
#Hapus duplikasi
dfr_drop = dfr.drop_duplicates()
```


```python
dft_drop = dft.drop_duplicates()
```


```python
dft_drop = dft
```


```python
dfr_drop
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>author2106</td>
      <td>Just after the Second World War, in the small ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>309.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>author1018</td>
      <td>Blame it on Hawaii’s rainbows, sparkling beach...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>author1087</td>
      <td>The Pulitzer Prize–winning, bestselling author...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>author1295</td>
      <td>THINGS ARE ABOUT TO GET SERIOUS FOR HARRY DRES...</td>
      <td>Hardcover</td>
      <td>First Edition</td>
      <td>418.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.30</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>author2622</td>
      <td>The Romanovs were the most successful dynasty ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>784.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.30</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3545</td>
      <td>author1144</td>
      <td>How much is too much to love? Travis Maddox le...</td>
      <td>Paperback</td>
      <td>Original Edition</td>
      <td>448.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.24</td>
      <td>0.10</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3546</td>
      <td>author2852</td>
      <td>Magneto and Professor X. Superman and Lex Luth...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>478.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.18</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3547</td>
      <td>author1309</td>
      <td>Following the launch of her #1 New York Times ...</td>
      <td>Hardcover</td>
      <td>First Edition</td>
      <td>352.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3548</td>
      <td>author1816</td>
      <td>Bachelors, beware. For those who keep secrets ...</td>
      <td>Kindle Edition</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3549</td>
      <td>author0882</td>
      <td>In the thrilling, nerve-wracking finale of Eze...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>315.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.10</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3441 rows × 39 columns</p>
</div>




```python
dft_drop
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>author2305</td>
      <td>Rachel Friedman has always been the consummate...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>295.0</td>
      <td>March 29, 2011</td>
      <td>publisher034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>129789.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>author0204</td>
      <td>As Dr. Marina Singh embarks upon an uncertain ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>353.0</td>
      <td>June 7, 2011</td>
      <td>publisher155</td>
      <td>NaN</td>
      <td>990</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>262465.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>author2300</td>
      <td>From the moment she took a job on Captain Cald...</td>
      <td>Paperback</td>
      <td>US edition</td>
      <td>373.0</td>
      <td>April 22, 2014</td>
      <td>publisher261</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.16</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>182195.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>author1746</td>
      <td>#1 New York Times bestseller Lisa Gardner, aut...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>423.0</td>
      <td>February 5, 2013</td>
      <td>publisher105</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.18</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>288596.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>author1716</td>
      <td>This is not your mother’s memoir. In The Chron...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>310.0</td>
      <td>April 1, 2011</td>
      <td>publisher166</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.24</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>230270.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>495</td>
      <td>author0029</td>
      <td>World War I stands as one of history's most se...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>448.0</td>
      <td>May 3, 2011</td>
      <td>publisher173</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.22</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>352263.0</td>
    </tr>
    <tr>
      <td>496</td>
      <td>author0164</td>
      <td>A sweeping, evocative epic of two women's inte...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>589.0</td>
      <td>November 5, 2013</td>
      <td>publisher109</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.33</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>307364.0</td>
    </tr>
    <tr>
      <td>497</td>
      <td>author0506</td>
      <td>She could remember standing in a park near the...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>April 4, 2011</td>
      <td>publisher147</td>
      <td>6 - 9 years</td>
      <td>1060L</td>
      <td>1 - 4</td>
      <td>...</td>
      <td>0.21</td>
      <td>0.14</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>274159.0</td>
    </tr>
    <tr>
      <td>498</td>
      <td>author2157</td>
      <td>Following on the success of Tender and Ripe, t...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>544.0</td>
      <td>September 24, 2013</td>
      <td>publisher350</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.19</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>489992.0</td>
    </tr>
    <tr>
      <td>499</td>
      <td>author0719</td>
      <td>For Nora Grey, romance was not part of the pla...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>120.0</td>
      <td>May 1, 2012</td>
      <td>publisher319</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.10</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>432966.0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 39 columns</p>
</div>




```python
dfr_drop.isnull().sum()
```




    author_id            1
    description         29
    bookformat           4
    bookedition       3217
    pages               90
    published_date    2881
    publisher_id      2881
    reading_age       3319
    lexile_measure    3358
    grade_level       3345
    weight            2929
    rating_value_0       1
    rating_value_1    2896
    rating_count_0       0
    rating_count_1       0
    dimension_0       2936
    dimension_1       2936
    dimension_2       2948
    genre_0            141
    genre_1            188
    genre_2            219
    genre_3            239
    genre_4            271
    genre_5            301
    genre_6            329
    genre_7            369
    genre_8            405
    genre_9            441
    genre_0_weight     141
    genre_1_weight     188
    genre_2_weight     219
    genre_3_weight     239
    genre_4_weight     271
    genre_5_weight     301
    genre_6_weight     329
    genre_7_weight     369
    genre_8_weight     405
    genre_9_weight     441
    price             2905
    dtype: int64




```python
dfr_drop.shape
```




    (3441, 39)




```python
dft_drop.shape
```




    (500, 39)




```python
ax = sns.boxplot(data=dfr_drop, orient="h", palette="Set2")
```


![png](output_33_0.png)



```python
fig, ax = plt.subplots(figsize=(10,8))
corrr = dfr_drop.corr()
sns.heatmap(corrr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2f2d60bad48>




![png](output_34_1.png)


# Imputasi 1


```python
#Melihat kemiringan dari data untuk imputasi
dfr_drop.skew(axis=0, skipna=True)
```




    pages              0.771851
    weight             0.820661
    rating_value_0    -4.962321
    rating_value_1    -1.100762
    rating_count_0    12.042760
    rating_count_1    11.135552
    dimension_0        0.373369
    dimension_1        3.690633
    dimension_2       -2.867969
    genre_0_weight     1.454005
    genre_1_weight     0.129424
    genre_2_weight     0.247111
    genre_3_weight     0.419861
    genre_4_weight     0.391143
    genre_5_weight     0.358376
    genre_6_weight     0.491903
    genre_7_weight     0.405005
    genre_8_weight     0.446851
    genre_9_weight     0.518299
    price              1.400949
    dtype: float64




```python
#imputasi numerik
dfr_drop['pages'].fillna(dfr_drop['pages'].mean(),inplace=True)
dfr_drop['weight'].fillna(dfr_drop['weight'].mean(),inplace=True)
dfr_drop['rating_value_0'].fillna(dfr_drop['rating_value_0'].median(),inplace=True)
dfr_drop['rating_value_1'].fillna(dfr_drop['rating_value_1'].mean(),inplace=True)
dfr_drop['rating_count_0'].fillna(dfr_drop['rating_count_0'].median(),inplace=True)
dfr_drop['rating_count_1'].fillna(dfr_drop['rating_count_1'].median(),inplace=True)
dfr_drop['dimension_0'].fillna(dfr_drop['dimension_0'].mean(),inplace=True)
dfr_drop['dimension_1'].fillna(dfr_drop['dimension_1'].median(),inplace=True)
dfr_drop['dimension_2'].fillna(dfr_drop['dimension_2'].median(),inplace=True)
dfr_drop['genre_0_weight'].fillna(dfr_drop['genre_0_weight'].mean(),inplace=True)
dfr_drop['genre_1_weight'].fillna(dfr_drop['genre_1_weight'].mean(),inplace=True)
dfr_drop['genre_2_weight'].fillna(dfr_drop['genre_2_weight'].mean(),inplace=True)
dfr_drop['genre_3_weight'].fillna(dfr_drop['genre_3_weight'].mean(),inplace=True)
dfr_drop['genre_4_weight'].fillna(dfr_drop['genre_4_weight'].mean(),inplace=True)
dfr_drop['genre_5_weight'].fillna(dfr_drop['genre_5_weight'].mean(),inplace=True)
dfr_drop['genre_6_weight'].fillna(dfr_drop['genre_6_weight'].mean(),inplace=True)
dfr_drop['genre_7_weight'].fillna(dfr_drop['genre_7_weight'].mean(),inplace=True)
dfr_drop['genre_8_weight'].fillna(dfr_drop['genre_8_weight'].mean(),inplace=True)
dfr_drop['genre_9_weight'].fillna(dfr_drop['genre_9_weight'].mean(),inplace=True)
```


```python
dft_drop.skew(axis=0, skipna=True)
```




    pages             0.919554
    weight            0.884496
    rating_value_0   -4.019568
    rating_value_1   -1.132931
    rating_count_0    9.648914
    rating_count_1    4.988044
    dimension_0       0.855239
    dimension_1       3.453721
    dimension_2      -3.482822
    genre_0_weight    1.427588
    genre_1_weight    0.275806
    genre_2_weight    0.348611
    genre_3_weight    0.580640
    genre_4_weight    0.613562
    genre_5_weight    0.595748
    genre_6_weight    0.395369
    genre_7_weight    0.655567
    genre_8_weight    0.685021
    genre_9_weight    0.610401
    price             4.131681
    dtype: float64




```python
dft_drop['pages'].fillna(dft_drop['pages'].mean(),inplace=True)
dft_drop['weight'].fillna(dft_drop['weight'].mean(),inplace=True)
dft_drop['rating_value_0'].fillna(dft_drop['rating_value_0'].median(),inplace=True)
dft_drop['rating_value_1'].fillna(dft_drop['rating_value_1'].mean(),inplace=True)
dft_drop['rating_count_0'].fillna(dft_drop['rating_count_0'].median(),inplace=True)
dft_drop['rating_count_1'].fillna(dft_drop['rating_count_1'].median(),inplace=True)
dft_drop['dimension_0'].fillna(dft_drop['dimension_0'].mean(),inplace=True)
dft_drop['dimension_1'].fillna(dft_drop['dimension_1'].median(),inplace=True)
dft_drop['dimension_2'].fillna(dft_drop['dimension_2'].median(),inplace=True)
dft_drop['genre_0_weight'].fillna(dft_drop['genre_0_weight'].mean(),inplace=True)
dft_drop['genre_1_weight'].fillna(dft_drop['genre_1_weight'].mean(),inplace=True)
dft_drop['genre_2_weight'].fillna(dft_drop['genre_2_weight'].mean(),inplace=True)
dft_drop['genre_3_weight'].fillna(dft_drop['genre_3_weight'].mean(),inplace=True)
dft_drop['genre_4_weight'].fillna(dft_drop['genre_4_weight'].mean(),inplace=True)
dft_drop['genre_5_weight'].fillna(dft_drop['genre_5_weight'].mean(),inplace=True)
dft_drop['genre_6_weight'].fillna(dft_drop['genre_6_weight'].mean(),inplace=True)
dft_drop['genre_7_weight'].fillna(dft_drop['genre_7_weight'].mean(),inplace=True)
dft_drop['genre_8_weight'].fillna(dft_drop['genre_8_weight'].mean(),inplace=True)
dft_drop['genre_9_weight'].fillna(dft_drop['genre_9_weight'].mean(),inplace=True)
```


```python
#imputasi string
dfr_drop['bookformat'].fillna(("Hardcover"),inplace=True)
dfr_drop['genre_0'].fillna("Historical Fiction",inplace=True)
dfr_drop['genre_1'].fillna("Fiction",inplace=True)
dfr_drop['genre_2'].fillna("Historical",inplace=True)
dfr_drop['genre_3'].fillna("Audiobook",inplace=True)
dfr_drop['genre_4'].fillna("Romance",inplace=True)
dfr_drop['genre_5'].fillna("Books About Books",inplace=True)
dfr_drop['genre_6'].fillna("Adult",inplace=True)
dfr_drop['genre_7'].fillna("Adult Fiction",inplace=True)
dfr_drop['genre_8'].fillna("British Literature",inplace=True)
dfr_drop['genre_9'].fillna("Chick Lit",inplace=True)
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\pandas\core\generic.py:6287: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._update_inplace(new_data)
    


```python
dft_drop['bookformat'].fillna(("Hardcover"),inplace=True)
dft_drop['genre_0'].fillna("Historical Fiction",inplace=True)
dft_drop['genre_1'].fillna("Fiction",inplace=True)
dft_drop['genre_2'].fillna("Historical",inplace=True)
dft_drop['genre_3'].fillna("Audiobook",inplace=True)
dft_drop['genre_4'].fillna("Romance",inplace=True)
dft_drop['genre_5'].fillna("Books About Books",inplace=True)
dft_drop['genre_6'].fillna("Adult",inplace=True)
dft_drop['genre_7'].fillna("Adult Fiction",inplace=True)
dft_drop['genre_8'].fillna("British Literature",inplace=True)
dft_drop['genre_9'].fillna("Chick Lit",inplace=True)
```


```python
dfr_drop['reading_age'].fillna("18 years and up", inplace=True)
dfr_drop['lexile_measure'].fillna("710L", inplace=True)
dfr_drop['grade_level'].fillna("7 - 9", inplace=True)
```


```python
dft_drop['reading_age'].fillna("18 years and up", inplace=True)
dft_drop['lexile_measure'].fillna("710L", inplace=True)
dft_drop['grade_level'].fillna("7 - 9", inplace=True)
```


```python
mis1 = dfr_drop.isnull().sum()
print(mis1)
```

    author_id            1
    description         29
    bookformat           0
    bookedition       3217
    pages                0
    published_date    2881
    publisher_id      2881
    reading_age          0
    lexile_measure       0
    grade_level          0
    weight               0
    rating_value_0       0
    rating_value_1       0
    rating_count_0       0
    rating_count_1       0
    dimension_0          0
    dimension_1          0
    dimension_2          0
    genre_0              0
    genre_1              0
    genre_2              0
    genre_3              0
    genre_4              0
    genre_5              0
    genre_6              0
    genre_7              0
    genre_8              0
    genre_9              0
    genre_0_weight       0
    genre_1_weight       0
    genre_2_weight       0
    genre_3_weight       0
    genre_4_weight       0
    genre_5_weight       0
    genre_6_weight       0
    genre_7_weight       0
    genre_8_weight       0
    genre_9_weight       0
    price             2905
    dtype: int64
    


```python
mis2 = dft_drop.isnull().sum()
print(mis2)
```

    author_id           0
    description         8
    bookformat          0
    bookedition       446
    pages               0
    published_date     31
    publisher_id       31
    reading_age         0
    lexile_measure      0
    grade_level         0
    weight              0
    rating_value_0      0
    rating_value_1      0
    rating_count_0      0
    rating_count_1      0
    dimension_0         0
    dimension_1         0
    dimension_2         0
    genre_0             0
    genre_1             0
    genre_2             0
    genre_3             0
    genre_4             0
    genre_5             0
    genre_6             0
    genre_7             0
    genre_8             0
    genre_9             0
    genre_0_weight      0
    genre_1_weight      0
    genre_2_weight      0
    genre_3_weight      0
    genre_4_weight      0
    genre_5_weight      0
    genre_6_weight      0
    genre_7_weight      0
    genre_8_weight      0
    genre_9_weight      0
    price               0
    dtype: int64
    

# Imputasi 2


```python
#Mengganti NA menjadi 0
dfr_0 = dfr_drop2.replace(np.nan, 0)
```

 # Encoding Variables


```python
from sklearn.preprocessing import LabelEncoder
```


```python
dfr_drop['lexile_measure']
```




    0       710L
    1       710L
    2       710L
    3       710L
    4       710L
            ... 
    3545    710L
    3546    710L
    3547    710L
    3548    710L
    3549    710L
    Name: lexile_measure, Length: 3441, dtype: object




```python
#Separating categorical and numerical columns
num_cols   = ['pages', 'weight', 'rating_value_0','rating_value_1','rating_count_0','rating_count_1','genre_0_weight','genre_1_weight','genre_2_weight','genre_3_weight','genre_4_weight','genre_5_weight','genre_6_weight','genre_7_weight','genre_8_weight','genre_9_weight','price']

#multi category columns
multi_cols = ['bookformat','genre_0','genre_1','genre_2','genre_3','genre_4','genre_5','genre_6','genre_7','genre_8','genre_9']

#ordinal
ordi_cols = ['lexile_measure','reading_age','grade_level']
```


```python
dfr_cb = dfr_drop
labelencoder = LabelEncoder()
dfr_en = labelencoder.fit_transform(data = dfr_cb, columns = multi_cols)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-33-fe2b94bbe6c2> in <module>
          1 dfr_cb = dfr_drop
          2 labelencoder = LabelEncoder()
    ----> 3 dfr_en = labelencoder.fit_transform(data = dfr_cb, columns = multi_cols)
    

    TypeError: fit_transform() got an unexpected keyword argument 'data'



```python
#Label encoding for ordinal multi category columns
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
dfr_or = dfr_drop
dfr_or[['lexile_measure']] = ord_enc.fit_transform(dfr_drop[["lexile_measure"]])
dfr_or[['reading_age']] = ord_enc.fit_transform(dfr_drop[["reading_age"]])
dfr_or[['grade_level']] = ord_enc.fit_transform(dfr_drop[["grade_level"]])
dfr_or
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\lenovo\Anaconda3\lib\site-packages\pandas\core\indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s
    C:\Users\lenovo\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    C:\Users\lenovo\Anaconda3\lib\site-packages\pandas\core\indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s
    C:\Users\lenovo\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\lenovo\Anaconda3\lib\site-packages\pandas\core\indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>author2106</td>
      <td>Just after the Second World War, in the small ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>309.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>author1018</td>
      <td>Blame it on Hawaii’s rainbows, sparkling beach...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>330.312146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>author1087</td>
      <td>The Pulitzer Prize–winning, bestselling author...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>496.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>author1295</td>
      <td>THINGS ARE ABOUT TO GET SERIOUS FOR HARRY DRES...</td>
      <td>Hardcover</td>
      <td>First Edition</td>
      <td>418.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>author2622</td>
      <td>The Romanovs were the most successful dynasty ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>784.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.090000</td>
      <td>0.080000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3545</td>
      <td>author1144</td>
      <td>How much is too much to love? Travis Maddox le...</td>
      <td>Paperback</td>
      <td>Original Edition</td>
      <td>448.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.240000</td>
      <td>0.100000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3546</td>
      <td>author2852</td>
      <td>Magneto and Professor X. Superman and Lex Luth...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>478.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.180000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3547</td>
      <td>author1309</td>
      <td>Following the launch of her #1 New York Times ...</td>
      <td>Hardcover</td>
      <td>First Edition</td>
      <td>352.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.140000</td>
      <td>0.130000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3548</td>
      <td>author1816</td>
      <td>Bachelors, beware. For those who keep secrets ...</td>
      <td>Kindle Edition</td>
      <td>NaN</td>
      <td>330.312146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3549</td>
      <td>author0882</td>
      <td>In the thrilling, nerve-wracking finale of Eze...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>315.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.090000</td>
      <td>0.090000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3441 rows × 39 columns</p>
</div>




```python
dft_or = dft_drop
dft_or[['lexile_measure']] = ord_enc.fit_transform(dft_drop[["lexile_measure"]])
dft_or[['reading_age']] = ord_enc.fit_transform(dft_drop[["reading_age"]])
dft_or[['grade_level']] = ord_enc.fit_transform(dft_drop[["grade_level"]])
dft_or
```


```python
#Label encoding for nominal multi category columns
dfr_le =pd.get_dummies(data = dfr_or,columns = multi_cols,drop_first=False)
```


```python
dfr_le
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>...</th>
      <th>genre_9_World History</th>
      <th>genre_9_World War I</th>
      <th>genre_9_World War II</th>
      <th>genre_9_Writing</th>
      <th>genre_9_Young Adult</th>
      <th>genre_9_Young Adult Contemporary</th>
      <th>genre_9_Young Adult Fantasy</th>
      <th>genre_9_Young Adult Historical Fiction</th>
      <th>genre_9_Young Adult Romance</th>
      <th>genre_9_Young Adult Science Fiction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>author2106</td>
      <td>Just after the Second World War, in the small ...</td>
      <td>NaN</td>
      <td>309.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>author1018</td>
      <td>Blame it on Hawaii’s rainbows, sparkling beach...</td>
      <td>NaN</td>
      <td>330.312146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>author1087</td>
      <td>The Pulitzer Prize–winning, bestselling author...</td>
      <td>NaN</td>
      <td>496.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>author1295</td>
      <td>THINGS ARE ABOUT TO GET SERIOUS FOR HARRY DRES...</td>
      <td>First Edition</td>
      <td>418.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>author2622</td>
      <td>The Romanovs were the most successful dynasty ...</td>
      <td>NaN</td>
      <td>784.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3545</td>
      <td>author1144</td>
      <td>How much is too much to love? Travis Maddox le...</td>
      <td>Original Edition</td>
      <td>448.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3546</td>
      <td>author2852</td>
      <td>Magneto and Professor X. Superman and Lex Luth...</td>
      <td>NaN</td>
      <td>478.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3547</td>
      <td>author1309</td>
      <td>Following the launch of her #1 New York Times ...</td>
      <td>First Edition</td>
      <td>352.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3548</td>
      <td>author1816</td>
      <td>Bachelors, beware. For those who keep secrets ...</td>
      <td>NaN</td>
      <td>330.312146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3549</td>
      <td>author0882</td>
      <td>In the thrilling, nerve-wracking finale of Eze...</td>
      <td>NaN</td>
      <td>315.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3441 rows × 2540 columns</p>
</div>




```python
dft_le =pd.get_dummies(data = dft_or,columns = multi_cols,drop_first=False)
```


```python
dft_le
```


```python
dfr_led = dfr_le.drop(labels=['author_id','description','publisher_id', 'published_date','bookedition'], axis=1)
```


```python
dft_led = dft_le.drop(labels=['author_id','description','publisher_id', 'published_date','bookedition'], axis=1)
```


```python
dfr_led.isnull().sum()
```




    pages                                     0
    reading_age                               0
    lexile_measure                            0
    grade_level                               0
    weight                                    0
                                             ..
    genre_9_Young Adult Contemporary          0
    genre_9_Young Adult Fantasy               0
    genre_9_Young Adult Historical Fiction    0
    genre_9_Young Adult Romance               0
    genre_9_Young Adult Science Fiction       0
    Length: 2535, dtype: int64



# Partition


```python
from sklearn.model_selection import train_test_split
```


```python
datalabel= dfr_led.dropna(subset=["price"], axis=0)
datalabel['price']
```




    9        98172.0
    17       57604.0
    29      103658.0
    33      649665.0
    44      247883.0
              ...   
    3518    262176.0
    3522    216411.0
    3529    152310.0
    3538    176853.0
    3541    216555.0
    Name: price, Length: 536, dtype: float64




```python
datalabel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>...</th>
      <th>genre_9_World History</th>
      <th>genre_9_World War I</th>
      <th>genre_9_World War II</th>
      <th>genre_9_Writing</th>
      <th>genre_9_Young Adult</th>
      <th>genre_9_Young Adult Contemporary</th>
      <th>genre_9_Young Adult Fantasy</th>
      <th>genre_9_Young Adult Historical Fiction</th>
      <th>genre_9_Young Adult Romance</th>
      <th>genre_9_Young Adult Science Fiction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9</td>
      <td>504.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3628.740000</td>
      <td>4.29</td>
      <td>4.600000</td>
      <td>26983</td>
      <td>504</td>
      <td>10.720000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.89</td>
      <td>4.518899</td>
      <td>27</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>29</td>
      <td>324.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3991.610000</td>
      <td>3.99</td>
      <td>4.600000</td>
      <td>43657</td>
      <td>1537</td>
      <td>13.310000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>33</td>
      <td>528.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>771.110000</td>
      <td>3.68</td>
      <td>4.300000</td>
      <td>19382</td>
      <td>1504</td>
      <td>16.510000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>44</td>
      <td>500.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>725.750000</td>
      <td>3.81</td>
      <td>3.600000</td>
      <td>32</td>
      <td>9</td>
      <td>15.240000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2535 columns</p>
</div>




```python
dataunlabel= dfr_led[dfr_led['price'].isnull()]
```


```python
dataunlabel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>...</th>
      <th>genre_9_World History</th>
      <th>genre_9_World War I</th>
      <th>genre_9_World War II</th>
      <th>genre_9_Writing</th>
      <th>genre_9_Young Adult</th>
      <th>genre_9_Young Adult Contemporary</th>
      <th>genre_9_Young Adult Fantasy</th>
      <th>genre_9_Young Adult Historical Fiction</th>
      <th>genre_9_Young Adult Romance</th>
      <th>genre_9_Young Adult Science Fiction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>309.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.76</td>
      <td>4.518899</td>
      <td>26625</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.48</td>
      <td>4.518899</td>
      <td>21</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>496.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.56</td>
      <td>4.518899</td>
      <td>59885</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>418.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.39</td>
      <td>4.518899</td>
      <td>26643</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>784.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.93</td>
      <td>4.518899</td>
      <td>11772</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2535 columns</p>
</div>




```python
##seperating dependent and independent variables 
train_X = datalabel.drop(labels='price',axis=1)
train_Y = datalabel['price']
```


```python
train1_X = dataunlabel.drop(labels='price',axis=1)
train1_Y = dataunlabel['price']
```

# Analisis 1

# Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=123)
param_grid = { 
    'n_estimators': [200,500,1000, 1500],
    'max_features': ['auto','log2'],
    'criterion' :['entropy','gini']
}
```


```python
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_X, train_Y)
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:657: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)
    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:814: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    




    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=RandomForestClassifier(bootstrap=True, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators='warn', n_jobs=None,
                                                  oob_score=False, random_state=123,
                                                  verbose=0, warm_start=False),
                 iid='warn', n_jobs=None,
                 param_grid={'criterion': ['entropy', 'gini'],
                             'max_features': ['auto', 'log2'],
                             'n_estimators': [200, 500, 1000, 1500]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
CV_rfc.best_params_
```




    {'criterion': 'gini', 'max_features': 'auto', 'n_estimators': 200}




```python
pred=CV_rfc.predict(train_X)
```


```python
RMSE = np.sqrt(np.mean(pow(pred - train_Y, 2)))
RMSE
```




    6207.430100411324



# SVR


```python
from sklearn.svm import SVR
esviem = SVR()
```


```python
grid_param1 = {
   'C': [1, 10, 100, 1000], 
   'gamma': [0.001, 0.0001], 
   'kernel': ['linear','rbf']
   }
grid_param2 = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
    }
grid_param3 = {
    'n_estimator':[10,30,50,70,100],
    'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1],
    'lambda' : [0,0.001,0.005,0.01,0.005,0.1,0.5],
    'alpha':[0,0.001,0.005,0.01,0.005,0.1,0.5]
}
```


```python
from sklearn.model_selection import GridSearchCV
gs1 = GridSearchCV(estimator=esviem,
                     param_grid=grid_param1,
                     cv=5,
                     n_jobs=-1)
gs1.fit(x_train, y_train)
best_parameters1 = gs1.best_params_
print(best_parameters1)
```


```python
svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
```


```python
mod = SVR().fit(train_X, train_Y)
print(mod)
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    

    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
        gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
        tol=0.001, verbose=False)
    


```python
yfit = mod.predict(train_X)
```


```python
RMSE2 = np.sqrt(np.mean(pow(yfit - train_Y, 2)))
RMSE2
```




    140078.4047935737



# XGB


```python
pip install xgboost
```

    Collecting xgboost
      Using cached https://files.pythonhosted.org/packages/d4/60/845dd265c7265d3bd44906d1c15be2908ff0655b32d9000935aeaeef6677/xgboost-1.4.2-py3-none-win_amd64.whl
    Requirement already satisfied: numpy in c:\users\lenovo\anaconda3\lib\site-packages (from xgboost) (1.16.5)
    Requirement already satisfied: scipy in c:\users\lenovo\anaconda3\lib\site-packages (from xgboost) (1.3.1)
    Installing collected packages: xgboost
    Successfully installed xgboost-1.4.2
    Note: you may need to restart the kernel to use updated packages.
    


```python
from xgboost import XGBClassifier

# fit model no training data
model = XGBClassifier()
model.fit(train_X,train_Y)
y_pred=model.predict(train_X)
hasilpred2 =pd.DataFrame(y_pred)
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [15:37:54] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    


```python
RMSE3 = np.sqrt(np.mean(pow(y_pred - train_Y, 2)))
RMSE3
```




    66030.8907489715




```python
from sklearn.metrics import mean_squared_error
score = mod.score(x,y)
MSE = mean_squared_error(y, yfit)
print("R-squared:", score)
print("MSE:", MSE)
```

    R-squared: -0.017640109533749948
    MSE: 19409767575.36197
    


```python
from math import sqrt
RMSE = sqrt(MSE)
print("RMSE:", RMSE)
```

    RMSE: 139318.94191157917
    

# Prediksi Unlabel


```python
unlabel_pred= CV_rfc.predict(train1_X)
hasilunlabel = pd.DataFrame(unlabel_pred)
hasilunlabel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>295814.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>72041.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>201974.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>122281.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>201974.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2900</td>
      <td>201974.0</td>
    </tr>
    <tr>
      <td>2901</td>
      <td>274159.0</td>
    </tr>
    <tr>
      <td>2902</td>
      <td>295670.0</td>
    </tr>
    <tr>
      <td>2903</td>
      <td>72041.0</td>
    </tr>
    <tr>
      <td>2904</td>
      <td>230848.0</td>
    </tr>
  </tbody>
</table>
<p>2905 rows × 1 columns</p>
</div>




```python

```

# Penggabungan


```python
df1 = datalabel
df1.reset_index(level=0, inplace=True)
df1 = df1.drop('index', axis=1)
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>...</th>
      <th>genre_9_World History</th>
      <th>genre_9_World War I</th>
      <th>genre_9_World War II</th>
      <th>genre_9_Writing</th>
      <th>genre_9_Young Adult</th>
      <th>genre_9_Young Adult Contemporary</th>
      <th>genre_9_Young Adult Fantasy</th>
      <th>genre_9_Young Adult Historical Fiction</th>
      <th>genre_9_Young Adult Romance</th>
      <th>genre_9_Young Adult Science Fiction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>504.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3628.740000</td>
      <td>4.29</td>
      <td>4.600000</td>
      <td>26983</td>
      <td>504</td>
      <td>10.720000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.89</td>
      <td>4.518899</td>
      <td>27</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>324.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3991.610000</td>
      <td>3.99</td>
      <td>4.600000</td>
      <td>43657</td>
      <td>1537</td>
      <td>13.310000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>528.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>771.110000</td>
      <td>3.68</td>
      <td>4.300000</td>
      <td>19382</td>
      <td>1504</td>
      <td>16.510000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>500.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>725.750000</td>
      <td>3.81</td>
      <td>3.600000</td>
      <td>32</td>
      <td>9</td>
      <td>15.240000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>531</td>
      <td>351.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>512.560000</td>
      <td>3.83</td>
      <td>4.200000</td>
      <td>71693</td>
      <td>5943</td>
      <td>15.880000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>532</td>
      <td>257.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>566.990000</td>
      <td>4.12</td>
      <td>4.700000</td>
      <td>14157</td>
      <td>1407</td>
      <td>16.230000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>533</td>
      <td>444.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>780.180000</td>
      <td>3.84</td>
      <td>4.400000</td>
      <td>7758</td>
      <td>240</td>
      <td>16.260000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>534</td>
      <td>64.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3265.860000</td>
      <td>4.26</td>
      <td>4.800000</td>
      <td>85767</td>
      <td>15504</td>
      <td>21.340000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>535</td>
      <td>76.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>1741.790000</td>
      <td>4.24</td>
      <td>4.800000</td>
      <td>85</td>
      <td>11</td>
      <td>14.810000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>536 rows × 2535 columns</p>
</div>




```python
df3[['price']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>295814.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>72041.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>201974.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>122281.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>201974.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2900</td>
      <td>201974.0</td>
    </tr>
    <tr>
      <td>2901</td>
      <td>274159.0</td>
    </tr>
    <tr>
      <td>2902</td>
      <td>295670.0</td>
    </tr>
    <tr>
      <td>2903</td>
      <td>72041.0</td>
    </tr>
    <tr>
      <td>2904</td>
      <td>230848.0</td>
    </tr>
  </tbody>
</table>
<p>2905 rows × 1 columns</p>
</div>




```python
df3= pd.DataFrame(dataunlabel)
df3[['price']]= hasilunlabel
df3.reset_index(level=0, inplace=True)
df3 = df3.drop(['index'], axis=1)
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>...</th>
      <th>genre_9_World History</th>
      <th>genre_9_World War I</th>
      <th>genre_9_World War II</th>
      <th>genre_9_Writing</th>
      <th>genre_9_Young Adult</th>
      <th>genre_9_Young Adult Contemporary</th>
      <th>genre_9_Young Adult Fantasy</th>
      <th>genre_9_Young Adult Historical Fiction</th>
      <th>genre_9_Young Adult Romance</th>
      <th>genre_9_Young Adult Science Fiction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>309.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.76</td>
      <td>4.518899</td>
      <td>26625</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.48</td>
      <td>4.518899</td>
      <td>21</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>496.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.56</td>
      <td>4.518899</td>
      <td>59885</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>418.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.39</td>
      <td>4.518899</td>
      <td>26643</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>784.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.93</td>
      <td>4.518899</td>
      <td>11772</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2900</td>
      <td>448.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.19</td>
      <td>4.518899</td>
      <td>172198</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2901</td>
      <td>478.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.21</td>
      <td>4.518899</td>
      <td>43149</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2902</td>
      <td>352.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.55</td>
      <td>4.518899</td>
      <td>5811</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2903</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.64</td>
      <td>4.518899</td>
      <td>14</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2904</td>
      <td>315.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.68</td>
      <td>4.518899</td>
      <td>1959</td>
      <td>1</td>
      <td>15.437287</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2905 rows × 2535 columns</p>
</div>




```python
df_2[['price']]= hasilunlabel
```


```python
df_1 = datalabel
df_2 = dataunlabel
```


```python
df_2 = df_2.drop(['index'], axis=1)
```


```python
df_1 = df_1.drop(['index'], axis=1)
```


```python
df_2['price'].tail()
```




    2900    201974.0
    2901    274159.0
    2902    295670.0
    2903     72041.0
    2904    230848.0
    Name: price, dtype: float64




```python
gabs = pd.concat([df_1,df_2])
```


```python
gabs['price']
```




    0        98172.0
    1        57604.0
    2       103658.0
    3       649665.0
    4       247883.0
              ...   
    2900    201974.0
    2901    274159.0
    2902    295670.0
    2903     72041.0
    2904    230848.0
    Name: price, Length: 3441, dtype: float64




```python
gabungan = pd.concat([df1,df3])
gabungan['price'].tail(130)
```




    2775   NaN
    2776   NaN
    2777   NaN
    2778   NaN
    2779   NaN
            ..
    2900   NaN
    2901   NaN
    2902   NaN
    2903   NaN
    2904   NaN
    Name: price, Length: 130, dtype: float64




```python
gabungan.shape
```




    (3441, 2535)




```python

```

# Analisis 2


```python
gab_X = gabs.drop(labels='price',axis=1)
gab_Y = gabs['price']
```


```python
gab_Y.isnull().sum()
```




    0




```python
rfc1=RandomForestClassifier(random_state=123)
param_grid = { 
    'n_estimators': [200,500,1000, 1500],
    'max_features': ['auto','log2'],
    'criterion' :['entropy','gini']
}
```


```python
RFC = 
```


```python
from sklearn.model_selection import GridSearchCV
CV_rfc1 = GridSearchCV(estimator=rfc1, param_grid=param_grid, cv= 2)
CV_rfc1.fit(gab_X, gab_Y)
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:657: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=2.
      % (min_groups, self.n_splits)), Warning)
    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:814: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    


```python
rfc1.fit(gab_X, gab_Y)
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=123,
                           verbose=0, warm_start=False)




```python

```


```python
CV_rfc1.best_params_
```


```python
pred2=CV_rfc1.predict(gab_X)
```


```python
pred4=rfc1.predict(gab_X)
```


```python
RMSEgab = np.sqrt(np.mean(pow(pred4 - gab_Y, 2)))
RMSEgab
```




    10630.510674083569



# Prediksi Test


```python

```


```python
predx = CV_rfc1.predict()
```
