# **Physics-Informed Surrogate Modeling for Large-Scale Wave Energy Farms**

High-fidelity hydrodynamic simulations are essential for evaluating wave energy converter (WEC) array layouts, but they are extremely costly to run—especially as the number of devices grows. The dataset used in this project contains simulation-derived absorbed power values for 49- and 100-device layouts in both Perth and Sydney wave climates.

This project uses that dataset to train machine-learning surrogate models that can predict absorbed power directly from the spatial coordinates of the array, without needing to rerun any simulations. By comparing baseline, advanced, and physics-informed models, the study identifies which approaches most accurately capture array interactions and provide a fast, reliable surrogate for layout evaluation and design-space exploration.

# ⭐ **Results Summary**

### **49-WEC Results**

| Region    | Model    | RMSE       | MAE         | R²       | Relative MAE (%) |
| --------- | -------- | ---------- | ---------   | -------- | -----------------|
| Perth_49  | LightGBM | 20,489.887 | 8,907.0336  | 0.971514 | 0.2262           |
| Sydney_49 | LightGBM | 3,930.0643 | 1,670.0844  | 0.996994 | 0.0414           |

### **100-WEC Results**

| Region     | Model    | RMSE       | MAE        | R²       | Relative MAE (%) |
| ---------- | -------- | ---------- | ---------- | -------- | -----------------|
| Perth_100  | LightGBM | 36,563.366 | 13,461.323 | 0.963700 | 0.1894           |
| Sydney_100 | LightGBM | 14,271.181 | 6,180.880  | 0.979158 | 0.0862           |

# **Dataset Citation**

> [Large-scale Wave Energy Farm - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm)

# **Introductory Paper**

> Neshat, Mehdi, Bradley Alexander, Nataliia Y. Sergiienko, and Markus Wagner. "Optimisation of large wave farms using a multi-strategy evolutionary framework." In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, pp. 1150-1158. 2020.
> [https://dl.acm.org/doi/10.1145/3377930.3390235](https://dl.acm.org/doi/10.1145/3377930.3390235)

# **Dataset Overview**

The **Wave Energy Converter (WEC) Array datasets** contain thousands of **simulated wave energy farm layouts**, each consisting of either **49** or **100** point-absorber devices arranged in a 2D plane.
For every layout, the dataset provides:

* The **(X, Y)** coordinates of every WEC in the array
* The **individual device absorbed power values** (`P1 … PN`)
* The **total absorbed power** (`Total_Power`), computed via a high-fidelity hydrodynamic simulation. It `Total_Power` represents the sum of absorbed power across all devices (`P1 + P2 + … + PN`).

The objective is to build a **surrogate model** that maps layout geometry to total absorbed power.

### **Dataset Size**

| Array Size | Region | #Samples |
| ---------- | ------ | -------- |
| 49 WECs    | Perth  | 36,043   |
| 49 WECs    | Sydney | 17,964   |
| 100 WECs   | Perth  | 7,277    |
| 100 WECs   | Sydney | 2,318    |

## **Perth vs. Sydney Wave Climates**

The paper highlights a critical physical distinction between the two environments:
**Perth and Sydney have fundamentally different wave directional spectra**, which strongly affects power absorption and the difficulty of the learning task.

These differences explain why identical ML models perform differently across regions.


### **Perth — Narrow Directional Spectrum**

> *“Perth has a small sector from which the prevailing waves arrive… For Perth, this can result in very pronounced constructive and destructive interference.”*

This leads to **strong WEC–WEC interactions**, sharp interference patterns, and a much more irregular response surface.
As a result, the Perth dataset is **more nonlinear** and consistently **more challenging** for surrogate modeling.


### **Sydney — Broad Directional Spectrum**

> *“Sydney’s wave directions vary much more… the same interference patterns are smeared out for Sydney.”*

Here, interference effects are diffused across the array, creating a **smoother, more predictable** power landscape.
This makes Sydney inherently **easier** for ML models to approximate accurately.

While Sydney is easier at 49-WEC, the **100-WEC Sydney dataset becomes significantly harder** due to:

* The increase from **98 → 200 coordinate features**
* A **sharp reduction in dataset size** (17,964 → 2,318 samples)
* Fewer samples than Perth’s 100-WEC dataset

This combination increases model variance and reduces generalization strength, making context essential when interpreting results and benchmarking performance.

# **Model Overview**

This project evaluates three broad families of models to predict **total absorbed power** from WEC array geometries. All models use only the raw layout coordinates; no hydrodynamic parameters are included.

## **1. Baseline Models**

These models provide a simple point of comparison and help establish how much of the problem can be solved without explicitly modeling physics. They capture only basic geometric trends and serve as lower-bound references for performance on both 49-WEC and 100-WEC datasets:

* **Linear Regression**
* **Ridge Regression**
* **Random Forest Regressor**

### **Results of Baseline Models**

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Perth_49</td>
      <td>RandomForest</td>
      <td>26910.829981</td>
      <td>10241.562399</td>
      <td>0.950863</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Perth_49</td>
      <td>LinearRegression</td>
      <td>49726.741388</td>
      <td>36706.651082</td>
      <td>0.832223</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perth_49</td>
      <td>RidgeRegression</td>
      <td>49726.741459</td>
      <td>36706.651038</td>
      <td>0.832223</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sydney_49</td>
      <td>RandomForest</td>
      <td>9778.648110</td>
      <td>3037.890742</td>
      <td>0.981390</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sydney_49</td>
      <td>RidgeRegression</td>
      <td>27610.806670</td>
      <td>16283.027167</td>
      <td>0.851628</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sydney_49</td>
      <td>LinearRegression</td>
      <td>27610.808255</td>
      <td>16283.027647</td>
      <td>0.851628</td>
    </tr>
  </tbody>
</table>
</div>

<br>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Perth_100</td>
      <td>RandomForest</td>
      <td>59475.049089</td>
      <td>24409.640709</td>
      <td>0.903953</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Perth_100</td>
      <td>RidgeRegression</td>
      <td>72857.502604</td>
      <td>50468.684021</td>
      <td>0.855868</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perth_100</td>
      <td>LinearRegression</td>
      <td>72857.503997</td>
      <td>50468.684471</td>
      <td>0.855868</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sydney_100</td>
      <td>RandomForest</td>
      <td>40363.452009</td>
      <td>15459.772873</td>
      <td>0.833280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sydney_100</td>
      <td>RidgeRegression</td>
      <td>99218.071193</td>
      <td>38465.725584</td>
      <td>-0.007377</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sydney_100</td>
      <td>LinearRegression</td>
      <td>99218.317347</td>
      <td>38465.795317</td>
      <td>-0.007382</td>
    </tr>
  </tbody>
</table>
</div>

## **2. Advanced Models**

To better capture nonlinear hydrodynamic interactions, we evaluate several modern machine learning models:

* **LightGBM**
* **XGBoost**
* **CatBoost**
* **MLPRegressor**

### **Results of Advanced Models**

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Perth_49</td>
      <td>CatBoost</td>
      <td>21296.611871</td>
      <td>9985.962019</td>
      <td>0.969227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Perth_49</td>
      <td>LightGBM</td>
      <td>21687.056442</td>
      <td>10405.109141</td>
      <td>0.968088</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perth_49</td>
      <td>XGBoost</td>
      <td>22242.270077</td>
      <td>9753.658921</td>
      <td>0.966433</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Perth_49</td>
      <td>MLPRegressor</td>
      <td>34699.687705</td>
      <td>20585.273158</td>
      <td>0.918303</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sydney_49</td>
      <td>LightGBM</td>
      <td>7256.901454</td>
      <td>3315.946978</td>
      <td>0.989751</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sydney_49</td>
      <td>XGBoost</td>
      <td>7447.537671</td>
      <td>2696.740874</td>
      <td>0.989205</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sydney_49</td>
      <td>CatBoost</td>
      <td>7597.399784</td>
      <td>3282.706560</td>
      <td>0.988766</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sydney_49</td>
      <td>MLPRegressor</td>
      <td>38055.530855</td>
      <td>17986.544624</td>
      <td>0.718143</td>
    </tr>
  </tbody>
</table>
</div>

<br>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Perth_100</td>
      <td>CatBoost</td>
      <td>45059.240907</td>
      <td>18730.207497</td>
      <td>0.944871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Perth_100</td>
      <td>LightGBM</td>
      <td>45091.410715</td>
      <td>18225.169907</td>
      <td>0.944792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perth_100</td>
      <td>XGBoost</td>
      <td>45898.013182</td>
      <td>17480.009691</td>
      <td>0.942800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Perth_100</td>
      <td>MLPRegressor</td>
      <td>102956.674527</td>
      <td>78496.603760</td>
      <td>0.712180</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sydney_100</td>
      <td>CatBoost</td>
      <td>31279.413702</td>
      <td>13860.880314</td>
      <td>0.899878</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sydney_100</td>
      <td>XGBoost</td>
      <td>35364.596923</td>
      <td>13673.702414</td>
      <td>0.872018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sydney_100</td>
      <td>LightGBM</td>
      <td>36880.730521</td>
      <td>13411.594325</td>
      <td>0.860810</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sydney_100</td>
      <td>MLPRegressor</td>
      <td>153192.383531</td>
      <td>73974.382329</td>
      <td>-1.401512</td>
    </tr>
  </tbody>
</table>
</div>

These models significantly outperform the baselines. CatBoost achieves the strongest raw predictive performance in most cases (lower RMSE/MAE and higher R²), but its training time scales poorly with both feature count and dataset size. LightGBM delivers very similar predictive quality while training much faster and using less memory, making it the most practical and well-balanced choice across all four datasets.

## **3. Physics-Informed Surrogate Models**

The absorbed power in a WEC array depends mainly on **how devices are spaced, oriented, and arranged** relative to each other.
To capture these interactions using only the `(X, Y)` coordinates, we add a small set of physics-motivated features that are efficient to compute and easy for ML models to learn from.

### **Pairwise Distances & Angles**

Distances and relative angles between every pair of WECs. These describe the basic interaction geometry that drives constructive and destructive wave interference.

### **Dominant Layout Direction (PCA)**

A simple PCA fit on the training layouts to estimate the overall orientation of the array. Used later for computing directional features.

### **Alignment Features**

Each pairwise vector is projected onto the dominant layout direction. This adds a crude approximation of how devices align with incoming wave energy.

### **Spatial Statistics**

Global descriptors such as minimum spacing, mean spacing, and spread of distances. These summarize how compact or dispersed a layout is.

### **Convex Hull Area**

A single feature capturing the overall footprint of the array.

### **Why These Features Are Useful**

These physics-informed features encode the **spatial structure**, **relative spacing**, and **interaction-relevant geometry** of the WEC array—factors that strongly influence absorbed power. They provide the model with physically meaningful relationships that are difficult to infer from raw (x, y) coordinates alone.

Although **CatBoost** achieved the strongest raw predictive performance in the earlier model sweep, its training time scaled poorly as feature dimensionality increased. **LightGBM** offered nearly identical predictive quality while being significantly faster and more memory-efficient, especially when introducing the larger physics-informed feature set.

For this reason, the physics-informed feature set was trained using LightGBM, which serves as the final surrogate model for both the 49-WEC and 100-WEC datasets.

### **Results of Physics-Informed Models**

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Perth_49</td>
      <td>LightGBM</td>
      <td>20489.887347</td>
      <td>8907.033611</td>
      <td>0.971514</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sydney_49</td>
      <td>LightGBM</td>
      <td>3930.064386</td>
      <td>1670.084483</td>
      <td>0.996994</td>
    </tr>
  </tbody>
</table>
</div>

<br>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Perth_100</td>
      <td>LightGBM</td>
      <td>36563.366481</td>
      <td>13461.323088</td>
      <td>0.963700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sydney_100</td>
      <td>LightGBM</td>
      <td>14271.181253</td>
      <td>6180.879836</td>
      <td>0.979158</td>
    </tr>
  </tbody>
</table>
</div>

# **How to Use This Repository**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

All experiments were run using the exact versions specified in `requirements.txt` to ensure full reproducibility.

### **2. Run Any Model Notebook**

Each notebook corresponds to a specific modeling stage applied to either the **49-WEC** or **100-WEC** dataset:

```
01_49_total_baseline.ipynb
02_49_total_advanced.ipynb
03_49_total_physics.ipynb
04_100_total_baseline.ipynb
05_100_total_advanced.ipynb
06_100_total_physics.ipynb
```

* **“49”** indicates experiments using the 49-device datasets (Perth_49 / Sydney_49)
* **“100”** indicates experiments using the 100-device datasets (Perth_100 / Sydney_100)
* **baseline** refers to linear models and Random Forest
* **advanced** refers to LightGBM, XGBoost, CatBoost, and MLP
* **physics** refers to models augmented with physics-informed features

Every notebook uses the **same train/validation/test split (`random_state=42`)** within its dataset group, making all results within the 49-WEC and 100-WEC setups directly comparable.