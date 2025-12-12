# recipe-rating-inference-and-prediction
Analysis of recipe ratings using ingredient similarity, hypothesis testing, and predictive modeling. Built for UCSD’s DSC 80 course, this project explores whether French recipes are rated higher and develops models to predict recipe ratings.

This project was completed by Mark Kaplunow.

### Research Question
**Do French recipes receive higher ratings than similar non-French recipes with the same ingredients?**

This question is interesting because French cuisine has a prestigious reputation worldwide, often associated with sophistication and high quality. By comparing French recipes to non-French recipes with similar ingredients, we can determine whether this reputation translates to higher user ratings, or if ratings are primarily driven by the recipe's actual content rather than its cultural association.

### Dataset Overview
- **Number of rows:** 83,782 recipes (after merging with ratings)
- **Date range:** 2008-2018
- **Relevant columns:**
  - `name`: Recipe name (string)
  - `minutes`: Cooking time in minutes (int)
  - `tags`: Recipe tags including cuisine types (list)
  - `n_ingredients`: Number of ingredients (int)
  - `n_steps`: Number of preparation steps (int)
  - `ingredients`: List of ingredient names (list)
  - `avg_rating`: Average user rating from 1-5 (float)
  - `num_ratings`: Number of user ratings received (int)

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

I performed the following cleaning steps to prepare the dataset for analysis:

1. **Replaced ratings of 0 with NaN**: On Food.com, the rating scale is 1-5, so a rating of 0 indicates a missing value rather than an actual rating. There were **51,832 such instances** (not 15,036 as previously noted).

2. **Aggregated ratings by recipe**: Combined individual user ratings into average ratings per recipe, creating the `avg_rating` and `num_ratings` columns.

3. **Parsed string representations to lists**: The `ingredients` and `steps` columns were stored as string representations of lists. I converted them to actual Python lists using `ast.literal_eval()` for easier analysis.

4. **Created `n_steps` column**: Counted the number of steps in each recipe.

5. **Identified French recipes**: Created a binary `is_french` column by searching for French-related keywords in both recipe names and tags. Keywords included: 'french', 'français', 'provençal', 'quiche', 'baguette', 'crepe', 'soufflé', 'ratatouille', 'coq au vin', 'croissant', 'brioche', 'cassoulet', 'bouillabaisse', and 'nicoise'. This identified **1,864 French recipes** (not 1,049).

6. **Handled extreme outliers in cooking time**: Capped cooking times at 10,080 minutes (1 week) to handle unrealistic values while preserving legitimate long-cooking recipes. 66 recipes exceeded this threshold.

7. **Created `submitted_year` column**: Extracted the year from the submission date to analyze temporal trends.

These cleaning steps ensure data quality and create the features necessary for both exploratory analysis and predictive modeling. The most impactful step was replacing 0 ratings with NaN, as this prevents artificially low average ratings from skewing our analysis.

### Head of Cleaned DataFrame

| name                                 |   minutes_cleaned |   n_ingredients |   n_steps |   avg_rating |   num_ratings | is_french   |   submitted_year |
|:-------------------------------------|------------------:|----------------:|----------:|-------------:|--------------:|:------------|------------------:|
| 1 brownies in the world    best ever |                40 |               9 |        10 |            4 |             1 | False       |             2008 |
| 1 in canada chocolate chip cookies   |                45 |              11 |        12 |            5 |             1 | False       |             2011 |
| 412 broccoli casserole               |                40 |               9 |         6 |            5 |             4 | False       |             2008 |
| millionaire pound cake               |               120 |               7 |         7 |            5 |             1 | False       |             2008 |
| 2000 meatloaf                        |                90 |              13 |        17 |            5 |             2 | False       |             2012 |

**Total recipes with ratings:** 81,173 (out of 83,782 total recipes)

---

### Univariate Analysis

<iframe src="assets/fig1_rating_distribution.html" width=800 height=600 frameBorder=0></iframe>

The distribution of average ratings is heavily left-skewed, with most recipes receiving ratings between 4.0 and 5.0. The **mode is 5.0**, indicating that users tend to rate recipes they tried very positively. This suggests a potential selection bias: users may be more likely to rate recipes they enjoyed and skip rating recipes they didn't like. The **mean rating is 4.625**, which is quite high on a 1-5 scale.

<iframe src="assets/fig2_ingredients_distribution.html" width=800 height=600 frameBorder=0></iframe>

The number of ingredients follows a roughly normal distribution centered around **9 ingredients**, with most recipes containing between 5-13 ingredients. There are some outliers with up to 37 ingredients, likely representing complex dishes or compilation recipes.

---

### Bivariate Analysis

<iframe src="assets/fig3_french_vs_nonfrench.html" width=800 height=600 frameBorder=0></iframe>

This box plot compares the rating distributions of French and non-French recipes. Both groups show **similar median ratings (around 5.0)**, with French recipes showing similar variance. The similar distributions suggest that French recipes do not have substantially higher ratings overall—though this doesn't account for ingredient similarity, which is addressed in our hypothesis test.

<iframe src="assets/fig4_time_vs_rating.html" width=800 height=600 frameBorder=0></iframe>

This scatter plot shows a **weak relationship** between cooking time and rating. Most highly-rated recipes (4.5-5.0) exist across the entire spectrum of cooking times, suggesting that users value recipes equally whether they're quick or time-intensive.

---

### Interesting Aggregates

**Table 1: Recipe Characteristics by Type**

|            |   Avg Rating |   Avg Minutes |   Avg Ingredients |   Avg Steps |   Recipe Count |
|:-----------|-------------:|--------------:|------------------:|------------:|---------------:|
| Non-French |         4.62 |         83.37 |              9.21 |       10.06 |          81918 |
| French     |         4.65 |         96.21 |              9.43 |       12.20 |           1864 |

French recipes have **marginally higher average ratings** (4.65 vs 4.62), slightly more ingredients on average (9.43 vs 9.21), and **more steps** (12.20 vs 10.06). French recipes also take **longer to prepare** on average (96.21 vs 83.37 minutes).

**Table 2: Recipe Characteristics by Complexity**

| complexity        |   Avg Rating |   Recipe Count |   Avg Minutes |   Avg Steps |
|:------------------|-------------:|---------------:|--------------:|------------:|
| Simple (≤6)       |         4.64 |          21021 |         80.37 |        6.98 |
| Medium (7-10)     |         4.61 |          33071 |         74.99 |        9.58 |
| Complex (11+)     |         4.63 |          27081 |         96.76 |       13.17 |

Interestingly, **simpler recipes (≤6 ingredients) have slightly higher average ratings** than medium-complexity recipes. This could indicate that users prefer straightforward recipes, or that simpler recipes have less room for error in execution. As expected, recipe complexity correlates strongly with both cooking time and number of steps.

---

## Assessment of Missingness

### NMAR Analysis

I believe the **`description`** column is likely **NMAR (Not Missing At Random)**.

The description column has **70 missing values**. This missingness is likely NMAR because the decision to leave a description blank depends on the description itself—specifically, its perceived necessity. Recipe contributors may not provide descriptions for recipes they consider self-explanatory or very simple.

The key insight is that the missingness mechanism is directly related to the *unobserved value* of the description: recipes with missing descriptions likely would have had simple or redundant descriptions if they were filled in.

**Additional data that could explain this missingness (making it MAR):**
- **User engagement metrics**: Number of recipes posted by each user, time spent on Food.com
- **Recipe complexity score**: An external measure of recipe complexity
- **User demographic data**: Contributor experience level or account age

---

### Missingness Dependency

I analyzed whether the missingness of **`avg_rating`** depends on other columns. Recipes have missing average ratings when they received no user ratings at all (2,609 recipes).

**Test 1: Does avg_rating missingness depend on n_ingredients?**

<iframe src="assets/fig6_missingness_ingredients.html" width=800 height=600 frameBorder=0></iframe>

- **Observed difference in mean ingredients:** -0.2542
  - Recipes WITH ratings: 9.21 ingredients
  - Recipes WITHOUT ratings: 9.46 ingredients
- **P-value:** 0.0000

With a p-value of 0.0000, we reject the null hypothesis. The missingness of `avg_rating` **does depend** on `n_ingredients`, making it **MAR (Missing At Random)**. Recipes with missing ratings have **slightly more ingredients** on average.

---

**Test 2: Does avg_rating missingness depend on submitted_year?**

<iframe src="assets/fig7_missingness_year.html" width=800 height=600 frameBorder=0></iframe>

- **Observed difference in mean year:** -0.7297
  - Recipes WITH ratings: avg year 2009.4
  - Recipes WITHOUT ratings: avg year 2010.2
- **P-value:** 0.0000

With a p-value of 0.0000, we reject the null hypothesis. The missingness of `avg_rating` **does depend** on `submitted_year`, making it **MAR**. Recipes submitted in **later years (2010+) are more likely to have missing ratings**, which makes sense: newer recipes have had less time to accumulate user ratings.

---

## Hypothesis Testing

**Research Question:** Are French recipes rated higher than similar non-French recipes?

**Null Hypothesis (H₀):** French recipes and similar non-French recipes (with matching ingredients) have the same average rating. Any observed difference is due to random chance.

**Alternative Hypothesis (H₁):** French recipes have a higher average rating than similar non-French recipes with the same ingredients.

**Test Statistic:** Difference in mean ratings (French - similar non-French)

**Significance Level:** α = 0.05

**Methodology:** 
1. Identify French recipes using keywords in names and tags
2. For each non-French recipe, compute Jaccard similarity with all French recipes based on ingredients
3. Keep only non-French recipes with similarity ≥ 0.5 to at least one French recipe
4. Compare average ratings using a permutation test with 10,000 iterations

<iframe src="assets/fig8_hypothesis_test.html" width=800 height=600 frameBorder=0></iframe>

**Results:**
- French recipes mean rating: 4.6488
- Similar non-French recipes mean rating: 4.5860
- **Observed difference: 0.0628**
- **P-value: 0.0008**

**Conclusion:** With a p-value of 0.0008, we **reject the null hypothesis** at the α = 0.05 significance level. 

The data suggests that **French recipes ARE rated higher than similar non-French recipes** when controlling for ingredients. This indicates that the prestigious reputation of French cuisine may indeed translate to higher user ratings, even when comparing recipes with similar ingredients.

---

## Framing a Prediction Problem

**Prediction Problem:** Predict the average rating a recipe will receive

**Type:** Regression (predicting a continuous value from 1-5)

**Response Variable:** `avg_rating`
- This represents the average of all user ratings for a recipe
- Chosen because it's a key metric of recipe quality and user satisfaction
- More stable than individual ratings since it aggregates multiple user opinions

**Evaluation Metric:** Root Mean Squared Error (RMSE)
- Chosen over MAE because RMSE penalizes large errors more heavily
- Since ratings are on a 1-5 scale, we want to avoid predictions that are way off
- RMSE is in the same units as our target (rating points)
- Also reporting R² to understand proportion of variance explained

**Information Available at Time of Prediction:**

At the time a recipe is submitted to Food.com, we would know:
- Recipe characteristics: ingredients, steps, cooking time, nutrition
- Recipe metadata: submission date, contributor info
- Recipe tags and categories

We would **NOT** know:
- Future ratings (`avg_rating`, `num_ratings`)
- Future user reviews
- Time-dependent features that occur after submission

Therefore, I only use features that would be available at recipe submission time, ensuring our model could realistically be deployed to predict ratings for new recipes.

---

## Baseline Model

**Model:** Linear Regression

**Features (4 total):**

1. **`minutes_cleaned`** (Quantitative): Cooking time in minutes
2. **`n_ingredients`** (Quantitative): Number of ingredients in recipe
3. **`n_steps`** (Quantitative): Number of preparation steps
4. **`is_french`** (Nominal - Binary): Whether recipe is French (encoded as 0/1)

**Feature Engineering:**
- Quantitative features: Standardized using `StandardScaler` to normalize across different scales
- Nominal feature: `is_french` converted from boolean to integer (0/1)

**Performance:**
- **Training RMSE:** 0.6419
- **Test RMSE:** 0.6361
- **Training R²:** 0.0002
- **Test R²:** -0.0007

**Is this model "good"?**

The baseline model achieves a test RMSE of 0.6361, meaning our predictions are off by about 0.64 rating points on average. Given that:
- The rating scale is 1-5 (range of 4 points)
- The standard deviation of ratings is 0.6408
- A naive prediction of the mean would give RMSE = 0.6408

This baseline model is **marginally better than simply predicting the mean** rating for all recipes, but there is **substantial room for improvement**. The R² of -0.0007 indicates we're explaining **very little variance** in ratings.

The model is **NOT particularly good**, but it establishes a reasonable baseline for comparison with our final model.

---

## Final Model

### Feature Engineering

I added **6 new features** beyond the baseline, each designed to capture aspects of recipes that might influence user ratings:

1. **`log_minutes`** (Quantitative transformation)
   - Log-transform of cooking time to handle the right-skewed distribution
   - **Why it's good:** The difference between 10 and 20 minutes is more perceptually significant than between 110 and 120 minutes

2. **`ingredients_per_step`** (Quantitative ratio)
   - Ratio of ingredients to steps
   - **Why it's good:** Captures recipe workflow complexity/density

3. **`has_many_ingredients`** (Nominal binary)
   - 1 if recipe has > 12 ingredients, 0 otherwise
   - **Why it's good:** Based on 75th percentile; very ingredient-heavy recipes may be perceived differently

4. **`is_quick_recipe`** (Nominal binary)
   - 1 if cooking time < 30 minutes, 0 otherwise
   - **Why it's good:** Quick recipes appeal to users seeking convenience

5. **`complexity_score`** (Quantitative engineered)
   - Formula: (n_ingredients × n_steps) / log(minutes + 1)
   - **Why it's good:** Combines multiple complexity dimensions into single score

6. **`submitted_year`** (Quantitative)
   - Year of recipe submission
   - **Why it's good:** User rating behavior may have changed over time on the platform

These features improve the model by capturing **non-linear relationships**, **interaction effects**, **threshold effects**, and **temporal trends** that a simple linear model would miss.

---

### Model Selection and Hyperparameter Tuning

**Model:** Random Forest Regressor
- Chosen over Linear Regression for ability to capture non-linear relationships
- Can handle feature interactions automatically
- More robust to outliers

**Hyperparameter Search:**

I used **GridSearchCV with 5-fold cross-validation** to tune:
- **`n_estimators`** [50, 100, 200]: Number of trees
- **`max_depth`** [None, 10, 20]: Maximum tree depth
- **`min_samples_split`** [2, 5, 10]: Minimum samples to split
- **`min_samples_leaf`** [1, 2, 4]: Minimum samples at leaf

**Total combinations tested:** 81

**Best Hyperparameters Found:**
- `n_estimators`: 200
- `max_depth`: 10
- `min_samples_split`: 10
- `min_samples_leaf`: 4

---

### Final Model Performance

**Performance Metrics:**
- **Training RMSE:** 0.6273
- **Test RMSE:** 0.6364
- **Training R²:** 0.0453
- **Test R²:** -0.0016

**Top 5 Most Important Features:**
1. `complexity_score` (0.3151)
2. `ingredients_per_step` (0.1945)
3. `log_minutes` (0.1132)
4. `minutes_cleaned` (0.1124)
5. `submitted_year` (0.1070)

<iframe src="assets/fig_predictions.html" width=800 height=600 frameBorder=0></iframe>

<iframe src="assets/fig_importance.html" width=800 height=600 frameBorder=0></iframe>

---

### Improvement Over Baseline

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| Test RMSE | 0.6361 | 0.6364 | -0.0003 (-0.05% reduction) |
| Test R² | -0.0007 | -0.0016 | -0.0009 |

**Important Note:** The final model shows **minimal improvement** over the baseline in terms of RMSE, and actually performs slightly worse in terms of R². This suggests that:
1. The additional features didn't substantially improve predictive power
2. Random Forest's ability to capture non-linear relationships didn't yield better predictions in this case
3. Recipe ratings may be influenced by factors not captured in our feature set

The **complexity_score** emerged as the most important feature, indicating that recipe complexity plays a role in how users rate recipes.

---

## Fairness Analysis

**Fairness Question:** Does our model perform fairly between French and Non-French recipes?

**Group X:** French recipes  
**Group Y:** Non-French recipes

**Evaluation Metric:** RMSE (Root Mean Squared Error)
- Measures average prediction error for each group
- Lower RMSE = better performance

**Null Hypothesis (H₀):** Our model is fair. The RMSE for French recipes and non-French recipes are roughly the same, and any observed difference is due to random chance.

**Alternative Hypothesis (H₁):** Our model is unfair. There is a significant difference in RMSE between French and non-French recipes.

**Test Statistic:** Absolute difference in RMSE between groups  
|RMSE_French - RMSE_Non-French|

**Significance Level:** α = 0.05

---

### Results

**Observed RMSE:**
- French recipes: 0.7170
- Non-French recipes: 0.6343
- **Absolute difference: 0.0826**

**Permutation Test:** 10,000 iterations  
**P-value:** 0.1236

<iframe src="assets/fig_fairness.html" width=800 height=600 frameBorder=0></iframe>

<iframe src="assets/fig_residuals.html" width=800 height=600 frameBorder=0></iframe>

---

### Conclusion

With a p-value of 0.1236, we **fail to reject the null hypothesis** at the α = 0.05 significance level.

**Conclusion:** Our model appears to perform **fairly** between French and non-French recipes. The difference in RMSE between these groups (0.0826) is not statistically significant and could be due to random chance.

This suggests that the model treats both types of recipes similarly and does not systematically favor one group over the other in terms of prediction accuracy. While French recipes have higher average error (0.7170 vs 0.6343), this difference is within the range we'd expect from random variation given the sample sizes (376 French vs 15,859 non-French recipes).

The fairness of our model is reassuring, especially given that French recipes are a minority class (only 2.2% of the dataset) and might have different rating patterns due to cultural factors.
