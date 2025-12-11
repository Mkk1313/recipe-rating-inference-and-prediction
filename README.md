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

1. **Replaced ratings of 0 with NaN**: On Food.com, the rating scale is 1-5, so a rating of 0 indicates a missing value rather than an actual rating. There were 15,036 such instances.

2. **Aggregated ratings by recipe**: Combined individual user ratings into average ratings per recipe, creating the `avg_rating` and `num_ratings` columns.

3. **Parsed string representations to lists**: The `ingredients` and `steps` columns were stored as string representations of lists. I converted them to actual Python lists using `ast.literal_eval()` for easier analysis.

4. **Created `n_steps` column**: Counted the number of steps in each recipe.

5. **Identified French recipes**: Created a binary `is_french` column by searching for French-related keywords in both recipe names and tags. Keywords included: 'french', 'français', 'provençal', 'quiche', 'baguette', 'crepe', 'soufflé', 'ratatouille', 'coq au vin', 'croissant', 'brioche', 'cassoulet', 'bouillabaisse', and 'nicoise'. This identified 1,049 French recipes.

6. **Handled extreme outliers in cooking time**: Capped cooking times at 10,080 minutes (1 week) to handle unrealistic values while preserving legitimate long-cooking recipes.

7. **Created `submitted_year` column**: Extracted the year from the submission date to analyze temporal trends.

These cleaning steps ensure data quality and create the features necessary for both exploratory analysis and predictive modeling. The most impactful step was replacing 0 ratings with NaN, as this prevents artificially low average ratings from skewing our analysis.

### Head of Cleaned DataFrame

| name                                 |   minutes_cleaned |   n_ingredients |   n_steps |   avg_rating |   num_ratings | is_french   |   submitted_year |
|:-------------------------------------|------------------:|----------------:|----------:|-------------:|--------------:|:------------|------------------:|
| 1 brownies in the world    best ever |                40 |               9 |        10 |            4 |             1 | False       |             2008 |
| 1 in canada chocolate chip cookies   |                45 |              11 |        12 |            5 |             1 | False       |             2011 |
| 412 broccoli casserole               |                40 |               9 |         6 |            5 |             3 | False       |             2008 |
| millionaire pound cake               |               120 |               7 |         7 |            5 |             1 | False       |             2008 |
| 2000 meatloaf                        |                90 |              13 |        17 |            5 |             1 | False       |             2012 |

**Total recipes with ratings:** 83,782

---

### Univariate Analysis

<iframe src="assets/fig1_rating_distribution.html" width=800 height=600 frameBorder=0></iframe>

The distribution of average ratings is heavily left-skewed, with most recipes receiving ratings between 4.0 and 5.0. The mode is approximately 5.0, indicating that users tend to rate recipes they tried very positively. This suggests a potential selection bias: users may be more likely to rate recipes they enjoyed and skip rating recipes they didn't like. The mean rating is 4.63, which is quite high on a 1-5 scale.

<iframe src="assets/fig2_ingredients_distribution.html" width=800 height=600 frameBorder=0></iframe>

The number of ingredients follows a roughly normal distribution with a slight right skew, centered around 9 ingredients. Most recipes contain between 5 and 13 ingredients. There are some outliers with 30+ ingredients, likely representing complex dishes or compilation recipes.

---

### Bivariate Analysis

<iframe src="assets/fig3_french_vs_nonfrench.html" width=800 height=600 frameBorder=0></iframe>

This box plot compares the rating distributions of French and non-French recipes. Interestingly, both groups show similar median ratings (around 4.75), with French recipes showing slightly less variance. However, the similar distributions suggest that French recipes do not have substantially higher ratings overall—though this doesn't account for ingredient similarity, which is addressed in our hypothesis test.

<iframe src="assets/fig4_time_vs_rating.html" width=800 height=600 frameBorder=0></iframe>

This scatter plot shows a weak relationship between cooking time and rating. Most highly-rated recipes (4.5-5.0) exist across the entire spectrum of cooking times, suggesting that users value recipes equally whether they're quick or time-intensive. There's a slight concentration of highly-rated recipes in the 30-60 minute range, possibly representing a "sweet spot" for home cooking.

---

### Interesting Aggregates

**Table 1: Recipe Characteristics by Type**

|            |   Avg Rating |   Avg Minutes |   Avg Ingredients |   Avg Steps |   Recipe Count |
|:-----------|-------------:|--------------:|------------------:|------------:|---------------:|
| Non-French |         4.63 |        110.55 |              9.05 |        9.86 |          82733 |
| French     |         4.64 |        103.92 |              9.52 |       10.12 |           1049 |

French recipes have marginally higher average ratings (4.64 vs 4.63), slightly more ingredients on average (9.52 vs 9.05), and slightly more steps (10.12 vs 9.86). However, these differences are quite small, and French recipes actually take slightly less time on average (103.92 vs 110.55 minutes). The small sample size of French recipes (1,049) compared to non-French (82,733) is notable.

**Table 2: Recipe Characteristics by Complexity**

| complexity        |   Avg Rating |   Recipe Count |   Avg Minutes |   Avg Steps |
|:------------------|-------------:|---------------:|--------------:|------------:|
| Simple (≤6)       |         4.64 |          23086 |         83.24 |        8.25 |
| Medium (7-10)     |         4.63 |          37341 |        110.17 |        9.44 |
| Complex (11+)     |         4.62 |          23355 |        136.39 |       12.17 |

Interestingly, simpler recipes (6 or fewer ingredients) have slightly higher average ratings than complex recipes. This could indicate that users prefer straightforward recipes, or that simpler recipes have less room for error in execution. As expected, recipe complexity correlates strongly with both cooking time and number of steps.

---

## Assessment of Missingness

### NMAR Analysis

I believe the **`description`** column is likely **NMAR (Not Missing At Random)**.

The description column has 114 missing values. This missingness is likely NMAR because the decision to leave a description blank depends on the description itself—specifically, its perceived necessity. Recipe contributors may not provide descriptions for recipes they consider self-explanatory or very simple (e.g., "toast" or "scrambled eggs"). Similarly, contributors might skip descriptions when they feel the recipe name is sufficiently descriptive (e.g., "Classic Chocolate Chip Cookies").

The key insight is that the missingness mechanism is directly related to the *unobserved value* of the description: recipes with missing descriptions likely would have had simple or redundant descriptions if they were filled in.

**Additional data that could explain this missingness (making it MAR):**
- **User engagement metrics**: Number of recipes posted by each user, time spent on Food.com (more engaged users might consistently write descriptions)
- **Recipe complexity score**: An external measure of recipe complexity could reveal whether simpler recipes systematically have more missing descriptions
- **User demographic data**: Contributor experience level or account age might predict description completeness

---

### Missingness Dependency

I analyzed whether the missingness of **`avg_rating`** depends on other columns. Recipes have missing average ratings when they received no user ratings at all.

**Test 1: Does avg_rating missingness depend on n_ingredients?**

<iframe src="assets/fig6_missingness_ingredients.html" width=800 height=600 frameBorder=0></iframe>

- **Observed difference in mean ingredients:** -0.5423
- **P-value:** 0.000

With a p-value of 0.000, we reject the null hypothesis. The missingness of `avg_rating` **does depend** on `n_ingredients`, making it **MAR (Missing At Random)**. Recipes with missing ratings have significantly fewer ingredients on average, suggesting that simpler recipes may be less likely to receive user ratings—possibly because users view them as too basic to warrant feedback.

---

**Test 2: Does avg_rating missingness depend on submitted_year?**

<iframe src="assets/fig7_missingness_year.html" width=800 height=600 frameBorder=0></iframe>

- **Observed difference in mean year:** -1.7688
- **P-value:** 0.000

With a p-value of 0.000, we reject the null hypothesis. The missingness of `avg_rating` **does depend** on `submitted_year`, making it **MAR**. Recipes submitted in later years are more likely to have missing ratings, which makes sense: newer recipes have had less time to accumulate user ratings compared to older recipes.

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
- French recipes mean rating: 4.6407
- Similar non-French recipes mean rating: 4.6325
- Observed difference: 0.0082
- **P-value: 0.5432**

**Conclusion:** With a p-value of 0.5432, we **fail to reject the null hypothesis** at the α = 0.05 significance level. The data does not provide sufficient evidence that French recipes are rated higher than similar non-French recipes when controlling for ingredients. 

This suggests that the prestige associated with French cuisine does not translate to higher user ratings on Food.com. When recipes have similar ingredients, users rate them similarly regardless of whether they're labeled as "French." This implies that recipe ratings are primarily driven by the actual ingredients and execution rather than cultural associations.

---

## Framing a Prediction Problem

**Prediction Problem:** Predict the average rating a recipe will receive

**Type:** Regression (predicting a continuous value from 1-5)

**Response Variable:** `avg_rating`
- This represents the average of all user ratings for a recipe
- I chose this because it's a key metric of recipe quality and user satisfaction
- It's more stable than individual ratings since it aggregates multiple user opinions

**Evaluation Metric:** Root Mean Squared Error (RMSE)
- Chosen over MAE because RMSE penalizes large errors more heavily
- Since ratings are on a 1-5 scale, we want to avoid predictions that are way off
- RMSE is in the same units as our target (rating points)
- Also reporting R² to understand the proportion of variance explained

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
- **Training RMSE:** 0.6256
- **Test RMSE:** 0.6260
- **Training R²:** 0.0107
- **Test R²:** 0.0105

**Is this model "good"?**

No, this baseline model is **not particularly good**. The test RMSE of 0.6260 means our predictions are off by about 0.63 rating points on average. Given that:
- The rating scale is 1-5 (a range of 4 points)
- The standard deviation of ratings is 0.6304
- A naive model that always predicts the mean would achieve RMSE ≈ 0.63

Our baseline model barely outperforms the naive mean prediction. The R² of 0.0105 indicates we're explaining only about 1% of the variance in ratings, which is very low. 

However, this establishes a reasonable baseline for comparison. The poor performance suggests that recipe ratings are influenced by factors beyond simple characteristics like cooking time, ingredient count, and cuisine type. This motivates the need for more sophisticated features and modeling approaches in our final model.

---

## Final Model

### Feature Engineering

I added **6 new features** beyond the baseline, each designed to capture aspects of recipes that might influence user ratings:

1. **`log_minutes`** (Quantitative transformation)
   - Log-transform of cooking time to handle the right-skewed distribution
   - **Why it's good:** The difference between 10 and 20 minutes is more perceptually significant than between 110 and 120 minutes. Log transformation captures this non-linear relationship between time and perceived complexity/convenience.

2. **`ingredients_per_step`** (Quantitative ratio)
   - Ratio of ingredients to steps (ingredients / steps)
   - **Why it's good:** This measures recipe density. A recipe with 12 ingredients but only 3 steps might be simpler to execute than one with 6 ingredients and 12 steps. This ratio captures recipe workflow complexity.

3. **`has_many_ingredients`** (Nominal binary)
   - 1 if recipe has more than 12 ingredients, 0 otherwise
   - **Why it's good:** Based on data exploration, 12 ingredients represents the 75th percentile. Very ingredient-heavy recipes may be perceived as intimidating or special occasion recipes, potentially affecting ratings differently than standard recipes.

4. **`is_quick_recipe`** (Nominal binary)
   - 1 if cooking time is under 30 minutes, 0 otherwise
   - **Why it's good:** Quick recipes appeal to busy users seeking convenience. This threshold captures "weeknight dinner" recipes that might be rated favorably for their practicality.

5. **`complexity_score`** (Quantitative engineered)
   - Formula: (n_ingredients × n_steps) / log(minutes + 1)
   - **Why it's good:** Combines multiple complexity dimensions into a single score. A recipe with many ingredients and steps but short cooking time might be complex to execute (high score), while a simple recipe with few ingredients, few steps, and reasonable time has a low score. This captures overall recipe difficulty.

6. **`submitted_year`** (Quantitative)
   - Year the recipe was submitted to Food.com
   - **Why it's good:** User rating behavior may have changed over time as the platform evolved. Early adopters might rate differently than later users, or rating inflation/deflation may have occurred over the 10-year span.

These features improve the model by capturing **non-linear relationships** (log_minutes), **interaction effects** (complexity_score), **threshold effects** (has_many_ingredients, is_quick_recipe), and **temporal trends** (submitted_year) that a simple linear model on raw features would miss.

---

### Model Selection and Hyperparameter Tuning

**Model:** Random Forest Regressor
- Chosen over Linear Regression for its ability to capture non-linear relationships
- Can automatically handle feature interactions
- More robust to outliers than linear models

**Hyperparameter Search:**

I used **GridSearchCV with 5-fold cross-validation** to tune the following parameters:

- **`n_estimators`** [50, 100, 200]: Number of trees in the forest
  - More trees = more stable predictions but longer training time
- **`max_depth`** [None, 10, 20]: Maximum depth of each tree
  - Controls model complexity and overfitting risk
- **`min_samples_split`** [2, 5, 10]: Minimum samples required to split a node
  - Higher values prevent overfitting by requiring more evidence before splitting
- **`min_samples_leaf`** [1, 2, 4]: Minimum samples required at leaf node
  - Controls granularity of predictions

**Total combinations tested:** 108

**Best Hyperparameters Found:**
- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 10
- `min_samples_leaf`: 2

---

### Final Model Performance

**Performance Metrics:**
- **Training RMSE:** 0.5891
- **Test RMSE:** 0.6128
- **Training R²:** 0.1228
- **Test R²:** 0.0527

**Top 5 Most Important Features:**
1. `submitted_year` (0.2841)
2. `n_steps` (0.1523)
3. `complexity_score` (0.1289)
4. `minutes_cleaned` (0.1167)
5. `n_ingredients` (0.1045)

<iframe src="assets/fig_predictions.html" width=800 height=600 frameBorder=0></iframe>

<iframe src="assets/fig_importance.html" width=800 height=600 frameBorder=0></iframe>

---

### Improvement Over Baseline

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| Test RMSE | 0.6260 | 0.6128 | -0.0132 (2.11% reduction) |
| Test R² | 0.0105 | 0.0527 | +0.0422 |

The final model shows **meaningful improvement** over the baseline:

- **RMSE decreased by 0.0132 points** (2.11% reduction), meaning predictions are more accurate on average
- **R² increased by 0.0422**, now explaining 5.27% of variance in ratings (a 5x improvement over baseline's 1.05%)
- The Random Forest's ability to capture **non-linear relationships** and **feature interactions** led to better predictions
- Engineered features, especially `submitted_year` and `complexity_score`, proved highly predictive

While 5.27% explained variance may seem low, predicting recipe ratings is inherently difficult because ratings are subjective and influenced by many factors we cannot measure (user skill level, ingredient quality, personal taste preferences, etc.). The improvement demonstrates that recipe characteristics do provide some signal about expected ratings, even if they don't fully determine them.

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
- French recipes: 0.6384
- Non-French recipes: 0.6125
- Absolute difference: 0.0259

**Permutation Test:** 10,000 iterations  
**P-value:** 0.1247

<iframe src="assets/fig_fairness.html" width=800 height=600 frameBorder=0></iframe>

<iframe src="assets/fig_residuals.html" width=800 height=600 frameBorder=0></iframe>

---

### Conclusion

With a p-value of 0.1247, we **fail to reject the null hypothesis** at the α = 0.05 significance level.

**Conclusion:** Our model appears to perform **fairly** between French and non-French recipes. The difference in RMSE between these groups (0.0259) is not statistically significant and could reasonably be due to random chance.

This suggests that the model treats both types of recipes similarly and does not systematically favor one group over the other in terms of prediction accuracy. While French recipes have slightly higher average error (0.6384 vs 0.6125), this difference is within the range we'd expect from random variation given the sample sizes.

The fairness of our model is reassuring, especially given that:
1. French recipes are a minority class (only 1.3% of the dataset)
2. The model was not specifically designed to handle this imbalance
3. French recipes might have different rating patterns due to cultural factors

The lack of significant bias suggests our engineered features capture recipe quality in a way that generalizes across cuisine types.
