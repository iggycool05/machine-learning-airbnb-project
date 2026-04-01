import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv('listings.csv/listings.csv')

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns)
print("\nFirst 10 rows:")
print(df.head(10))

# ==============================
# CLEAN TARGET VARIABLE (PRICE)
# ==============================

print("\nOriginal price values:")
print(df['price'].head())
print("\nPrice column type before cleaning:")
print(df['price'].dtype)

# Convert price from text like "$128.00" to numeric
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)

# Remove the top 2% most expensive listings to reduce extreme outlier influence
df = df[df['price'] < df['price'].quantile(0.98)]

# Apply log transform so the model can learn price patterns more easily
df['price'] = np.log1p(df['price'])

print("\nCleaned and transformed price values:")
print(df['price'].head())
print("\nPrice summary stats after cleaning:")
print(df['price'].describe())

# ==============================
# HANDLE MISSING VALUES
# ==============================

print("\nMissing values by column:")
print(df.isnull().sum())

# ==============================
# FEATURE PREPARATION
# ==============================

# Convert yes/no style columns into 1/0
df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})
df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0})

# Select features that may help predict Airbnb price
features = [
    'accommodates',
    'bedrooms',
    'beds',
    'bathrooms',
    'room_type',
    'property_type',
    'neighbourhood',
    'number_of_reviews',
    'review_scores_rating',
    'review_scores_location',
    'review_scores_value',
    'review_scores_cleanliness',
    'review_scores_communication',
    'latitude',
    'longitude',
    'availability_30',
    'availability_60',
    'availability_365',
    'instant_bookable',
    'host_is_superhost',
    'host_identity_verified',
    'host_listings_count',
    'host_total_listings_count',
    'minimum_nights'
]

# Keep only needed columns plus target
df = df[features + ['price']].copy()

# ==============================
# FILL MISSING VALUES
# ==============================

# Fill size-related numeric fields with medians
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
df['beds'] = df['beds'].fillna(df['beds'].median())

# Fill review columns with medians
review_cols = [
    'review_scores_rating',
    'review_scores_location',
    'review_scores_value',
    'review_scores_cleanliness',
    'review_scores_communication'
]

for col in review_cols:
    df[col] = df[col].fillna(df[col].median())

# If review count is missing, treat it as 0
df['number_of_reviews'] = df['number_of_reviews'].fillna(0)

# Fill availability columns with medians
for col in ['availability_30', 'availability_60', 'availability_365']:
    df[col] = df[col].fillna(df[col].median())

# Fill host-related fields
df['host_is_superhost'] = df['host_is_superhost'].fillna(0)
df['host_identity_verified'] = df['host_identity_verified'].fillna(0)
df['host_listings_count'] = df['host_listings_count'].fillna(df['host_listings_count'].median())
df['host_total_listings_count'] = df['host_total_listings_count'].fillna(df['host_total_listings_count'].median())

# Fill booking/pricing fields
df['instant_bookable'] = df['instant_bookable'].fillna(0)
df['minimum_nights'] = df['minimum_nights'].fillna(df['minimum_nights'].median())

# Only drop rows if critical fields are still missing
df = df.dropna(subset=[
    'price',
    'accommodates',
    'latitude',
    'longitude',
    'room_type',
    'property_type',
    'neighbourhood'
])

print("\nDataset shape after cleaning:")
print(df.shape)

# ==============================
# FEATURE ENGINEERING
# ==============================

# Add a simple location feature showing how far a listing is from the dataset center
df['distance_from_center'] = (
    (df['latitude'] - df['latitude'].mean()) ** 2 +
    (df['longitude'] - df['longitude'].mean()) ** 2
)

# ==============================
# ENCODE CATEGORICAL VARIABLES
# ==============================

# Convert text categories into numeric columns
df = pd.get_dummies(
    df,
    columns=['room_type', 'property_type', 'neighbourhood'],
    drop_first=True
)

print("\nDataset shape after encoding:")
print(df.shape)

# ==============================
# CORRELATION ANALYSIS
# ==============================

print("\nCorrelation with price:")
print(df.corr(numeric_only=True)['price'].sort_values(ascending=False))

# ==============================
# TRAIN / TEST SPLIT
# ==============================

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# STAGE 1: FEATURE SELECTION MODEL
# ==============================

# Use Random Forest first to estimate which features matter most
feature_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

feature_model.fit(X_train, y_train)

# Rank features by importance
importance = pd.Series(feature_model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 15 Features from Feature Selection Model:")
print(importance.head(15))

# Keep only the top 25 features for the final model
top_features = importance.head(25).index.tolist()

print("\nTop 25 selected features:")
print(top_features)

# Filter train and test data to only those top features
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# ==============================
# STAGE 2: FINAL MODEL
# ==============================

# Gradient Boosting often performs very well on tabular data
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train_top, y_train)

# ==============================
# EVALUATION
# ==============================

preds = model.predict(X_test_top)

# MAE in log-transformed space
mae_log = mean_absolute_error(y_test, preds)
print("\nModel MAE (log):", mae_log)

# Convert predictions and actual values back to dollar scale
preds_real = np.expm1(preds)
y_test_real = np.expm1(y_test)

# MAE in real dollars
mae_real = mean_absolute_error(y_test_real, preds_real)
print("Model MAE (real dollars):", mae_real)

# R^2 score in real-dollar space
r2 = r2_score(y_test_real, preds_real)
print("R^2 Score:", r2)

# ==============================
# FINAL MODEL FEATURE IMPORTANCE
# ==============================

# Gradient Boosting also provides feature importance
final_importance = pd.Series(model.feature_importances_, index=X_train_top.columns).sort_values(ascending=False)

print("\nTop 15 Features from Final Model:")
print(final_importance.head(15))

# ==============================
# VISUALIZATION
# ==============================

plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, preds_real, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Airbnb Prices")
plt.show()