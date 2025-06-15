##This project involves the development of a Recommendation System using collaborative filtering techniques. The system predicts user preferences for items they haven't rated yet, and recommends top-rated items based on predicted scores.
##We use the Surprise library in Python, which provides built-in tools for building and evaluating recommendation algorithms. The algorithm used is SVD (Singular Value Decomposition), which is widely used for matrix factorization in collaborative filtering tasks.

# Step 1: Import required libraries
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Step 2: Load sample data
# You can replace this with your own dataset (e.g., movies.csv, ratings.csv)
ratings_dict = {
    "item": ["A", "A", "A", "B", "B", "C", "C", "D"],
    "user": ["user1", "user2", "user3", "user1", "user2", "user1", "user3", "user2"],
    "rating": [5, 3, 2, 4, 2, 2, 5, 3],
}
df = pd.DataFrame(ratings_dict)

# Step 3: Prepare data using Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

# Step 4: Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Step 5: Build and train the model
model = SVD()
model.fit(trainset)

# Step 6: Make predictions
predictions = model.test(testset)

# Step 7: Evaluate the model
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Step 8: Recommend items for a user (e.g., 'user1')
def get_top_n_recommendations(predictions, n=5):
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # sort by estimated rating
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n_recommendations(predictions, n=5)
print("Top Recommendations:")
for user, recs in top_n.items():
    print(user, recs)
