import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# Accept customer reviews
reviews = []
n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input("Enter review: "))

# Convert reviews into numerical vectors
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(reviews)

# Apply LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Get words
words = vectorizer.get_feature_names_out()

# Display topics
print("\nTopics:")
for i, topic in enumerate(lda.components_):
    print("\nTopic", i + 1)
    top_words = topic.argsort()[-5:][::-1]
    for j in top_words:
        print(words[j])

# Apply t-SNE
X_dense = X.toarray()

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=min(2, n - 1)
)

X_tsne = tsne.fit_transform(X_dense)

# Display t-SNE coordinates
print("\nt-SNE Coordinates:")
for i, point in enumerate(X_tsne):
    print("Review", i + 1, ":", point)

# Visualize clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i in range(len(reviews)):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], "R" + str(i + 1))

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()