# <span style='color:steelblue'> NLP </span>

NLP stands for Natural Language Processing. NLP allows us to extract meaning and insights from massive amounts of text information.

## Distance Metrics vs Similarity Functions

``Distance Metrics``: These methods calculate a numerical value representing the distance between the two vectors in a high-dimensional space. A smaller distance indicates the vectors are closer together.

``Similarity Functions``: These techniques assess how similar the directions of the two vectors are, regardless of their lengths. A higher similarity score suggests the vectors point in similar directions and might represent similar information.

**Distance Metrics**:

- **Euclidean Distance**: This is the most common distance metric. It calculates the straight-line distance between two points represented by the vectors. Imagine two points in space; Euclidean distance finds the regular distance between them.

    - **Objective**: Euclidean distance considers both the magnitude (length) and direction of the difference between two vectors. In k-means and KNN, we want to find data points (represented by vectors) that are close together in the overall feature space. Euclidean distance captures this well.

- **Manhattan Distance**: This method sums the absolute differences between corresponding elements of the vectors. Think of traveling along a grid city, where you can only move horizontally or vertically. Manhattan distance reflects the total distance traveled in such a scenario.

**Similarity Functions**:

- **Cosine Similarity**: This approach measures the direction similarity between vectors. It calculates the cosine of the angle between the two vectors in high-dimensional space. A higher cosine similarity score (closer to 1) indicates the vectors are pointing in more similar directions.

    - **Objective**: When comparing documents (often represented as TF-IDF vectors), we care more about the similarity in content (word usage, topics) rather than the actual word count (document length).

- **Jaccard Similarity**: This method is often used for sparse vectors (mostly zeros with few non-zero values). It measures the ratio of elements that are the same (both 1 or both 0) between the two vectors.

### Cosine Similarity

Cosine similarity is a metric used to compare two vectors (ex:- two documents words list). It essentially tells you how similar the direction of those vectors are in a high-dimensional space. Value between -1 and 1.

- A high cosine similarity score indicates two vectors indicating in similar directions.
- A low cosine similarity score indicates two vectors indicating in different directions.
- A negative cosine similarity score indicates two vectors indicating in opposite directions.

```text
Formula,

cosine_similarity(u, v) = dot(u, v) / ||u|| ||v||

u and v: The two vectors you want to compare.
dot(u, v): The dot product of vectors u and v. It measures the magnitude of their parallel components. In other words, measure how much two arrows (the vectors) point in the same direction.
||u|| and ||v||: The magnitudes (lengths) of vectors u and v, respectively.
```

It calculates the dot product of the two vectors. The dot product considers both the direction and magnitude of the vectors.
It divides the dot product by the product of the magnitudes of the individual vectors. This step removes the influence of the vectors' lengths, focusing purely on their directional similarity.
The final result is a value between -1 and 1, which is very similar to the range of cosθ. However, it's not the exact angle itself, but rather a scaled version that captures how "close" the cosine function's output (cosθ) is to 1 (indicating similar directions).

Cosine similarity is useful more than distnace calculation methods in terms documents similarity identifications, because this approach focuses on Angle but not on Magnitude / Distance between two vectors. This approach doesn't get biased if one document i.e., vector1 has higher words or lower words compared to other document i.e., vector2. It just focuses on similarity of both documents (in other words, how commonly both documents shares the topic => even one time of same shared words in both documents then mostly into same direction) but not how each of the both documents speaks majorily or quantitly about a topic.

In essence, cosine similarity borrows the concept of cosθ to quantify how similar two vectors are in direction, but it provides a more user-friendly and interpretable score between -1 and 1.

#### Vectors

A vector has two features,

- ``Magnitude (size)``: The length of the arrow represents the vector's size or strength. A longer arrow signifies a larger magnitude.
- ``Direction``: The arrow points in a specific direction. This direction is crucial information that a vector carries.
Think of it like this:

Speed is a ``scalar quantity`` (it only has magnitude, like 50 kilometers per hour). You can't say "speed towards the north."
Velocity, however, is a ``vector quantity``. It includes both speed (magnitude) and direction (e.g., 50 kilometers per hour towards the north).

`Vectors can have two dimensions or more than two dimensions! The number of dimensions refers to the number of independent directions a vector can represent.`

Higher-Dimensional Vectors (nD Vectors): Used in machine learning (representing features of data points), signal processing (audio and image data), and natural language processing (representing word embeddings).

- **Sum of Vectors**

Adding vectors results in a new vector where corresponding elements (x, y, z components) are added together. Imagine placing the tail of one vector at the head of the other. The sum is a new vector starting from the original tail and ending at the new head, formed by connecting the tips of the original vectors head-to-tail.

- **Dot Product of Vectors**

The dot product of two vectors, written as dot(u, v), is like a way to measure how much two arrows (the vectors) point in the same direction. The dot product considers both the direction and length of the arrows. The dot product is a concept that falls under the umbrella of linear algebra. Linear algebra is a branch of mathematics that deals with vectors.

The dot product, also called the scalar product, specifically applies to vectors in Euclidean space (real numbers with geometric interpretations). It utilizes the concept of vectors as arrows with both direction and magnitude to calculate a single number representing their alignment.

The dot product multiplies corresponding elements of two vectors and then sums those products. It represents the projection of one vector onto another. A high dot product indicates the vectors are pointing in similar directions, while a low dot product suggests they're more perpendicular.

Difference between sum and dot product: The sum creates a new vector, while the dot product gives a single scalar value. The sum combines the magnitudes and directions of both vectors, while the dot product captures the extent to which they're aligned.

- **Magnitude of Vector**

The magnitude of a vector represents its length or overall size. It's a non-negative scalar value (a single number).

## Vectorizer Methods

CountVectorizer and TfidfVectorizer are two common techniques in Natural Language Processing (NLP) used to convert textual data into numerical features suitable for machine learning algorithms.

### CountVectorizer

It creates a simple bag-of-words representation. It counts the occurrences of each word in a document, ignoring word order or frequency within the document.

```text
CountVectorizer => simply counting the occurrences of each word in a document.
```

- `Keypoints`:
    - Simpler and faster to compute compared to TfidfVectorizer.
    - Can be useful when word frequency is directly relevant, like analyzing word co-occurrence patterns.
    - You want a quick and basic understanding of word frequency distribution in documents.
    - The order of words doesn't matter, and you're mainly interested in word presence.
    - Dealing with short text snippets where word importance might be based on simple frequency.

### TfidfVectorizer (Term Frequency-Inverse Document Frequency)

It considers both the word frequency (importance within a document) and the inverse document frequency (how rare the word is across all documents). This helps downplay the importance of common words and emphasize words that are more specific to a document or topic.

```md
TF[Term Frequency](t, d) = (Number of times term t appears in document d) / (Total number of words in document d)

This measures how often a word appears in a specific document relative to the total number of words in that document.

IDF[Inverse Document Frequency](t) = log (Total number of documents / Number of documents containing term t)

This considers how common a word is across all documents in the corpus.

TF-IDF[Term Frequency-Inverse Document Frequency](t, d) = TF(t, d) * IDF(t)

It considers both how often a word appears in a document (importance within the document) and how uncommon it is across all documents (informative content).
```

- `Keypoints`:
    - Captures the relative importance of words, giving more weight to terms that are frequent within a document but rare across the entire corpus.
    - Less susceptible to noise from common words like "the," "a," etc.
    - You want to represent documents based on their semantic content and thematic differences.
    - The order of words might not be crucial, but the relative importance of terms within a document matters.
    - Dealing with larger document collections where focusing on informative terms is beneficial.
    - Anology
        - Imagine analyzing documents about animals. CountVectorizer would simply count how many times each word appears (e.g., "cat" - 3, "dog" - 2, "furry" - 1). TfidfVectorizer would consider both frequency and rarity. "Cat" and "dog" might get lower weights because they're common, while "furry" might get a higher weight because it's less frequent but potentially informative about the documents' content.

In essence:

- Use ``CountVectorizer`` for a quick word frequency analysis when order doesn't matter. Notice that CountVectorizer doesn't care about the order in which these words appear in the sentence. It simply creates a dictionary-like representation where each word is a key and its value is the number of times it appears.
- Use ``TfidfVectorizer`` for a more robust representation that considers both word importance within a document and its rarity across the document collection.
