# Lab1- DimRed

#### Author: David Zapata Chaves

Data App: https://dimensionality-reduction-mnist-davidzeta.streamlit.app/


9. **What are the underlying mathematical principles behind UMAP? What is it useful for?**

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that is used to visualize and explore high-dimensional data. UMAP is based on a mathematical framework that involves concepts from topology, graph theory, and optimization.

UMAP works by constructing a high-dimensional graph of the data, where each point in the data is a node in the graph, and edges between nodes are weighted based on the similarity between the corresponding data points.

To determine connectedness, UMAP extends a radius outwards from each point, connecting points when those radii overlap. Choosing this radius is critical - too small a choice will lead to small, isolated clusters, while too large a choice will connect everything together. UMAP overcomes this challenge by choosing a radius locally, based on the distance to each point's nth nearest neighbor. UMAP then makes the graph "fuzzy" by decreasing the likelihood of connection as the radius grows. Finally, by stipulating that each point must be connected to at least its closest neighbor, UMAP ensures that local structure is preserved in balance with global structure.

The goal of UMAP is to find a low-dimensional embedding of the data that preserves the underlying structure of the high-dimensional graph. To achieve this, UMAP uses a combination of optimization techniques that involve minimizing a cost function that balances the preservation of local and global structure in the data.

UMAP is useful for a wide range of applications in data analysis and machine learning, including but not limited to:

* Visualizing high-dimensional data in a low-dimensional space, which can help identify patterns and relationships in the data.

* Clustering and classification of high-dimensional data, where UMAP can be used as a preprocessing step to reduce the dimensionality of the data and make it more manageable for other algorithms.

* Exploring complex data sets, where UMAP can be used to identify subpopulations or clusters of data points that may not be apparent in the original high-dimensional space.

* Anomaly detection, where UMAP can be used to identify data points that are significantly different from the rest of the data.

* Reinforcement learning, where UMAP can be used to represent high-dimensional states and actions in a lower-dimensional space, which can make the problem more tractable for learning algorithms.


10. **What are the underlying mathematical principles behind LDA? What is it useful for?**

Linear Discriminant Analysis or LDA, is a dimensionality reduction technique that is used to find a linear combination of features that best separates different classes in the data. LDA is based on a mathematical framework that involves concepts from linear algebra, probability theory, and optimization.

LDA focuses primarily on projecting the features in higher dimension space to lower dimensions. This three steps explain the process:

* First, is important to calculate the separability between classes which is the distance between the mean of different classes. This is called the between-class variance.

* Secondly, calculate the distance between the mean and sample of each class. It is also called the within-class variance.

* Finally, there is constructed the lower-dimensional space which maximizes the between-class variance and minimizes the within-class variance. P is considered as the lower-dimensional space projection, also called Fisher’s criterion.

At resume, LDA works by transforming the original features of the data into a new set of features that maximize the separation between different classes while minimizing the variation within each class. The goal of LDA is to find a set of linear discriminant functions that project the data onto a lower-dimensional space while preserving the maximum amount of class-discriminating information.

LDA is useful for a wide range of applications in data analysis and machine learning, including but not limited to:

* Classification of data into two or more classes, where LDA can be used to find the linear discriminant functions that best separate the classes.

* Dimensionality reduction, where LDA can be used to transform high-dimensional data into a lower-dimensional space while preserving the maximum amount of class-discriminating information.

* Feature selection, where LDA can be used to identify the most important features that contribute to the separation between different classes.

* Image and signal processing, where LDA can be used to classify images or signals into different classes based on their features.

* Text classification, where LDA can be used to classify text documents into different categories based on the words and phrases they contain.

## How use the streamlit App?


* 1. Install [dependencies](requirements.txt)

    ```
    pip install -r requirements.txt
    
    ```


* 2. Run demo [wep-app](app.py)
    ```
    streamlit run app.py
    
    ```