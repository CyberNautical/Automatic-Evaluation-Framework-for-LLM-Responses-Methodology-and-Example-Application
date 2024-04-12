# Automatic Evaluation of LLM Responses and Example Application

## Introduction
This document explores the integration of an unsupervised mathematical evaluation method with tailored relevance evaluations to assess AI-generated content effectively. The approach, detailed in the whitepaper by researchers from Peking and Tianjin Universities, titled [Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment](https://ar5iv.labs.arxiv.org/html/2402.01830) employs peer-review mechanisms among Large Language Models (LLMs). These models evaluate each other’s responses to unlabeled questions, aiming to align their outputs with human preferences efficiently without direct feedback. This highlights the advantages of automatic evaluation methods over traditional supervised evaluations, particularly in how quickly and effectively they can be applied.

## Methodology:
The methodology combines mathematical techniques with a custom relevance evaluation to analyze the consistency and quality of the models' responses:
- **Permutation Entropy (PEN)**: Quantifies the randomness of response rankings compared to human rankings.
- **Count Inversions (CIN)**: Measures the degree of disorder within these rankings.
- **Longest Increasing Subsequence (LIS)**: Identifies the length of the most consistent sequence of responses, mirroring human judgment.
- **Custom Relevance Evaluation**: Scores responses based on criteria such as accuracy, completeness, engagement, or alignment with a given prompt.

Together, these tools provide a robust framework for evaluating AI-generated responses, particularly in contexts where responses are open-ended and there is no single correct answer.

## Example Application
**Scenario**:
The goal is to evaluate whether LLM-generated marketing content is suitably structured for posting on a social media blog. This content must be locally relevant and align with the targeted audience, brand, and voice. It's important to note that the content creation here exclusively employs typical Prompt Engineering without the use of more complex techniques like Retrieval Augmented Generation (RAG) or model fine-tuning. This choice underscores the efficiency of the evaluation process—content was created and begun evaluation within the same day, demonstrating the rapid deployment capability of this approach.

### Approach: 
1. **System Prompt**: Define the prompt used for generating marketing content by the LLM. [System Prompt](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/System%20Prompt.md).
2. Automated Relevance Evaluation: Employ an LLM to score the relevance of responses based on curated examples, employing an evaluation system developed in Promptflow. [Relevance Evaluation Prompt](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/Relevance%20Eval.md).
3. Extended Evaluation Metrics: Implement additional evaluations for PEN, CIN, and LIS to assess the flow of content relevance over a series of responses.

### Implementation and Results

- **Variant 1**:  A prompt requesting an advertisement for "Neb Cafe latte" for a blog post aimed at boosting brand awareness among adults who prefer home-brewed coffee. The response emphasized luxury, convenience, and local relevance, earning a perfect score for relevance. [Variant 1](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/Variant%201.md)
- **Variant 2**: Modified the response to introduce an irrelevant product mix (pizza in coffee), which was poorly evaluated due to its inconsistency with the brand and audience expectations, demonstrating the effectiveness of the LLM in distinguishing relevant content. [Variant 2](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/Variant%202.md)

### Evaluation for PEN, CIN, LIS
Below is the Python code I used to calculate the mentioned metrics. This code is crucial for quantitatively assessing the structure and consistency of the LLM-generated content. Understanding these metrics helps in tuning the AI models to produce outputs that are not only relevant but also diverse and engaging.
```python
import numpy as np
from scipy.stats import entropy
import itertools

def permutation_entropy(time_series, m, delay):
    """Calculate the Permutation Entropy."""
    n = len(time_series) # Number of elements in the time series
    permutations = np.array(list(itertools.permutations(range(m))))  # All possible permutations of order 'm'
    c = np.zeros(len(permutations)) # Initialize a count array for each permutation
    
    for i in range(n - delay * (m - 1)): # Iterate over the time series with the specified delay
        # Sorted time series pattern index 
        sorted_index_array = np.argsort(time_series[i:i + delay * m:delay]) # Get the indices that would sort the subsequence
        for j in range(len(permutations)): # Check each permutation
            if np.array_equal(sorted_index_array, permutations[j]): # If the sorted indices match a permutation
                c[j] += 1 # Increment the count for this permutation

    c /= c.sum() # Normalize the counts to get a probability distribution
    pe = entropy(c) # Calculate the Shannon entropy of the distribution
    return pe

def count_inversions(sequence):
    """Count inversions in the sequence."""
    n = len(sequence) # Length of the sequence
    count = 0 # Initialize inversion count
    for i in range(n): # Iterate over each element in the sequence
        for j in range(i + 1, n): # Iterate over elements after the current element
            if sequence[i] > sequence[j]: # If an element later in the sequence is smaller
                count += 1 # It's an inversion, increment the count
    return count

def longest_increasing_subsequence(sequence):
    """Calculate the Longest Increasing Subsequence."""
    n = len(sequence) # Length of the sequence
    lis = [1] * n # Initialize LIS value for each element to 1

    for i in range(1, n): # Start from the second element
        for j in range(0, i): # Compare with all elements before it
            if sequence[i] > sequence[j] and lis[i] < lis[j] + 1: # If the current element can extend the increasing sequence
                lis[i] = lis[j] + 1 # Update the LIS for this element

    return max(lis) # Return the maximum value in LIS array

# Example sequence with relevance scores from 1 to 5
relevance_scores = [5, 4, 5, 4, 5, 4, 5, 4]

# Calculate metrics
perm_entropy = permutation_entropy(relevance_scores, m=3, delay=1)
inversions = count_inversions(relevance_scores)
lis_length = longest_increasing_subsequence(relevance_scores)

perm_entropy, inversions, lis_length
```
```
Result: (0.6931471805599453, 10, 2)
```
Here are the results from evaluating the given relevance scores [5, 4, 5, 4, 5, 4, 5, 4] using the specified metrics:

**1. Permutation Entropy**: 0.69
- This measures the randomness in the arrangement of relevance scores. A higher value suggests a greater diversity in response quality as perceived through the evaluation. Lower the value of permutation entropy suggests less randomness and a more predictable pattern in the scores. 

**2. Count of Inversions**: 10
  - This number represents how many times a higher score appears before a lower score in your sequence. A lower count would indicate a more consistent or orderly ranking, while a higher count suggests some disorder in how responses were scored relative to each other. Despite the high quality of the responses, the alternating pattern leads to a higher count of inversions, indicating significant disorder due to the way scores alternate between two values.

**3. Longest Increasing Subsequence**: 2
- The length of the longest subsequence where each score is higher than or equal to the previous score in the sequence. This indicates the largest set of responses that show an increasing or at least non-decreasing trend in quality. The longest subsequence where each score is non-decreasing is limited to two, reflecting the back-and-forth pattern in the scoring.

These results demonstrate how the specified metrics can quantitatively evaluate the structure and consistency of LLM-generated content, providing a systematic approach for refining marketing strategies and extending these principles to various forms of generative AI applications. The calculated Permutation Entropy of 0.69, while indicating some diversity, suggests that there could be greater variability in the AI's responses. This insight can guide us to adjust the prompt engineering process to encourage a wider range of creative outputs. Meanwhile, the high Count of Inversions indicates a need to improve the sequence ordering to make the narrative flow more logical and appealing.

### Benefits of Using Combined Techniques:
- **Comprehensive Feedback**: The combination allows evaluators to not only measure how "correct" or "appropriate" an individual response is but also to analyze the overall behavior of the AI across a series of responses. This is crucial for applications like chatbots, creative content generation, and educational tools, where context and progression are important.
- **Improved Model Training**: These metrics can also inform the training process for AI models. Understanding where and how frequently inversions occur or how long the longest increasing subsequence is can help in tuning the model to produce more coherent and contextually appropriate responses.
- **Scalability**: Automatic evaluations are scalable and can handle large volumes of data without the need for extensive human intervention. This is especially useful in iterative development environments and continuous deployment settings.
- **Objective Analysis**: By quantifying aspects of the generated content, developers and researchers can more objectively compare different models or different configurations of the same model, leading to more data-driven decision-making.

### Expanding the Horizon
The potential of this evaluation framework extends beyond textual content generation. It can be adapted to assess a variety of open-ended responses beyond textual content. For instance:

- **GenAI Images**: Evaluate the relevance and creativity of images generated from textual descriptions. By applying similar metrics, one can assess how closely the sequence of generated images aligns with a series of descriptions or the thematic consistency across a portfolio of generated artwork.
- **Interactive Media**: In interactive applications like video games or virtual reality, the framework can be used to evaluate narrative coherence or the adaptive responses of AI entities to user interactions.
- **Educational Content**: For AI-generated educational material, these metrics can help in assessing the alignment of content with educational standards and learning objectives, ensuring the material's utility and relevance.

### Conclusion
The integration of advanced mathematical metrics with custom relevance evaluations presents a new frontier in the automatic assessment of AI-generated content. This approach not only accelerates the evaluation process but also enhances its precision, making it indispensable in the era of rapid AI development. Future work should explore the adaptation of these metrics across different forms of AI outputs, such as visual content generated by AI and interactive AI systems in gaming. 
