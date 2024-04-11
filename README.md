# Automatic Evaluation of LLM Responses and Example Application

## Introduction
This document explores an unsupervised evaluation method detailed in the whitepaper by researchers from Peking and Tianjin Universities, titled [Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment](https://ar5iv.labs.arxiv.org/html/2402.01830). This innovative approach employs peer-review mechanisms among Large Language Models (LLMs) to evaluate each other’s responses to unlabeled questions. The method aims to efficiently align their rankings with human preferences without direct human feedback, highlighting the effectiveness of automatic evaluation methods over traditional supervised evaluations.

## Methodology:
The methodology utilizes mathematical techniques to analyze the consistency and quality of the models' evaluations:
- **Permutation Entropy (PEN)**: Quantifies the randomness of response rankings compared to human rankings.
- **Count Inversions (CIN)**: Measures the degree of disorder within these rankings.
- **Longest Increasing Subsequence (LIS)**: Identifies the length of the most consistent sequence of responses, mirroring human judgment.

These tools collectively provide nuanced insights into how well the models' responses align with human-like evaluations, facilitating an effective ranking of models based on their performance.

## Example Application
**Scenario**:
The objective is to evaluate whether LLM-generated marketing content is appropriately structured for posting on a social media blog, ensuring it is locally relevant and aligns with our targeted audience, brand, and voice.

### Approach: 
1. **System Prompt**: Define the prompt used for generating marketing content by the LLM. [System Prompt](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/System%20Prompt.md).
2. Automated Relevance Evaluation: Employ an LLM to score the relevance of responses based on curated examples, employing an evaluation system developed in Promptflow. [Relevance Evaluation Prompt](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/Relevance%20Eval.md).
3. Extended Evaluation Metrics: Implement additional evaluations for PEN, CIN, and LIS to assess the flow of content relevance over a series of responses.

### Implementation and Results

- **Variant 1**:  A prompt requesting an advertisement for "Neb Cafe latte" for a blog post aimed at boosting brand awareness among adults who prefer home-brewed coffee. The response emphasized luxury, convenience, and local relevance, earning a perfect score for relevance. [Variant 1](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/Variant%201.md)
- **Variant 2**: Modified the response to introduce an irrelevant product mix (pizza in coffee), which was poorly evaluated due to its inconsistency with the brand and audience expectations, demonstrating the effectiveness of the LLM in distinguishing relevant content. [Variant 2](https://github.com/armansalimi-microsoft/Automatic-Evaluation-LLM-Relevance-Example/blob/main/Variant%202.md)

### Custom Evaluation for PEN, CIN, LIS
```python
import numpy as np
from scipy.stats import entropy
import itertools

def permutation_entropy(time_series, m, delay):
    """Calculate the Permutation Entropy."""
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    c = np.zeros(len(permutations))
    
    for i in range(n - delay * (m - 1)):
        # Sorted time series pattern index
        sorted_index_array = np.argsort(time_series[i:i + delay * m:delay])
        for j in range(len(permutations)):
            if np.array_equal(sorted_index_array, permutations[j]):
                c[j] += 1

    c /= c.sum()
    pe = entropy(c)
    return pe

def count_inversions(sequence):
    """Count inversions in the sequence."""
    n = len(sequence)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sequence[i] > sequence[j]:
                count += 1
    return count

def longest_increasing_subsequence(sequence):
    """Calculate the Longest Increasing Subsequence."""
    n = len(sequence)
    lis = [1] * n

    for i in range(1, n):
        for j in range(0, i):
            if sequence[i] > sequence[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j] + 1

    return max(lis)

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

The evaluations underscore the framework's ability to quantify aspects of relevance and structure in LLM-generated content, offering a systematic approach to assessing and refining marketing strategies.


