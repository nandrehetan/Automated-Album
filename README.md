# Automated Album

## Introduction

*Moments Become a Memory, Memories Become Treasure.* Today, the most popular way to save a memory is to capture it, but not all captures are perfect. This project aims to present you with the most perfect captures in an organized manner.

## Problem Statement

Sorting out the best images while filtering out the blurred and unoriented ones. Organizing them into different clusters based on similarities.

## Solution

### Identification

- **VGG-16 for Natural Images:** VGG-16, characterized by its deep architecture with 16 layers, is effective in natural image classification tasks.

- **Face Orientation:** Utilizing MediaPipe for facial landmark detection and calculating a score based on the sum of yaw, pitch, and roll angles to identify the face orientation. A lower score indicates a better image.

- **Clustering:** Using face recognition to find face encodings and applying clustering based on Euclidean distance to group similar images into clusters.

- **Blur Detection:** Using the Variance_of_Laplacian function to calculate the variance of Laplacian, serving as a measure of image sharpness. A higher score indicates a sharper image.

## Limitations

1. **Eye Orientation:** Sometimes, even if the head is facing the camera, the eyes may not.
2. **Changing Trends:** Not every person looks into the camera (e.g., Portraits, Aesthetics, Candids).
3. **Face Accessories:** Precision in detecting faces may be compromised when people wear sunglasses, masks, etc.

## Future Scope

1. Syncing the system to the calendar to enrich memories of festivals in a land of diverse cultures like India.
2. Implementing eye orientation detection by detecting the position of the iris.

*Despite all the challenges and limitations, every new feature added is worth all the efforts and happiness stored in the albums. Every organized photo album is a day in itself!*
