# SSAL Research Framework

A modular research framework for Self-Supervised Learning (SSL), Active Learning (AL), and Self-Supervised Active Learning (SSAL).

## Overview

This repository is developed as part of an M.Tech Thesis focused on improving label-efficient image classification through the integration of:

* Self-Supervised Learning
* Deep Active Learning
* Class Rebalancing
* Uncertainty-Based Querying
* Diversity-Aware Sampling

The framework reproduces the SSAL methodology and serves as a platform for future research involving:

* SimCLR
* MoCo
* BYOL
* DeepCluster
* MAE
* BADGE
* CoreSet
* BALD
* Novel SSL + AL combinations

---

## Research Goals

1. Reproduce SSAL results on standard benchmarks.
2. Benchmark modern SSL techniques in active learning settings.
3. Investigate class-balanced active learning strategies.
4. Improve label efficiency under limited annotation budgets.
5. Build a reusable open-source research framework.

---

## Supported Datasets

* CIFAR-10
* CIFAR-100
* SVHN
* FashionMNIST
* TinyImageNet

---

## Planned SSL Methods

* Rotation Prediction
* SimCLR
* MoCo
* BYOL
* DeepCluster
* MAE

---

## Planned Query Strategies

* Random Sampling
* Entropy Sampling
* Least Confidence
* Margin Sampling
* BADGE
* CoreSet
* BALD
* UMF + CReGS (SSAL)

---

## Project Status

Current Stage:
Repository Foundation and SSAL Reproduction

---

graph TD
    classDef data fill:#e8f4f8,stroke:#2b6cb0,stroke-width:2px,color:#1a202c;
    classDef ssl fill:#fefcbf,stroke:#dd6b20,stroke-width:2px,color:#1a202c;
    classDef backbone fill:#edf2f7,stroke:#4a5568,stroke-width:2px,color:#1a202c;
    classDef al fill:#ebf4ff,stroke:#4c51bf,stroke-width:2px,color:#1a202c;
    classDef cregs fill:#faf5ff,stroke:#6b46c1,stroke-width:2px,color:#1a202c;
    classDef oracle fill:#fff5f5,stroke:#d53f8c,stroke-width:2px,color:#1a202c;
    classDef criteria fill:#fff5f5,stroke:#e53e3e,stroke-width:2px,stroke-dasharray: 5 5,color:#1a202c;

    %% 1. Dataset & Prep
    D1[(1. Dataset<br/>CIFAR-10<br/>50k Train / 10k Test)]:::data --> Prep[Data Preparation<br/>Normalization & Augmentation]:::data
    Prep --> Split{Split into Pools}:::data
    Split --> L_Pool[Initial Labeled Pool]:::data
    Split --> U_Pool[Unlabeled Pool]:::data

    %% 2. SSL Pretraining
    U_Pool --> SSL_Box
    subgraph SSL_Box [2. Self-Supervised Pretraining]
        direction LR
        Rot[Rotation SSL<br/>Predict rotation angle]:::ssl
        Sim[SimCLR<br/>Contrastive learning]:::ssl
        DC[DeepCluster<br/>Clustering in feature space]:::ssl
        MAE[Masked SSL<br/>Reconstruct patches]:::ssl
    end

    %% Backbone
    SSL_Box --> RN[ResNet-18 Backbone<br/>Feature Extraction]:::backbone
    RN -.-> PH[Projection Head<br/>For Contrastive/Clustering]:::backbone

    %% Fine Tuning
    L_Pool --> FT[Supervised Fine-Tuning<br/>On Initial Labeled Pool]:::backbone
    RN --> FT

    %% 3. Active Learning Cycle
    FT --> AL1
    U_Pool -.-> AL1
    subgraph AL_Cycle [3. Active Learning Cycle]
        direction TB
        AL1[1. Predict Unlabeled Data]:::al --> AL2[2. Calculate Uncertainty<br/>Entropy]:::al
        AL2 --> AL3[3. Calculate Mastery<br/>Density & Min Distance]:::al
        AL3 --> AL4[4. Fusion Scoring<br/>S = αU + βM]:::al
    end

    %% 4. CReGS & Oracle
    AL4 --> CReGS[4. CReGS<br/>Class Balancing]:::cregs
    CReGS --> Oracle[5. Label Query Simulation<br/>Oracle]:::oracle
    Oracle --> Update[6. Add to Labeled Pool<br/>Update Dataset]:::data
    Update --> Retrain[7. Model Retraining<br/>Supervised]:::backbone

    %% Loop & Stopping
    Retrain -.-> Stop{Stopping Criteria<br/>Budget Exhausted<br/>Max Iterations}:::criteria
    Stop -- Continue --> AL1
    Stop -- Terminate --> Eval[8. Final Evaluation<br/>Test Set]:::al
