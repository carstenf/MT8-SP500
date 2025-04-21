
```mermaid
flowchart TD
    subgraph "1 Data Preparation"
        A[Import S&P500 Historical Data] --> B[Initial Cleaning]
        B --> C[Multi-Index Organization]
        C --> D[Verify Date Alignment]
        D --> E[Create Stock Metadata]
    end

    subgraph "2 Feature Engineering"
        F[Calculate Daily Returns] --> G[Compute Market Average Returns]
        G --> H[Calculate Excess Returns]
        H --> I[Generate Lagged Features]
        I --> J[Create Target Variable]
        J --> K[Feature Scaling]
        K --> L[Class Distribution Analysis]
    end

    subgraph "3 Model Development"
        M[Time-Based Cross-Validation] --> N[Train Baseline Model]
        N --> O[Train Advanced Models]
        O --> P[Model Comparison]
        P --> Q[Feature Selection]
        Q --> R[Bias-Variance Analysis]
    end

    subgraph "4 Model Optimization"
        S[Feature Subset Selection] --> T[Hyperparameter Tuning]
        T --> U[Learning Curve Analysis]
        U --> V[Optimize Class Weights]
        V --> W[Final Model Selection]
    end

    subgraph "5 Model Evaluation"
        X[Test Set Evaluation] --> Y[Performance Metrics Calculation]
        Y --> Z[Ticker-Level Analysis]
        Z --> AA[Time-Period Analysis]
        AA --> AB[Performance Visualization]
    end

    subgraph "6 Model Explainability"
        AC[SHAP Value Generation] --> AD[Feature Importance Analysis]
        AD --> AE[Partial Dependence Plots]
        AE --> AF[Individual Stock Case Studies]
        AF --> AG[Error Analysis]
    end

    subgraph "7 Output Generation"
        AH[Performance Report PDF] --> AI[Technical Documentation]
        AI --> AJ[Interactive Explanations]
    end

    E --> F
    L --> M
    W --> X
    AB --> AC
    AG --> AH