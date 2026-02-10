# Medium SEO Experiment: Product Sense

This project is a dashboard designed to measure the impact of SEO metadata changes (Title & Description) on the performance of a Medium article. It uses a **Synthetic Control Method** to isolate the effect of the changes from platform-wide trends.

## 🧪 Experiment Overview

*   **Hypothesis**: Aligning SEO metadata with high-intent search queries (e.g., "Product Sense Interview Framework") will increase external traffic (Google View Velocity), leading to higher overall views, reads, and earnings.
*   **Treatment**: The "Product Sense in Data Science" article (Title/Description updated on 2026-01-28).
*   **Control Group**: A synthetic composite of three similar articles ("Analytics Projects", "Tiered Metrics", "CLTV Modeling") to serve as a baseline for normal traffic behavior.

## 📊 Dashboard Features

The Streamlit dashboard (`dashboard.py`) provides real-time analysis across three key dimensions:

### 1. Velocity Analysis
Measures the speed of view accumulation (views/day) before and after the experiment start date.
*   **Acceleration**: The percentage increase in daily views compared to the historical baseline.
*   **Net Lift**: The acceleration of the treated article minus the acceleration of the control group (filtering out seasonality).

### 2. Indexed Growth (Lift)
Normalizes all articles to start at **100** on Day 0 of the experiment to visualize divergence.
*   **Summary Charts**: Shows the "Treated" line vs. the "Control Range" (gray band representing min/max of controls).
*   **Detailed Breakdown**: Itemized growth curves for every individual article, grouped by **Topic**.

### 3. Efficiency Metrics
Tracks the quality of the traffic, not just the volume.
*   **Earnings Per View (EPV)**: Value generated per unique view.
*   **Read Ratio**: Percentage of viewers who read the article (engagement quality).

## 🚀 Setup & Usage

### Prerequisites
*   Python 3.8+
*   `pip`

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment variables (if needed) in `.env`.

### Running the Dashboard
```bash
streamlit run dashboard.py
```
The dashboard will open locally at `http://localhost:8501`.

## 📂 Project Structure
*   `dashboard.py`: Main application logic.
*   `requirements.txt`: Python dependencies.
*   `README.md`: Activity documentation.
