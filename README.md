# Outbreaks
**Tagline:** Preempting Waterborne Disease Outbreaks with AI and Satellite Data.

## Core Problem
Over 2 billion people use a drinking water source contaminated with feces. Waterborne diseases like cholera, typhoid, and dysentery cause hundreds of thousands of deaths annually, often in low-resource communities. Outbreaks are traditionally detected reactively—after people get sick—leading to delayed and costly responses.

## Technical Solution
Outbreaks is an early-warning predictive modeling platform that shifts response from reactive to proactive.

### Data Fusion Engine
Aggregates and processes multi-modal data:
- **Remote Sensing:** NASA/USGS/ESA satellite data on sea surface temperature, chlorophyll-a levels (algae bloom proxy), precipitation, and flood inundation maps.
- **Climate and Weather:** NOAA forecasts, historical precipitation patterns, and drought indices.
- **Socio-economic and Infrastructure Data:** Population density, access to improved water sources (JMP data), and sanitation infrastructure from local governments and NGOs.
- **Crowdsourced and Sentinel Data:** Anonymous and aggregated mobility data to track population movement post-flooding, and local water quality reports from partner health clinics.

### Predictive AI Model
An ensemble model (e.g., Gradient Boosting plus LSTMs) trained on historical outbreak data. It identifies high-risk "hotspots" by correlating environmental precursors (e.g., a heatwave followed by heavy flooding) with a high probability of pathogen contamination in water supplies.

### Actionable Dashboard and Alerts
Provides a clear, GIS-based interface for public health officials at NGOs and government agencies. The system sends targeted SMS alerts to community health workers in predicted high-risk areas weeks before a potential outbreak.

## Impact and Equity Focus
- **Saves Lives:** Drastically reduces morbidity and mortality from preventable diseases.
- **Resource Optimization:** Allows NGOs and governments to pre-position water purification tablets, medical supplies, and health teams before a crisis hits, maximizing the impact of limited aid dollars.
- **Closes the Data Gap:** Brings advanced, space-age analytics to the most vulnerable communities who lack on-the-ground water testing resources.

## MVP: Predictive Risk Scoring
This proof-of-concept trains a model on synthetic, multi-modal features and predicts a **Waterborne Disease Risk Score (0–100)** for a given latitude, longitude, and date.

### How it Works
1. Generate synthetic data that mimics environmental + socio-economic drivers.
2. Train a tree-based regressor (**XGBoost if available**, else GradientBoosting).
3. Score new coordinates and render a Folium risk map.

### Outputs
- `results/synthetic_training_data.csv`
- `results/model_report.json`
- `results/risk_scored_points.csv`
- `results/risk_map.html`

### Static Demo (GitHub Pages)
The `docs/` folder contains a static demo suitable for GitHub Pages:
- `docs/index.html` (map)
- `docs/data/` (precomputed outputs)
