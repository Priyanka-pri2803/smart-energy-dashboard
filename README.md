# Smart Energy Dashboard

 **Smart Energy Consumption Prediction Dashboard**  

This project is a fully interactive **Streamlit-based energy prediction dashboard** with AI models (LSTM & XGBoost) for predicting energy consumption. It includes **animated line charts, bar charts, pie charts, metrics counters, and live simulation effects**, making it look like a professional SaaS dashboard.

---

## Features

- **AI Models**: LSTM and XGBoost for energy prediction.
- **Live-streaming Simulation**: Line charts and metrics update in real-time.
- **Animated Metrics**: Total, Average, and Max energy usage with animated counters.
- **Interactive Charts**:  
  - Flowing line chart for predicted energy usage over time  
  - Gradient & shiny bar charts for top devices  
  - Donut-style pie chart for energy distribution  
- **Shiny Headers & UI Effects**: Gradient, hover, and shining highlights.
- **CSV Upload & Download**: Upload your own CSV and download predictions.
- **Google Drive Model Download**: Automatically downloads models if not present.
- **Optional Live Simulation Toggle**: For faster static preview or full animation.

---

## Installation & Running

### Step 1: Clone repository 
```bash
git clone https://github.com/Priyanka-pri2803/smart-energy-dashboard
cd smart-energy-dashboard/backend
Step 2: Create and activate virtual environment

Windows:

python -m venv .venv
.\.venv\Scripts\activate


Linux / MacOS:

python3 -m venv .venv
source .venv/bin/activate

Step 3: Install requirements
pip install -r requirements.txt

Step 4: Run Streamlit dashboard
streamlit run app_streamlit.py


Your dashboard should open in the browser. Upload a CSV file and see live animated predictions, metrics, and charts.

License

© 2025 Priyanka K. & HithaShree K. All rights reserved.
Unauthorized copying or use of this project is prohibited.

Authors


Priyanka K – Lead Frontend Developer & Project Implementer  
> Designed and developed the complete dashboard interface, frontend visuals, and data integration components.  
> Also implemented the AI model integration and managed the overall project development.  
> Contact: priyanka280303@gmail.com  

HithaShree K  – Supporting Developer (Dashboard & Testing)  
> Contributed to interface testing and assisted in minor frontend adjustments during the implementation phase.  
> Contact: hithagshree.govi@gmail.com  


Acknowledgements

Models based on prior research and NSL-KDD dataset.

Streamlit and Plotly for dashboard visualization.

Google Drive for hosting models.


---

### **requirements.txt**


streamlit
pandas
numpy
scikit-learn
tensorflow
xgboost
joblib
gdown
plotly



---
