Robocon 2025 Ball Launch Prediction

This repository contains the intelligence behind our Robocon 2025 robot's ball launching system! It explores both machine learning models and a physics-based solution to accurately predict the required RPM for precise ball launches.
✨ Project Overview

Our goal for Robocon 2025 is precision in ball launching. This project focuses on:

    Predicting Optimal RPM: Determining the exact motor RPM needed for the ball launcher.

    Machine Learning Approach: Developing and evaluating neural networks and a Random Forest model, pre-trained on synthetic datasets.

    Physics-Based Approach: An analytical solution to calculate launch parameters.

📦 Key Components
📊 Synthetic Datasets

You'll find two synthetic datasets used for training and evaluating our models:

    launch_dataset_small.csv

    launch_dataset_large.csv
    These datasets simulate various launch scenarios, providing distance_m, angle_deg, and rpm values.

🧠 Machine Learning Models (model_V1.py - model_V5.py)

We've explored several iterations of models to predict rpm based on distance and angle:

    model_V1.py: Initial neural network, predicting both angle and RPM from distance.

    model_V2.py: Improved neural network, focusing on RPM prediction with both distance and angle as inputs, using StandardScaler.

    model_V3.py: Further refinement using the larger dataset and MinMaxScaler for better performance.

    model_V4.py: Advanced neural network with feature engineering (e.g., trigonometric features, interaction terms) and regularization techniques like BatchNormalization and Dropout.

    model_V5.py: An alternative approach using a RandomForestRegressor to predict RPM.

    model_V6.py: This is an alternate apporach that uses XGBoost to predict the rpm.

    model_V7.py: This is an advanced method of predicting rpm which uses the ensemble model.
    
Note: All these models are pre-trained on the synthetic datasets. Due to practical constraints, real-life datasets for fine-tuning were not available.
📐 Physics-Based Launch Solution

This is a robust, analytical approach that complements the ML models. It's designed to calculate the optimal initial velocity (and thus the required RPM) and launch angle for a ball to hit a target basket. It takes into account:

    Horizontal and vertical distances to the target.

    Constraints on both the launch angle and the desired impact angle (the angle at which the ball hits the basket).
    This solution provides a reliable, physics-driven RPM value for your launcher.

🚀 How to Use
Running the ML Models

To run and evaluate any of the machine learning model scripts:

    Ensure the launch_dataset_small.csv and launch_dataset_large.csv files are in a src/ directory relative to your model scripts.

    Install the necessary Python libraries:

    pip install numpy pandas scikit-learn tensorflow keras scipy matplotlib seaborn

    Execute the desired model script:

    python model_V4.py

    (Replace model_V4.py with the version you want to run.)

Integrating with a Robot System

The calculated RPM (from either the ML models or the physics solution) would be fed into your robot's control system. For instance, in a ROS 2 setup, this RPM value would be sent via a service call to a dedicated launcher node that controls the physical motors.
⚙️ Installation

    Clone the repository:

    git clone <your-repository-url>
    cd <your-repository-name>

    (Replace <your-repository-url> and <your-repository-name> with your actual repo details.)

    Place datasets: Ensure launch_dataset_small.csv and launch_dataset_large.csv are in a src/ folder within the cloned directory.

    Install Python dependencies:

    pip install numpy pandas scikit-learn tensorflow keras scipy matplotlib seaborn

📄 License

This project is open-sourced under the GNU General Public License. See the LICENSE file for more details.
📧 Contact

If you have any questions or feedback, please feel free to reach out!
