# D-ABM: Spatiotemporal Dynamics Simulator

This project is a dynamic Agent-Based Model (ABM) simulator built with Streamlit, designed to visualize spatiotemporal dynamics, specifically focusing on Warning Response Latency (WRL) and flow regimes (Inflow/Outflow) under disaster scenarios.

## Features

*   **Real-time Agent Distribution (Map)**: Visualizes agents moving in a 2D grid environment, distinguishing between exposed, sheltering, and safe agents, as well as risk zones.
*   **Warning Response Latency (WRL)**: Displays the distribution of time steps it takes for agents to respond to warnings and take protective action.
*   **Flow Regime (DII vs DOI)**: Analyzes the dynamic flow of agents into and out of risk zones using Dynamic Inflow Index (DII) and Dynamic Outflow Index (DOI), categorizing the state into four quadrants: Transit, Source, Stranded, and Quiet.
*   **Interactive Controls**:
    *   **Environment**: Adjust population size, grid dimensions, and initial home ratio.
    *   **Mechanism**: Tune information diffusion rate, compliance probability, and agent movement speed.
    *   **Simulation Control**: Continuous run, single-step execution, stop, and reset functionality. Adjustable simulation speed and batch size.

## Project Structure

```
App/
├── app.py              # Main Streamlit application code
├── requirements.txt    # Python dependencies
├── images/             # Folder containing logo images
│   ├── PKU_logo.png
│   └── PKU_logo2.png
└── README.md           # Project documentation
```

## Installation

1.  **Prerequisites**: Ensure you have Python installed (Python 3.8+ recommended).

2.  **Install Dependencies**:
    Open a terminal in the project root directory and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Application**:
    In your terminal, navigate to the `App` directory and execute:
    ```bash
    streamlit run app.py
    ```

2.  **Interact with the Simulator**:
    *   The application will open in your default web browser.
    *   Use the sidebar on the left to configure simulation parameters.
    *   Click **"▶ Run"** to start the simulation loop.
    *   Click **"⏯ Step"** to advance the simulation by one time step.
    *   Click **"⏹ Stop"** to pause the continuous run.
    *   Use the sliders to adjust animation speed and calculation steps per update.

## Technologies Used

*   **Streamlit**: For the interactive web interface.
*   **NumPy**: For efficient numerical computations and agent state management.
*   **Matplotlib & Seaborn**: For generating high-quality static visualizations and heatmaps.
*   **Pandas**: For data handling (if extended for data export).

## Credits

**Resilience Safety and Disaster Planning Workshop**
Peking University
