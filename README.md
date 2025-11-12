# Tennis Movement Tracking and Analysis

This project provides a system for tracking and analyzing tennis movements from video footage and comparing them to a database of professional tennis players' movements.

## Features

- Real-time pose estimation for tennis players
- Movement tracking and analysis
- Comparison with professional player movements
- Visualization of movement patterns

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `database` folder and add professional tennis videos for comparison.

## Usage

1. Run the main script:
```bash
python main.py
```

2. Follow the prompts to:
   - Select a video file for analysis
   - Choose comparison parameters
   - View the analysis results

## Project Structure

- `main.py`: Main script for video processing and analysis
- `pose_estimation.py`: Pose estimation and tracking module
- `movement_analysis.py`: Movement analysis and comparison module
- `database/`: Directory for storing professional tennis videos
- `output/`: Directory for storing analysis results 