# Tennis Swing Tracket

3d post detection of tennis swings from video footage using MediaPipe
- extracts joint coords, computes geometric transformation to normalize 3d data (uses court boundaries + net as camera angle baselines)
- cross-video comparative analysis across database of professional 3d swing paths
- explores a "similarity scoring" to suggest optimal playing style


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
