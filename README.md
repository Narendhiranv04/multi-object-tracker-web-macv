

# **MULTI-OBJECT TRACKER WEB APPLICATION - MACV**  


## **Overview**  

This project focuses on building a **multi-object tracking web application** using tracking models, with a special emphasis on **DeepSORT** and **SORT**.  

The application processes video uploads, applies the selected tracking model, and outputs annotated videos for download.  


![Screenshot from 2024-12-22 14-30-48](https://github.com/user-attachments/assets/f91f0a80-26f1-43f4-b300-8424e5b62ed1)




---

## **Features**  

- Upload video files for object tracking.  
- Choose between **DeepSORT** and **SORT** models for tracking.  
- Real-time processing with progress indication.  
- Download annotated video outputs.  

---

## **Models Considered**  (For analysis purposes)

| Model          | Description                                                                       | Strengths                                            | Limitations                                           | Decision                                      |
|----------------|-----------------------------------------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------|-------------------------------------------------|
| **SORT**       | A simple and fast tracker based on Kalman filters and Hungarian algorithm.        | Lightweight, real-time speed.                       | Struggles with occlusion and identity switches.       | Selected for its speed and simplicity.       |
| **DeepSORT**   | An extension of SORT with appearance descriptors using deep learning.             | Handles occlusion and re-identification well.       | Computationally heavier than SORT.                    | Selected for its balance of accuracy.         |
| **FairMOT**    | Joint detection and embedding-based tracking.                                    | High accuracy, end-to-end system.                   | Requires significant computational resources.         | Not chosen due to complexity.                 |
| **ByteTrack**  | Combines high and low-confidence detections for robust tracking.                  | Robust in noisy environments.                        | Computationally intensive for real-time deployment.    | Not chosen for simplicity.                    |
| **TrackFormer**| Transformer-based tracking model.                                                 | Leverages transformer capabilities.                  | Requires high-end GPUs, slower inference speed.        | Not chosen for hardware limitations.          |

---

## **Why DeepSORT?**  

1. **Accuracy vs. Speed Tradeoff**  

2. **Applicability**  

3. **Ease of Integration**  

4. **Resource Constraints**  

---

## **How to Run the Application**  
```
Step 1: **Clone this repository**  

```bash
git clone <repository_url>
cd <repository_name>
```

Step 2: **Install dependencies**  

```bash
pip install -r requirements.txt
```

Step 3: **Start the application**  

```bash
uvicorn app:app --reload
```

Step 4: **Open the web application**  

Go to this link in your browser:
```
http://127.0.0.1:8000
```

