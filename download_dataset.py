from roboflow import Roboflow
rf = Roboflow(api_key="rQyjZbfNLHVqKIwsbInZ")
project = rf.workspace("augmented-startups").project("drowsiness-detection-cntmz")
dataset = project.version(1).download("yolov8")