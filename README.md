Traffic Prediction System using Deep Learning (METR-LA)

This project presents a real-time traffic prediction system built using deep learning and geospatial visualization. It leverages the METR-LA dataset to forecast traffic speeds and displays predictions interactively on a map.

![image alt](https://github.com/Nandhagopal2912/Traffimap-lstm/blob/9384786bf681119b5f25fc25a211c049a2cadd0f/traffimap_snap.jpg)

![image alt](https://github.com/Nandhagopal2912/Traffimap-lstm/blob/9384786bf681119b5f25fc25a211c049a2cadd0f/trffimap_snap_2.jpg)


The system integrates:

A trained deep learning model (LSTM)
A backend API for inference
A frontend map interface for visualization

Download Dataset Files
Download METR-LA .h5 file
Download sensor coordinates:
https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv


Run Backend Server
uvicorn main:app --reload

Server runs at:

http://127.0.0.1:8000

Run Frontend
Open index.html in browser

Real-time traffic prediction
Interactive map visualization
Color-coded traffic intensity:
🟢 Green → Smooth traffic
🟠 Orange → Moderate traffic
🔴 Red → Heavy traffic


