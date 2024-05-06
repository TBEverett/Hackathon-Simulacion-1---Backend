# HackTheChallenge Backend

Recopilación de dos carpetas colaborativas mediante el siguiente árbol

```
.
├── chatbot
│   ├── category.csv
│   ├── chatbot.ipynb
│   └── PyTorch-Chatbot
│       ├── app.py
|       ...
├── data-model
│   ├── german.csv
│   ├── modelAPI.py
│   ├── model_dump.joblib
│   └── simul1.ipynb
├── LICENSE
└── README.md
```

* **`chatbot` folder**: Dos alternativas para chatbot. En `chatbot.ipynb` está la carpeta con `DeBERTa` backbone y head personalizado en base a `category.csv`. En la carpeta `PyTorch-Chatbot`, se encuentra una versión minimalista de una red neuronal para chatbot (a testear).

* **`data-model` folder**: Modelo encargado de predicción de data desbalanceada