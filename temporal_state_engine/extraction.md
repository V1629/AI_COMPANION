flowchart TD

A[1 Week Chat Logs] --> B[Language Detection Layer]
B --> C[Text Normalization + Hinglish Handling]

C --> D[Message Embedding Model<br/>Multilingual E5 / MPNet]
C --> E[Emotion Classifier<br/>XLM-R / GoEmotions]
C --> F[Sentiment Model<br/>CardiffNLP]

D --> G[Message Feature Vector]
E --> G
F --> G

G --> H[Temporal Message Store<br/>Vector DB + Metadata]

H --> I[Sliding Window Analyzer<br/>Last N Messages]
I --> J[Short Term State<br/>Volatility + Dominant Emotion]

H --> K[Daily Aggregator]
K --> L[Mid Term State<br/>Trend + EMA + Stability]

H --> M[Weekly Sequence Model<br/>Hierarchical Transformer]
M --> N[Long Term State<br/>Baseline Mood + Personality Drift]

J --> O[State Manager]
L --> O
N --> O

O --> P[Adaptive Response Generator]
