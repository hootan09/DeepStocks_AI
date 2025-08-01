$uri = "http://localhost:8000/predict"
$body = @"
{
  "ticker": "AAPL",
  "news": "Apple Inc. is reportedly in talks with OpenAI to integrate ChatGPT into iOS 18",
  "price_df": [
    {"Date":"2024-06-03","Open":195.4000,"High":197.5800,"Low":194.1400,"Close":194.4800,"Volume":5.115E+07,"RSI":40.12,"MACD":-0.87,"MACD_Signal":-0.34,"MACD_Hist":-0.53,"BB_Upper":200.12,"BB_Middle":194.83,"BB_Lower":189.54,"VWAP":195.89},
    {"Date":"2024-06-04","Open":194.4900,"High":196.4400,"Low":193.0500,"Close":194.3500,"Volume":4.201E+07,"RSI":39.87,"MACD":-1.05,"MACD_Signal":-0.41,"MACD_Hist":-0.64,"BB_Upper":199.98,"BB_Middle":194.75,"BB_Lower":189.52,"VWAP":195.12},
    {"Date":"2024-06-05","Open":195.2000,"High":196.4500,"Low":193.8900,"Close":194.1700,"Volume":4.204E+07,"RSI":39.52,"MACD":-1.22,"MACD_Signal":-0.48,"MACD_Hist":-0.74,"BB_Upper":199.85,"BB_Middle":194.67,"BB_Lower":189.49,"VWAP":194.87},
    {"Date":"2024-06-06","Open":194.5000,"High":196.5600,"Low":193.9000,"Close":194.4800,"Volume":3.899E+07,"RSI":40.00,"MACD":-1.28,"MACD_Signal":-0.55,"MACD_Hist":-0.73,"BB_Upper":199.77,"BB_Middle":194.61,"BB_Lower":189.45,"VWAP":194.93},
    {"Date":"2024-06-07","Open":194.1400,"High":197.1100,"Low":193.6900,"Close":196.8900,"Volume":5.204E+07,"RSI":45.05,"MACD":-0.93,"MACD_Signal":-0.53,"MACD_Hist":-0.40,"BB_Upper":199.91,"BB_Middle":194.68,"BB_Lower":189.45,"VWAP":196.01},
    {"Date":"2024-06-10","Open":194.5400,"High":195.7700,"Low":192.5100,"Close":193.1200,"Volume":5.697E+07,"RSI":36.80,"MACD":-1.44,"MACD_Signal":-0.64,"MACD_Hist":-0.80,"BB_Upper":199.66,"BB_Middle":194.48,"BB_Lower":189.30,"VWAP":193.87},
    {"Date":"2024-06-11","Open":192.9300,"High":195.4100,"Low":192.5800,"Close":193.1300,"Volume":4.410E+07,"RSI":36.83,"MACD":-1.63,"MACD_Signal":-0.74,"MACD_Hist":-0.89,"BB_Upper":199.48,"BB_Middle":194.35,"BB_Lower":189.22,"VWAP":193.94},
    {"Date":"2024-06-12","Open":193.8500,"High":195.8800,"Low":192.6300,"Close":194.3500,"Volume":4.212E+07,"RSI":39.81,"MACD":-1.58,"MACD_Signal":-0.81,"MACD_Hist":-0.77,"BB_Upper":199.39,"BB_Middle":194.25,"BB_Lower":189.11,"VWAP":194.10},
    {"Date":"2024-06-13","Open":194.7600,"High":196.4500,"Low":193.8900,"Close":194.6800,"Volume":4.356E+07,"RSI":40.60,"MACD":-1.46,"MACD_Signal":-0.85,"MACD_Hist":-0.61,"BB_Upper":199.33,"BB_Middle":194.18,"BB_Lower":189.03,"VWAP":194.82},
    {"Date":"2024-06-14","Open":195.1500,"High":196.4900,"Low":193.6100,"Close":194.1700,"Volume":4.743E+07,"RSI":39.47,"MACD":-1.45,"MACD_Signal":-0.89,"MACD_Hist":-0.56,"BB_Upper":199.24,"BB_Middle":194.09,"BB_Lower":188.94,"VWAP":194.67},
    {"Date":"2024-06-17","Open":193.8900,"High":196.1200,"Low":193.5400,"Close":195.8900,"Volume":4.389E+07,"RSI":43.40,"MACD":-1.18,"MACD_Signal":-0.87,"MACD_Hist":-0.31,"BB_Upper":199.34,"BB_Middle":194.09,"BB_Lower":188.84,"VWAP":195.11},
    {"Date":"2024-06-18","Open":197.4800,"High":198.2300,"Low":195.5300,"Close":195.8900,"Volume":4.800E+07,"RSI":43.40,"MACD":-1.09,"MACD_Signal":-0.88,"MACD_Hist":-0.21,"BB_Upper":199.25,"BB_Middle":194.03,"BB_Lower":188.81,"VWAP":196.95},
    {"Date":"2024-06-20","Open":196.0000,"High":197.5800,"Low":195.0100,"Close":196.3300,"Volume":4.220E+07,"RSI":44.41,"MACD":-0.93,"MACD_Signal":-0.87,"MACD_Hist":-0.06,"BB_Upper":199.20,"BB_Middle":194.02,"BB_Lower":188.84,"VWAP":196.31},
    {"Date":"2024-06-21","Open":196.2100,"High":197.2800,"Low":194.6400,"Close":195.8900,"Volume":5.154E+07,"RSI":43.33,"MACD":-0.93,"MACD_Signal":-0.88,"MACD_Hist":-0.05,"BB_Upper":199.05,"BB_Middle":194.00,"BB_Lower":188.95,"VWAP":195.91},
    {"Date":"2024-06-24","Open":194.2700,"High":196.2900,"Low":193.6200,"Close":195.2200,"Volume":4.229E+07,"RSI":41.84,"MACD":-0.99,"MACD_Signal":-0.89,"MACD_Hist":-0.10,"BB_Upper":198.92,"BB_Middle":193.98,"BB_Lower":189.04,"VWAP":194.87},
    {"Date":"2024-06-25","Open":195.0800,"High":196.4900,"Low":194.1000,"Close":195.8900,"Volume":3.902E+07,"RSI":43.44,"MACD":-0.87,"MACD_Signal":-0.88,"MACD_Hist":0.01,"BB_Upper":198.87,"BB_Middle":193.99,"BB_Lower":189.11,"VWAP":195.30},
    {"Date":"2024-06-26","Open":195.8800,"High":197.5300,"Low":195.2000,"Close":197.2300,"Volume":4.091E+07,"RSI":46.77,"MACD":-0.62,"MACD_Signal":-0.83,"MACD_Hist":0.21,"BB_Upper":198.97,"BB_Middle":194.05,"BB_Lower":189.13,"VWAP":196.50},
    {"Date":"2024-06-27","Open":197.5800,"High":198.3600,"Low":196.0800,"Close":196.8900,"Volume":4.164E+07,"RSI":45.93,"MACD":-0.56,"MACD_Signal":-0.79,"MACD_Hist":0.23,"BB_Upper":198.94,"BB_Middle":194.08,"BB_Lower":189.22,"VWAP":197.09},
    {"Date":"2024-06-28","Open":197.0000,"High":198.1300,"Low":196.4800,"Close":197.2300,"Volume":4.150E+07,"RSI":46.77,"MACD":-0.46,"MACD_Signal":-0.74,"MACD_Hist":0.28,"BB_Upper":198.97,"BB_Middle":194.12,"BB_Lower":189.27,"VWAP":197.25},
    {"Date":"2024-07-01","Open":197.6800,"High":199.6200,"Low":197.2600,"Close":199.2300,"Volume":3.991E+07,"RSI":51.44,"MACD":-0.21,"MACD_Signal":-0.66,"MACD_Hist":0.45,"BB_Upper":199.18,"BB_Middle":194.27,"BB_Lower":189.36,"VWAP":198.69},
    {"Date":"2024-07-02","Open":199.2600,"High":200.2400,"Low":198.4500,"Close":199.8900,"Volume":3.835E+07,"RSI":53.10,"MACD":-0.02,"MACD_Signal":-0.59,"MACD_Hist":0.57,"BB_Upper":199.44,"BB_Middle":194.47,"BB_Lower":189.50,"VWAP":199.47},
    {"Date":"2024-07-03","Open":200.0500,"High":200.4700,"Low":198.8200,"Close":200.0000,"Volume":2.831E+07,"RSI":53.39,"MACD":0.12,"MACD_Signal":-0.51,"MACD_Hist":0.63,"BB_Upper":199.69,"BB_Middle":194.68,"BB_Lower":189.67,"VWAP":199.78},
    {"Date":"2024-07-05","Open":200.4500,"High":201.8400,"Low":199.4600,"Close":200.4500,"Volume":3.546E+07,"RSI":54.48,"MACD":0.25,"MACD_Signal":-0.42,"MACD_Hist":0.67,"BB_Upper":199.99,"BB_Middle":194.93,"BB_Lower":189.87,"VWAP":200.64},
    {"Date":"2024-07-08","Open":200.5800,"High":201.1200,"Low":198.3800,"Close":198.8800,"Volume":4.347E+07,"RSI":50.58,"MACD":0.23,"MACD_Signal":-0.34,"MACD_Hist":0.57,"BB_Upper":200.12,"BB_Middle":195.10,"BB_Lower":190.08,"VWAP":199.77},
    {"Date":"2024-07-09","Open":199.5000,"High":200.7500,"Low":198.4300,"Close":200.4500,"Volume":3.891E+07,"RSI":54.48,"MACD":0.32,"MACD_Signal":-0.26,"MACD_Hist":0.58,"BB_Upper":200.35,"BB_Middle":195.35,"BB_Lower":190.35,"VWAP":199.81},
    {"Date":"2024-07-10","Open":200.6500,"High":202.3900,"Low":200.5400,"Close":202.2500,"Volume":3.812E+07,"RSI":58.78,"MACD":0.50,"MACD_Signal":-0.15,"MACD_Hist":0.65,"BB_Upper":200.68,"BB_Middle":195.64,"BB_Lower":190.60,"VWAP":201.44},
    {"Date":"2024-07-11","Open":202.8000,"High":203.7900,"Low":202.0500,"Close":203.2200,"Volume":3.767E+07,"RSI":61.23,"MACD":0.69,"MACD_Signal":-0.02,"MACD_Hist":0.71,"BB_Upper":201.04,"BB_Middle":195.96,"BB_Lower":190.88,"VWAP":202.88},
    {"Date":"2024-07-12","Open":203.5000,"High":204.4600,"Low":202.9100,"Close":203.8800,"Volume":3.850E+07,"RSI":62.88,"MACD":0.85,"MACD_Signal":0.12,"MACD_Hist":0.73,"BB_Upper":201.44,"BB_Middle":196.31,"BB_Lower":191.18,"VWAP":203.70},
    {"Date":"2024-07-15","Open":204.2700,"High":205.1500,"Low":203.4600,"Close":204.7200,"Volume":3.741E+07,"RSI":64.77,"MACD":1.01,"MACD_Signal":0.25,"MACD_Hist":0.76,"BB_Upper":201.90,"BB_Middle":196.71,"BB_Lower":191.52,"VWAP":204.41},
    {"Date":"2024-07-16","Open":204.2700,"High":205.1500,"Low":203.4600,"Close":204.7200,"Volume":3.741E+07,"RSI":64.77,"MACD":1.01,"MACD_Signal":0.25,"MACD_Hist":0.76,"BB_Upper":201.90,"BB_Middle":196.71,"BB_Lower":191.52,"VWAP":204.41}
  ]
}
"@

$response = Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json"
$response