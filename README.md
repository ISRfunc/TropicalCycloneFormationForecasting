# Tropical Cyclone Formation Forecasting

Cài môi trường conda: 
```
conda create -n cyclone_forecasting python=3.9
conda activate cyclone_forecasting
pip install -r requirements.txt
```

Cài đặt biến môi trường cho conda env: 
```
conda env config vars set PYTHONPATH="/path/to/your/project"
```

Chỉnh đường dẫn data trong file configs.yaml: 
```
data:
    root: "/path/to/your/data"
```

Chạy thử code dataset: 
```
python test.py
```

Chạy code lấy mean và std của các giá trị cho mỗi biến bằng file generateMeanAndStd:
```
python preprocessing/generateMeanAndStd.py
```

Chạy code lấy outlier của các giá trị cho mỗi biến bằng file generateFillValues:
```
python preprocessing/generateFillValues.py
```
