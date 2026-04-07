FROM python:3.9-slim

WORKDIR /app

# Sabse pehle requirements file copy karo
COPY requirements.txt .

# Zaruri libraries install karo
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saara code copy karo
COPY . .

# Scaler ko inference.py chahiye, toh wahi run karo
CMD ["python", "inference.py"]