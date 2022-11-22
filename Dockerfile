# Must use a Cuda version 10+
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git && apt-get install -y gcc

# Clone audio2head
# RUN git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model
RUN git clone https://github.com/wangsuzhen/Audio2Head

# Change into the repo dir
WORKDIR "/Audio2Head"
# ADD videos videos
# copy corrected requirements.txt
# RUN rm requirements.txt
# COPY requirements.txt requirements.txt

# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
#NOTE: I was not able to get these to install with the correct versions from requirements.txt
# RUN pip3 install torch===1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install torchvision===0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html


# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
# (in this case we have a python script)
ADD download_checkpoints.sh .
RUN bash download_checkpoints.sh

# # Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py

# CMD bash