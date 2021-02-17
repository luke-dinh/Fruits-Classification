#Set up images
FROM python:3

#Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

#Install all the requirements
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

#Make port 8000 available
EXPOSE 8000

#Run app.py 
CMD ["python", "app.py"]