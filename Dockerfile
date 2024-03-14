# Use the official Ubuntu Jammy image as the base image
FROM ubuntu:jammy

# Update the package list and install essential packages
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    swig

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip3 install numpy
RUN pip3 install h5py
run pip3 install scipy

# Copy the local code to the container
COPY . .

RUN make python_package_install

# Specify the command to run on container start
CMD ["bash"]

