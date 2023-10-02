# Use the official Ubuntu as a parent image
FROM ubuntu:latest
# Avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
# Update and install necessary packages, including cron
RUN apt-get update -y && \
    apt-get install -y python3.9 python3-pip cron && \
    apt-get clean
# Add crontab file in the cron directory
ADD crontab /etc/cron.d/hello-cron
# Create a directory for your project (change "crm-ml-playground" to your project name)
WORKDIR /crm-ml-playground
# Copy the current directory contents into the container at /crm-ml-playground
ADD . /crm-ml-playground
# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/hello-cron
# Apply cron job
RUN crontab /etc/cron.d/hello-cron
# Create the log file to be able to run tail
 RUN touch /var/log/cron.log

 # Start cron and your application in the foreground
CMD env > /etc/environment && cron && tail -f /var/log/cron.log
