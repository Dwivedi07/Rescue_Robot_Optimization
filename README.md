# Rescue_Robot_Optimization

# Coverage Maximization and Sensing for UAV Rescue Operations

## Project Description

This project, undertaken at IIT Bombay under the guidance of Prof. Avinash Bharadwaj, focuses on optimizing Unmanned Aerial Vehicle (UAV) networks for rescue operations in non-convex domains. The work involves:

- **Mathematical Formulation:** The Search and Rescue (SaR) problem is mathematically formulated as an optimization problem to minimize time.
- **Genetic Algorithm:** A novel genetic algorithm-based method is implemented to maximize the area of coverage for UAV surveillance tasks in non-convex and disconnected placement domains.
- **Image Processing Framework:** An image processing framework, utilizing the Ramer-Douglas-Peucker algorithm, is devised to approximate region boundaries to polygonal sets, ensuring the autonomous functioning of the algorithm.

## Publication

The results from this project were presented in the paper:

**Arpit Dwivedi, Chinmay Pimpalkhare.** "Coverage Maximization for UAV Surveillance on Non-convex Domains using Genetic Algorithm." Paper presented virtually at the [6th National Conference on Multidisciplinary Design, Analysis and Optimization](https://event.iitg.ac.in/ncmdao/), IIT Guwahati, India.

## Folder Structure

The repository contains the following main folders and files:

- **Genetic Algo:** Contains the implementation of the genetic algorithm.
- **Images:** Contains image data used for processing.
- **Python Files:** Contains various Python scripts related to the project.

## Running the Pipeline

To run the pipeline, use the following command in your terminal:

```bash
python3 Rescue_optimization.py --calamity <Calamity_type> --rescuerobots <Number_of_rescue_robots> --typerr <Types_of_Robots> --sensorrobots <Number_of_sensor_robots> --typesr <types_of_sensor_robots>


You can adjust the placeholders and content to match the specific details of your project.

In order to run the pipeline, write the following in terminal:

python3 Rescue_optimization.py --calamity <Calamity_type> --rescuerobots <Number_of_rescue_robots>> --typerr <Types_of_Robots> --sensorrobots <Number_of_sensor_robots> --typesr <types_of_sensor_robots>
