# co2calc
Missing value generator for co2data.fi

<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Final project for the Building AI course



## Summary

In my final assignment I used Random Forest algorithm to fill in/predict co2 values  that are currently missing from the database. A starting point is a Finnish National Emissions Database available at co2data.fi, namely its INFRA segment that is used as a reference for co2 emission calculations in civil construction projects.
I used a small bit of it that covers concrete pipes  because it seemed to be an easy task, and the sanity check was easy to perform. The database covers all popular sizes, e.g. 300, 400 or 500 mm but middle sizes such as 350 are currently missing. First, I tried to predict them and when I made sure the model works, I performed the sanity check. I simply deleted a couple of known values and tried to predict them and then compare predicted and actual values. I the sanity check did show  97% accuracy.
I also compared different labelling and prediction techniques and so far, the best one was Random Forest Regressor combined with One Hot Encoder.  A python file in the repository illustrates that combination of instruments.
The problem derives from my actual work. Missing values is a problem I faced a few times so this little algorithm is my humble attempt to solve it and make my routines easier.

![Screenshot](https://github.com/user-attachments/assets/c39b2f71-d524-4773-a195-1e987333a53b)


## Data sources and AI methods

Attached csv is generated from original database json, with some mino modifications:

| Type      | Material | Diameter | Class | Description | CO2 kg/m |
| --------- | -------- | -------- |------ | ----------- | -------- |
| Pipe      | Concrete | 1000     | Br    | round       | 215      |
| Pipe      | Concrete | 1000     | Dr    | footed      | 217      |
| etc...    | ...      | ...      | ...   | ...         | ...      |


## What next?

The model can be developed further to cover other element and material types
https://co2data.fi/infra/


## Acknowledgments
![ActualPredictedHot](https://github.com/user-attachments/assets/a5db6d6b-ae37-49fd-8514-f131ba73d6d6)
