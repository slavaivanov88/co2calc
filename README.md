# co2calc
Missing value generator for co2data.fi

<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Final project for the Building AI course
![Screenshot](https://github.com/user-attachments/assets/c39b2f71-d524-4773-a195-1e987333a53b)


## Summary

It uses Random Forest algorithm to fill in missing co2 valuese. A starting point is a Finnish National Emissions Database available at co2data.fi
I used a small bit of it that covers concrete pipes and managed to predict missing values(custom diameters or shapes) with 97% accuracy.
I tested different labelling and prediction techniques and so far the best one was Random Forest Regressor combined wiht One Hot Encoder. 
A python example is attached

## Background

Missing values is a problem I faced a few times at my actula work so this little algorithm is my humble attempt to solve it.

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
