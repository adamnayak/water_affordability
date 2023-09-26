# water_affordability
This public repository contains all code for the project entitled: "Socio-hydrological impacts of rate design on water affordability during drought"

## Contents
### Data
estimated_household_demand_Optimize.final is the Excel spreadsheet that calculates all inputs to the model for water demand based upon the monthly cyclostationary water use and household size. These are the only direct data inputs into the model, as the rest of the results are simulated. You will find the rate scheme optimization in this file as well, which uses Excel Solver to reoptimize rates to fit the rate design constraints for both an increasing block rate and decreasing block rate using Santa Cruz's current rate scheme.

### Code & Model
final_model.py is the final affordability model that can be run to generate water bill estimates for Santa Cruz, CA based upon varying rate design configurations. All inputs to the model are specifies in the "model inputs" region of the Python file. Outputs include timeseries data for household water use, utility water use, household water bills by income group, and utility costs. A summary across the drought period is provided in the "final_summary.xlsx" file.

### Contact Me!
If you have general questions about the code, model, or data please feel free to reach out and I am always happy to try to do my best to help out. If you're interested in using similar method or working on a new project, I am always looking to collaborate and am happy to contribute more broadly! Email is always in flux - but try me at adam.nayak@columbia.edu, adam.nayak@alumni.stanford.edu, adamnayak1@gmail.com, or feel free to ping me on LinkedIn.
