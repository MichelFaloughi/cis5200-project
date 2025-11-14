Hola chicos I hope you're well

Here's a quick guide to stuff we'll do:

I need everyone of you to make an account here: https://cds.climate.copernicus.eu/
and retrieve your API key
Once you do that, you will have to create a .cdsapirc file (in your home directory), that looks like this 
url: https://cds.climate.copernicus.eu/api
key: <YOUR API KEY HERE>

so that you'll be able to query data from the Copernicus Climate Data Store (CDS), not to be confused with the Coperniucs Climate Change Service (CDS)
to see how to query data from CDS, see the playground.ipynb notebook, where I'm trying a bunch of stuff

---

Questions:
- How to make the compute not super expensive ? Should we focus on a smaller specific geography ?
- What do we mean by 'baseline method' ? Is it the most accurate method ? How would we know before evaluating all of them ?
- Wouldn't it be hard to find 5 machine learning methods that are suitable for our task ? What if there are only like 2 and the others aren't suitable ? Could we still use them and say why they are unsuitable ?

Concerns:
- Make sure one method is novel: a non-standard training loss, transfer learning, regularization, or such
- Cite packages


TO DO:
- Check the literature, the current state of the art, etc
- Pre-process data.
- Build 5 models
- Build common evaluation metrics on which to test each model. Figures, etc. Like a 5-subplot plot for each. The losses, the results, etc


Maybe as a step one, I can build an RNN from start to finish, including data preprocessing and post processing, then use that for all 4 other models


---
TO DO: Define the task formally:
"We predict <TIME HORIZON> 10m wind speed using past 24h ERA5 <VARIABLES> at <LOCATION>
time horizon: 1-hour ? Maybe do autoregressive rollouts ?





Data: TO DO:
- Figure out what subset data of ERA5 you want
- Figure out if your data needs to be the same for the 5 models ?

Eval: TO DO:
- Figure out what evaluation metrics you want per model
- Create a standardized evaluation 'score card' for all models
- Then write some code to show the results side by side

Models: TO DO:
- Figure out what 5 models we want (RNN ? Ensemble ?)
- Figure out what the baseline model is


Hence, the next steps are:
- Define the task formally in your notebook: “We predict next-hour 10m wind speed using past 24h ERA5 variables at location X.”
- Build the dataset (single DataFrame used by all models).
- Implement Baseline (Persistence) and evaluate it.
- Implement Linear Regression + Random Forest + XGBoost → plug into shared eval.
- Implement MLP.
- (If time) Implement LSTM or 1D CNN as the 5th model.
- Create the scorecard table + 1–2 plots.
- Mirror this into Overleaf sections (Models + Evaluation).




Latest:
- I'm having trouble downloading the dataset from cdsapi... my request are too large, I need to split them into years or months, etc. then concatenate everything...